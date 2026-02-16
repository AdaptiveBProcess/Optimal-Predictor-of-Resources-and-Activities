import simpy
import pandas as pd
from environment.simulator.core.setup import SimulationSetup
from environment.entities.Case import Case
import json as js

class SimulatorEngine:
    def __init__(self, simulationSetup: SimulationSetup):
        self.start_timestamp = pd.to_datetime(simulationSetup.start_timestamp)
        self.setup = simulationSetup
        self.reset()

    def reset(self, max_cases=None):
        self.env = simpy.Environment()
        self.event_log = []

        # termination bookkeeping
        self.active_cases = 0
        self.no_more_arrivals = False
        self.current_activities = {}
        self.waiting_requests = {}
        self.completed_cases = [] 
        self.pending_decisions = [] # Decisions waiting for RL (activity, resource)

        self.simpy_resources = {
            r.id: simpy.Resource(self.env, capacity=r.capacity)
            for r in self.setup.resource_policy.resources
        }

        self.all_done = self.env.event()
        self.decision_event = self.env.event()
        
        self.env.process(self.case_generator(max_cases))
        self.max_utilization = 0

    def simulate(self, until: float = None, max_cases: int = None):
        self.reset(max_cases=max_cases)

        if until is not None:
            self.env.run(until=until)
        else:
            self.env.run(until=self.all_done)

        return self.event_log

    def has_pending_decision(self):
        return len(self.pending_decisions) > 0

    def run_until_decision(self):
        """
        Runs the simulation until a decision is needed or simulation finishes.
        """
        while not self.has_pending_decision() and not self.all_done.triggered:
            if not self.env._queue:
                break
            self.env.step()

        completed = self.completed_cases
        self.completed_cases = []
        return completed

    def get_case_needing_decision(self):
        if self.pending_decisions:
            return self.pending_decisions[0]["case"]
        return None

    def apply_decision(self, activity, resource):
        """
        Applies a (activity, resource) decision from an external agent.
        If activity is None, the case is considered finished.
        """
        if not self.pending_decisions:
            return False
            
        decision = self.pending_decisions.pop(0)
        decision["callback"].succeed((activity, resource))
        
        # If no more pending decisions, reset the event for the next run_until_decision
        if not self.pending_decisions:
            self.decision_event = self.env.event()
            
        return True

    def case_generator(self, max_cases=None):
        case_count = 0

        while max_cases is None or case_count < max_cases:
            interarrival = self.setup.arrival_policy.get_next_arrival_time()
            yield self.env.timeout(interarrival)

            case_count += 1
            case = Case(case_id=f"case_{case_count}", events=[])
            self.env.process(self.process_case(case))

        self.no_more_arrivals = True
        self._check_termination()

    def process_case(self, case: Case):
        case.start_time = self.env.now
        self.active_cases += 1
        
        while True:
            # Pause and ask for (Activity, Resource)
            decision_fulfilled = self.env.event()
            self.pending_decisions.append({
                "case": case,
                "callback": decision_fulfilled
            })
            
            # Signal that a decision is needed
            if not self.decision_event.triggered:
                self.decision_event.succeed()
            
            # Wait for intervention
            activity, resource = yield decision_fulfilled
            
            if activity is None:
                # Case termination signaled by agent
                break

            self.current_activities[case.case_id] = activity
            yield self.env.process(self.execute_activity(case, activity, resource))
            
            if case.case_id in self.current_activities:
                del self.current_activities[case.case_id]
            
            # Check for max utilization (optional monitoring)
            avg_utilization = sum(res.count for res in self.simpy_resources.values()) / sum(res.capacity for res in self.simpy_resources.values()) if sum(res.capacity for res in self.simpy_resources.values()) > 0 else 0
            if avg_utilization > self.max_utilization:
                self.max_utilization = avg_utilization
                js.dump(self.state(), open("state_at_max_utilization.json", "w"), indent=4)

        self.active_cases -= 1
        case.end_time = self.env.now
        case.cycle_time = case.end_time - case.start_time
        self.completed_cases.append(case)
        self._check_termination()
    
    def execute_activity(self, case: Case, activity, resource):
        simpy_resource = self.simpy_resources[resource.id]

        # calendar wait
        next_time = self.setup.calendar_policy.next_working_time(self.env.now)
        if next_time > self.env.now:
            yield self.env.timeout(next_time - self.env.now)

        process = self.env.active_process
        self.waiting_requests[process] = (case, activity)

        with simpy_resource.request() as req:
            yield req

            del self.waiting_requests[process]

            start = self.env.now
            duration = self.setup.processing_time_policy.get_activity_duration(activity, resource)
            yield self.env.timeout(duration)
            end = self.env.now

 
            self.event_log.append({
                "case": case.case_id,
                "activity": activity,
                "resource": resource.id,
                "start": self.start_timestamp + pd.to_timedelta(start, unit=self.setup.time_unit),
                "end": self.start_timestamp + pd.to_timedelta(end, unit=self.setup.time_unit)
            })
    
    def _check_termination(self):
        if self.no_more_arrivals and self.active_cases == 0:
            if not self.all_done.triggered: 
                self.all_done.succeed()


    @property
    def all_activities(self):
        # Derive all activities from the routing policy
        if hasattr(self.setup.routing_policy, 'probabilities'):
            activities = set()
            for src, targets in self.setup.routing_policy.probabilities.items():
                if src is not None:
                    activities.add(src)
                for tgt in targets:
                    if tgt is not None:
                        activities.add(tgt)
            return sorted(list(activities))
        return []

    @property
    def all_resources(self):
        # Derive all resources from the resource policy
        if hasattr(self.setup.resource_policy, 'resources'):
            return self.setup.resource_policy.resources
        return []

    @property
    def num_activities(self):
        return len(self.all_activities)

    @property
    def num_resources(self):
        return len(self.all_resources)

    def state(self):
        resource_occupancy = {}
        total_cases_processing = 0
        
        # Calculate occupancy and processing counts
        for res_id, simpy_res in self.simpy_resources.items():
            resource_occupancy[res_id] = {
                "in_use": simpy_res.count,
                "capacity": simpy_res.capacity,
                "utilization": simpy_res.count / simpy_res.capacity if simpy_res.capacity > 0 else 0,
                "waiting": len(simpy_res.queue) # Still useful to have per-resource wait count
            }
            total_cases_processing += simpy_res.count
        
        # Calculate waiting activities and total waiting count
        activities_with_waiting_cases = {}
        
        # 1. Cases waiting for SimPy resources (already allocated but busy)
        for process, (case, activity) in self.waiting_requests.items():
            activity_str = str(activity)
            activities_with_waiting_cases[activity_str] = activities_with_waiting_cases.get(activity_str, 0) + 1
        
        # 2. Cases waiting for RL decision
        for req in self.pending_decisions:
            activity_str = str(req["activity"])
            activities_with_waiting_cases[activity_str] = activities_with_waiting_cases.get(activity_str, 0) + 1

        total_cases_waiting = len(self.waiting_requests) + len(self.pending_decisions)

        # Active cases and their current activities
        active_cases_status = {
            case_id: str(activity) for case_id, activity in self.current_activities.items()
        }

        # Day of week / time of day for calendar effects
        current_absolute_time = self.start_timestamp + pd.to_timedelta(self.env.now, unit=self.setup.time_unit)
        time_info = {
            "current_time_internal_units": self.env.now,
            "current_absolute_timestamp": str(current_absolute_time),
            "day_of_week": current_absolute_time.day_name(),
            "time_of_day": str(current_absolute_time.time())
        }

        return {
            "resource_occupancy": resource_occupancy,
            "total_cases_processing": total_cases_processing,
            "total_cases_waiting": total_cases_waiting,
            "activities_with_waiting_cases": activities_with_waiting_cases,
            "active_cases_status": active_cases_status,
            "time_info": time_info
        }
