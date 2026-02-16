import simpy
import pandas as pd
from environment.simulator.core.setup import SimulationSetup
from environment.entities.Case import Case
import json as js

class SimulatorEngine:
    def __init__(self, simulationSetup: SimulationSetup):
        self.start_timestamp = pd.to_datetime(simulationSetup.start_timestamp)
        self.setup = simulationSetup
        self.is_rl_mode = False
        
        # Simple Cache: Calculate these once so we don't repeat work in loops
        self._activities = self._get_all_activities_list()
        self._resources = self.setup.resource_policy.resources if hasattr(self.setup.resource_policy, 'resources') else []
        
        self.reset()

    def reset(self, max_cases=None):
        self.env = simpy.Environment()
        self.event_log = []
        self.active_cases = 0
        self.no_more_arrivals = False
        self.current_activities = {}
        self.last_activities = {} 
        self.waiting_requests = {}
        self.completed_cases = [] 
        self.pending_decisions = [] 

        self.simpy_resources = {
            r.id: simpy.Resource(self.env, capacity=r.capacity)
            for r in self._resources
        }

        self.all_done = self.env.event()
        self.decision_event = self.env.event() # Signals when RL intervention is needed
        
        self.env.process(self.case_generator(max_cases))

    def simulate(self, until: float = None, max_cases: int = None):
        """Standard SimPy simulation (Fast)"""
        self.is_rl_mode = False 
        self.reset(max_cases=max_cases)
        if until is not None:
            self.env.run(until=until)
        else:
            self.env.run(until=self.all_done)
        return self.event_log

    def run_until_decision(self):
        """RL Simulation (Optimized Fast-Forward)"""
        self.is_rl_mode = True 
        
        # If no one is waiting for a decision, let SimPy run at full speed
        # until a decision event is triggered or the simulation finishes.
        if not self.pending_decisions and not self.all_done.triggered:
            self.env.run(until=simpy.events.AnyOf(self.env, [self.decision_event, self.all_done]))

        completed = self.completed_cases
        self.completed_cases = []
        return completed

    def get_case_needing_decision(self):
        if self.pending_decisions:
            return self.pending_decisions[0]["case"]
        return None

    def apply_decision(self, activity, resource):
        """Resumes a paused case with the agent's choice"""
        if not self.pending_decisions:
            return False
            
        decision = self.pending_decisions.pop(0)
        decision["callback"].succeed((activity, resource))
        
        # Reset the signal so we can wait for the next one
        if not self.pending_decisions:
            self.decision_event = self.env.event()
            
        return True

    def process_case(self, case: Case):
        case.start_time = self.env.now
        self.active_cases += 1
        
        while True:
            if self.is_rl_mode:
                # 1. Pause point for RL
                decision_fulfilled = self.env.event()
                self.pending_decisions.append({"case": case, "callback": decision_fulfilled})
                
                if not self.decision_event.triggered:
                    self.decision_event.succeed()
                
                activity, resource = yield decision_fulfilled
            else:
                # 2. Automatic path for normal simulation
                last_act = self.last_activities.get(case.case_id)
                activity = self.setup.routing_policy.get_next_activity(case, last_act)
                if activity is None: break
                resource = self.setup.resource_policy.select_resource(activity, case)
            
            if activity is None: break

            self.current_activities[case.case_id] = activity
            yield self.env.process(self.execute_activity(case, activity, resource))
            
            self.last_activities[case.case_id] = activity
            if case.case_id in self.current_activities:
                del self.current_activities[case.case_id]

        self.active_cases -= 1
        case.end_time = self.env.now
        case.cycle_time = case.end_time - case.start_time
        self.completed_cases.append(case)
        self._check_termination()

    def execute_activity(self, case: Case, activity, resource):
        simpy_resource = self.simpy_resources[resource.id]
        
        # Calendar wait logic
        next_time = self.setup.calendar_policy.next_working_time(self.env.now)
        if next_time > self.env.now:
            yield self.env.timeout(next_time - self.env.now)

        process = self.env.active_process
        self.waiting_requests[process] = (case, activity)

        with simpy_resource.request() as req:
            yield req
            del self.waiting_requests[process]

            duration = self.setup.processing_time_policy.get_activity_duration(activity, resource)
            yield self.env.timeout(duration)
            
            self.event_log.append({
                "case": case.case_id, "activity": activity, "resource": resource.id,
                "start": self.env.now - duration, "end": self.env.now
            })
    
    def state(self):
        """Minimal state dictionary (No Pandas)"""
        res_occ = {rid: {"in_use": r.count, "capacity": r.capacity, "waiting": len(r.queue)} 
                   for rid, r in self.simpy_resources.items()}
        
        # Calculate activity waiting counts
        act_wait = {}
        for _, (_, act) in self.waiting_requests.items():
            act_wait[str(act)] = act_wait.get(str(act), 0) + 1
        for req in self.pending_decisions:
            a_str = str(req.get("activity", "PENDING"))
            act_wait[a_str] = act_wait.get(a_str, 0) + 1

        return {
            "resource_occupancy": res_occ,
            "activities_waiting": act_wait,
            "total_waiting": len(self.waiting_requests) + len(self.pending_decisions),
            "internal_time": self.env.now
        }

    def _get_all_activities_list(self):
        if hasattr(self.setup.routing_policy, 'probabilities'):
            acts = set()
            for src, targets in self.setup.routing_policy.probabilities.items():
                if src: acts.add(src)
                for tgt in targets: 
                    if tgt: acts.add(tgt)
            return sorted(list(acts)) + [None]
        return [None]

    @property
    def all_activities(self): return self._activities
    @property
    def all_resources(self): return self._resources
    @property
    def num_activities(self): return len(self._activities)
    @property
    def num_resources(self): return len(self._resources)

    def case_generator(self, max_cases=None):
        case_count = 0
        while max_cases is None or case_count < max_cases:
            yield self.env.timeout(self.setup.arrival_policy.get_next_arrival_time())
            case_count += 1
            self.env.process(self.process_case(Case(case_id=f"case_{case_count}", events=[])))
        self.no_more_arrivals = True
        self._check_termination()

    def _check_termination(self):
        if self.no_more_arrivals and self.active_cases == 0:
            if not self.all_done.triggered: self.all_done.succeed()
