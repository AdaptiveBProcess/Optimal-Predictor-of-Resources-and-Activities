import simpy
import pandas as pd
from environment.simulator.core.setup import SimulationSetup
from environment.entities.Case import Case

class SimulatorEngine:
    def __init__(self, simulationSetup: SimulationSetup):
        self.setup = simulationSetup
        self.start_timestamp = pd.to_datetime(simulationSetup.start_timestamp)

        self.routing_policy = simulationSetup.routing_policy
        self.resource_policy = simulationSetup.resource_policy
        self.processing_policy = simulationSetup.processing_time_policy
        self.calendar_policy = simulationSetup.calendar_policy
        self.arrival_policy = simulationSetup.arrival_policy
        self.event_log = []

        # termination bookkeeping
        self.active_cases = 0
        self.no_more_arrivals = False

    def simulate(self, until: float = None, max_cases: int = None):
        self.env = simpy.Environment()

        self.simpy_resources = {
            r.id: simpy.Resource(self.env, capacity=r.capacity)
            for r in self.resource_policy.resources
        }

        self.all_done = self.env.event()

        self.env.process(self.case_generator(max_cases))

        if until is not None:
            self.env.run(until=until)
        else:
            self.env.run(until=self.all_done)

        return self.event_log
    
    def case_generator(self, max_cases=None):
        case_count = 0

        while max_cases is None or case_count < max_cases:
            interarrival = self.arrival_policy.get_next_arrival_time()
            yield self.env.timeout(interarrival)

            case_count += 1
            case = Case(case_id=f"case_{case_count}", events=[])
            self.env.process(self.process_case(case))

        self.no_more_arrivals = True
        self._check_termination()

    def process_case(self, case: Case):
        self.active_cases += 1

        activity = self.routing_policy.get_next_activity(case)
        while activity is not None:
            yield self.env.process(self.execute_activity(case, activity))
            activity = self.routing_policy.get_next_activity(case, activity)

        self.active_cases -= 1
        self._check_termination()
    
    def execute_activity(self, case: Case, activity):
        resource = self.resource_policy.select_resource(activity, case)
        simpy_resource = self.simpy_resources[resource.id]

        # calendar wait
        next_time = self.calendar_policy.next_working_time(self.env.now)
        if next_time > self.env.now:
            yield self.env.timeout(next_time - self.env.now)

        with simpy_resource.request() as req:
            yield req

            start = self.env.now
            duration = self.processing_policy.get_activity_duration(activity, resource)
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





