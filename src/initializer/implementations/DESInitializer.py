import pandas as pd


from environment.simulator.core.log_names import LogColumnNames
from environment.simulator.core.process_model import ProcessModel
from environment.simulator.models.distributions.SkillBasedResourcePolicy import SkillBasedResourcePolicy
from environment.simulator.models.distributions.EmpiricalArrivalPolicy import EmpiricalArrivalPolicy
from environment.simulator.models.distributions.WeeklyCalendarPolicy import WeeklyCalendarPolicy
from environment.simulator.policies import ArrivalPolicy, CalendarPolicy, ProcessingTimePolicy, RoutingPolicy
from environment.simulator.policies.WaitingTImePolicy import WaitingTimePolicy
from initializer.Initializer import Initializer
from environment.simulator.core.setup import SimulationSetup

from environment.entities.Resource import Resource
    
from collections import defaultdict, Counter

import numpy as np

from environment.simulator.models.distributions.ProbabilisticRoutingPolicy import ProbabilisticRoutingPolicy
from environment.simulator.models.distributions.EmpiricalProcessingTimePolicy import EmpiricalProcessingTimePolicy


class DESInitializer(Initializer):

    def build(self, log, log_names: LogColumnNames, start_timestamp: str, time_unit: str) -> SimulationSetup:
        self.log_names = log_names

        # activities = self._extract_activities(log)
        # starts, ends = self._extract_start_end_activities(log)
        # process_model = ProcessModel(
        #     activities=activities,
        #     start_activities=starts,
        #     end_activities=ends,
        # )


        routing = self._build_routing_policy(log)
        processing_times = self._build_processing_time_policy(log, time_unit)
        waiting_times = self._build_waiting_time_policy(log, time_unit)
        calendar = self._build_calendar_policy(log, start_timestamp)
        arrivals = self._build_arrival_policy(log, time_unit)
        resources = self._build_resource_policy(log)
        return SimulationSetup(
            time_unit=time_unit,
            start_timestamp=start_timestamp,
            routing_policy=routing,
            waiting_time_policy=waiting_times,
            processing_time_policy=processing_times,
            calendar_policy=calendar,
            arrival_policy=arrivals, 
            resource_policy=resources
        )


    def _build_routing_policy(self, log) -> RoutingPolicy:
        # counts[current_activity][next_activity] = frequency
        counts = defaultdict(Counter)

        for case_id, group in log.groupby(self.log_names.case_id):
            prev = None  # start of case
            for _, event in group.iterrows():
                act = event[self.log_names.activity]
                counts[prev][act] += 1
                prev = act

            last_act = prev  # end of case
            counts[last_act][None] += 1  # terminal state
            # ensure terminal state exists
            counts[prev]  # creates empty Counter if missing

        # normalize into probabilities
        transition_probs = {}

        for current, counter in counts.items():
            total = sum(counter.values())
            if total == 0:
                transition_probs[current] = {}
            else:
                transition_probs[current] = {
                    nxt: cnt / total for nxt, cnt in counter.items()
                }

        return ProbabilisticRoutingPolicy(transition_probs)

    def _build_processing_time_policy(self, log, time_unit: str) -> ProcessingTimePolicy:
        durations_by_activity = defaultdict(list)

        for _, row in log.iterrows():
            start = pd.to_datetime(row[self.log_names.start_timestamp])
            end = pd.to_datetime(row[self.log_names.end_timestamp])
            if pd.isna(start) or pd.isna(end):
                continue

            duration = (end - start).total_seconds()  # convert to seconds
            if time_unit == "minutes":
                duration /= 60
            elif time_unit == "hours":
                duration /= 3600
            if duration < 0:
                continue

            act = row[self.log_names.activity]
            durations_by_activity[act].append(duration)

        return EmpiricalProcessingTimePolicy(durations_by_activity)


    def _build_waiting_time_policy(self, log, time_unit: str) -> WaitingTimePolicy:
        pass

    def _build_calendar_policy(self, log, start_timestamp: str) -> CalendarPolicy:
        # 7 days x 24 hours
        calendar_counts = np.zeros((7, 24))

        # count event starts
        for _, row in log.iterrows():
            ts = pd.to_datetime(row[self.log_names.start_timestamp])

            if pd.isna(ts):
                continue

            d = ts.weekday()  # 0 = Monday  
            h = ts.hour       # 0..23

            calendar_counts[d, h] += 1

        # normalize per weekday (optional but stabilizes thresholding)
        with np.errstate(divide="ignore", invalid="ignore"):
            calendar_probs = calendar_counts / calendar_counts.sum(axis=1, keepdims=True)
            calendar_probs = np.nan_to_num(calendar_probs)

        # thresholding to infer availability
        percentile = 25  # you can make this configurable
        threshold = np.percentile(calendar_probs.flatten(), percentile)

        availability = calendar_probs > threshold

        return WeeklyCalendarPolicy(availability, start_timestamp)


    def _build_arrival_policy(self, log, time_unit: str) -> ArrivalPolicy:
        # get first event per case
        case_starts = (
            log.groupby(self.log_names.case_id)[self.log_names.start_timestamp]
            .min()
            .sort_values()
        )
        # convert to timestamps
        start_times = pd.to_datetime(case_starts).astype("int64") / 1e9


        # compute inter-arrival times
        inter_arrivals = np.diff(start_times)

        if time_unit == "minutes":
            inter_arrivals /= 60
        elif time_unit == "hours":
            inter_arrivals /= 3600

        # filter invalid values
        inter_arrivals = inter_arrivals[inter_arrivals > 0]

        return EmpiricalArrivalPolicy(inter_arrivals.tolist()) 
    
    def _build_resource_policy(self, log):
        sillks = defaultdict(set)
        for _, row in log.iterrows():
            resource_name = row[self.log_names.resource]
            activity = row[self.log_names.activity]
            sillks[resource_name].add(activity)
        resources_names = self._extract_resources(log)
        resources = []
        for r_name in resources_names:
            resource = Resource(id=r_name, skills=sillks[r_name])
            resources.append(resource)
        return SkillBasedResourcePolicy(resources=resources)

    def _extract_activities(self, log):
        activities = set(log[self.log_names.activity].unique())
        return activities

    def _extract_start_end_activities(self, log):
        starts = set()
        ends = set()
        grouped = log.groupby(self.log_names.case_id)
        for case_id, group in grouped:
            sorted_group = group.sort_values(by=self.log_names.start_timestamp)
            starts.add(sorted_group.iloc[0][self.log_names.activity])
            ends.add(sorted_group.iloc[-1][self.log_names.activity])
        return starts, ends
    
    def _extract_resources(self, log):
        resources = set(log[self.log_names.resource].unique())
        return resources