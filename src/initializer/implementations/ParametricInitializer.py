import pandas as pd
import numpy as np

from initializer.Initializer import Initializer
from environment.simulator.core.setup import SimulationSetup
from environment.simulator.core.log_names import LogColumnNames

from environment.simulator.policies import ArrivalPolicy, CalendarPolicy, ProcessingTimePolicy, RoutingPolicy
from environment.simulator.policies.WaitingTImePolicy import WaitingTimePolicy

# Reusing empirical policies for now
from environment.simulator.models.empirical.ProbabilisticRoutingPolicy import ProbabilisticRoutingPolicy
from environment.simulator.models.empirical.SkillBasedResourcePolicy import SkillBasedResourcePolicy
from environment.simulator.models.empirical.WeeklyCalendarPolicy import WeeklyCalendarPolicy

# New parametric policies
from environment.simulator.models.distributions.ExponentialArrivalPolicy import ExponentialArrivalPolicy
from environment.simulator.models.distributions.NormalProcessingTimePolicy import NormalProcessingTimePolicy

from initializer.implementations.DESInitializer import DESInitializer

from collections import defaultdict # Import defaultdict at the top


class ParametricInitializer(DESInitializer): # Inherit from DESInitializer to reuse common methods

    def build(self, log, log_names: LogColumnNames, start_timestamp: str, time_unit: str) -> SimulationSetup:
        self.log_names = log_names

        routing = self._build_routing_policy(log) # Reuses from DESInitializer
        processing_times = self._build_processing_time_policy(log, time_unit) # Overridden
        waiting_times = self._build_waiting_time_policy(log, time_unit) # Reuses from DESInitializer (passes for now)
        calendar = self._build_calendar_policy(log, start_timestamp) # Reuses from DESInitializer
        arrivals = self._build_arrival_policy(log, time_unit) # Overridden
        resources = self._build_resource_policy(log) # Reuses from DESInitializer
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

        if len(inter_arrivals) == 0:
            # Handle case where no inter-arrivals can be computed (e.g., only one case)
            # Default to a very slow arrival rate, or raise an error
            print("Warning: No inter-arrival times found. Defaulting to a small lambda for arrival policy.")
            lambda_param = 0.0001 # A small rate
        else:
            lambda_param = 1 / np.mean(inter_arrivals) if np.mean(inter_arrivals) > 0 else 0.0001

        return ExponentialArrivalPolicy(lambda_param)

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

        params_by_activity = {}
        for activity, durations in durations_by_activity.items():
            if len(durations) > 0:
                mean = np.mean(durations)
                std_dev = np.std(durations)
                params_by_activity[activity] = (mean, std_dev)
            else:
                # Default to some values if no durations found for an activity
                params_by_activity[activity] = (0.0, 0.0) # Or some other sensible default

        return NormalProcessingTimePolicy(params_by_activity)
