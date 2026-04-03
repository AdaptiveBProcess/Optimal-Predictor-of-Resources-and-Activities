import pandas as pd
import numpy as np
from collections import defaultdict, Counter

from environment.simulator.core.log_names import LogColumnNames
from environment.simulator.implementations.empirical.SkillBasedResourcePolicy import SkillBasedResourcePolicy
from environment.simulator.implementations.empirical.EmpiricalArrivalPolicy import EmpiricalArrivalPolicy
from environment.simulator.implementations.distributions.WeeklyArrivalPolicy import WeeklyArrivalPolicy

from environment.simulator.implementations.empirical.WeeklyCalendarPolicy import WeeklyCalendarPolicy
from environment.simulator.implementations.empirical.WeeklyResourceCalendarPolicy import WeeklyResourceCalendarPolicy
from environment.simulator.policies import ArrivalPolicy, CalendarPolicy, ProcessingTimePolicy, RoutingPolicy
from environment.simulator.implementations.empirical.ExtraneousWaitingTimePolicy import ExtraneousWaitingTimePolicy
from environment.simulator.policies.WaitingTImePolicy import WaitingTimePolicy
from initializer.Initializer import Initializer
from environment.simulator.core.setup import SimulationSetup
from environment.entities.Resource import Resource
from environment.simulator.implementations.empirical.ProbabilisticRoutingPolicy import ProbabilisticRoutingPolicy
from environment.simulator.implementations.empirical.SecondOrderRoutingPolicy import SecondOrderRoutingPolicy
from environment.simulator.implementations.empirical.EmpiricalProcessingTimePolicy import EmpiricalProcessingTimePolicy
from environment.simulator.implementations.empirical.EmpiricalResourceActivityProcessingTimePolicy import EmpiricalResourceActivityProcessingTimePolicy


class DDPSInitializer(Initializer):

    def build(self, log, log_names: LogColumnNames, start_timestamp: str, time_unit: str) -> SimulationSetup:
        self.log_names = log_names

        # ── Pre-parse timestamps once ────────────────────────────────
        # Avoids repeated pd.to_datetime() calls in every sub-method
        log = log.copy()
        log[log_names.start_timestamp] = pd.to_datetime(
            log[log_names.start_timestamp], format="mixed"
        )
        log[log_names.end_timestamp] = pd.to_datetime(
            log[log_names.end_timestamp], format="mixed"
        )

        routing = self._build_routing_policy(log)
        print("First-order Markov routing policy built.")

        processing_times = self._build_resource_activity_processing_time_policy(log, time_unit)
        print("Empirical resource-activity processing time policy built.")

        # ── Build calendar ONCE, reuse in waiting time policy ────────
        calendar = self._build_calendar_policy(log, start_timestamp)
        print("Weekly calendar policy built.")

        waiting_times = self._build_waiting_time_policy(log, time_unit, calendar)
        print("Empirical extraneous waiting time policy built.")

        arrivals = self._build_arrival_policy(log, time_unit, start_timestamp)
        print("Empirical Weekly arrival policy built.")

        resource_list = self._build_resource_list(log)
        print("Skill-based resource policy built.")

        resource_policy = SkillBasedResourcePolicy(resource_list)
        print("Resource policy built.")

        activities = sorted(self._extract_activities(log))
        print("Activities extracted.")

        return SimulationSetup(
            time_unit=time_unit,
            start_timestamp=start_timestamp,
            routing_policy=routing,
            waiting_time_policy=waiting_times,
            processing_time_policy=processing_times,
            calendar_policy=calendar,
            arrival_policy=arrivals,
            resource_policy=resource_policy,
            activities=activities,
            resources=resource_list,
        )

    # ─────────────────────────────────────────────────────────────────
    # ROUTING — vectorised with shift() instead of iterrows()
    # ─────────────────────────────────────────────────────────────────
    def _build_routing_policy(self, log) -> RoutingPolicy:
        col_case = self.log_names.case_id
        col_act = self.log_names.activity

        # Sort so that shift() gives us the correct next activity
        sorted_log = log.sort_values(by=[col_case, self.log_names.start_timestamp])
        activities = sorted_log[col_act].values
        case_ids = sorted_log[col_case].values

        # Identify boundaries where case changes
        same_case = case_ids[:-1] == case_ids[1:]

        # Build current→next pairs
        current_acts = activities[:-1]
        next_acts = activities[1:]

        # Mask cross-case transitions: for these, current is terminal (→ None)
        # and next is a start (None →)
        counts = defaultdict(Counter)

        # ── Within-case transitions ──
        mask = same_case
        for cur, nxt in zip(current_acts[mask], next_acts[mask]):
            counts[cur][nxt] += 1

        # ── Start-of-case transitions (None → first activity) ──
        # First event in log is always a start
        counts[None][activities[0]] += 1
        # Every position where the previous row was a different case
        not_same = ~same_case
        for nxt in next_acts[not_same]:
            counts[None][nxt] += 1

        # ── End-of-case transitions (last activity → None) ──
        for cur in current_acts[not_same]:
            counts[cur][None] += 1
        # Last event in log is always a case end
        counts[activities[-1]][None] += 1

        # ── Normalise ──
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

    def _build_second_order_routing_policy(self, log) -> RoutingPolicy:
        """
        Second-order Markov routing policy:
        P(next_activity | previous_activity, current_activity).
        """
        col_case = self.log_names.case_id
        col_act = self.log_names.activity

        sorted_log = log.sort_values(by=[col_case, self.log_names.start_timestamp])
        activities = sorted_log[col_act].values
        case_ids = sorted_log[col_case].values

        counts = defaultdict(Counter)

        for _, group in sorted_log.groupby(col_case, sort=False):
            acts = group[col_act].values
            # Build history: [None, a0, a1, ..., aN, None]
            history = np.empty(len(acts) + 2, dtype=object)
            history[0] = None
            history[1:-1] = acts
            history[-1] = None

            for i in range(1, len(history) - 1):
                counts[(history[i - 1], history[i])][history[i + 1]] += 1

        transition_probs = {}
        for bigram, counter in counts.items():
            total = sum(counter.values())
            transition_probs[bigram] = {
                nxt: cnt / total for nxt, cnt in counter.items()
            }

        fallback = self._build_routing_policy(log)
        return SecondOrderRoutingPolicy(transition_probs, fallback)

    # ─────────────────────────────────────────────────────────────────
    # PROCESSING TIMES — fully vectorised column arithmetic
    # ─────────────────────────────────────────────────────────────────
    def _build_processing_time_policy(self, log, time_unit: str) -> ProcessingTimePolicy:
        starts = log[self.log_names.start_timestamp]
        ends = log[self.log_names.end_timestamp]

        # Vectorised duration in seconds
        durations_sec = (ends - starts).dt.total_seconds()

        # Build a mask: both timestamps valid AND duration >= 0
        valid = starts.notna() & ends.notna() & (durations_sec >= 0)

        activities = log.loc[valid, self.log_names.activity].values
        durations = self._time_unit_conversion(durations_sec[valid].values, time_unit)

        # Group into dict of lists
        durations_by_activity = defaultdict(list)
        for act, dur in zip(activities, durations):
            durations_by_activity[act].append(dur)

        return EmpiricalProcessingTimePolicy(durations_by_activity)

    def _build_resource_activity_processing_time_policy(self, log, time_unit: str) -> ProcessingTimePolicy:
        """
        Builds an EmpiricalResourceActivityProcessingTimePolicy stratified by
        (activity, resource) pair, with an activity-only fallback.
        """
        col_res = self.log_names.resource
        starts = log[self.log_names.start_timestamp]
        ends = log[self.log_names.end_timestamp]

        durations_sec = (ends - starts).dt.total_seconds()
        valid = starts.notna() & ends.notna() & (durations_sec >= 0)

        activities = log.loc[valid, self.log_names.activity].values
        resources = log.loc[valid, col_res].values
        durations = self._time_unit_conversion(durations_sec[valid].values, time_unit)

        by_pair = defaultdict(list)
        by_activity = defaultdict(list)
        for act, res, dur in zip(activities, resources, durations):
            by_pair[(act, res)].append(dur)
            by_activity[act].append(dur)

        return EmpiricalResourceActivityProcessingTimePolicy(dict(by_pair), dict(by_activity))

    # ─────────────────────────────────────────────────────────────────
    # WAITING TIMES — analytical off-time instead of minute-by-minute loop
    # ─────────────────────────────────────────────────────────────────
    def _build_waiting_time_policy(
        self, log, time_unit: str, calendar_policy: CalendarPolicy
    ) -> WaitingTimePolicy:
        """
        Empirical waiting time policy stratified by (activity, resource) pair
        with an activity-only fallback.  Raw inter-event gaps within each case
        are used directly (no calendar subtraction).
        """
        col_case = self.log_names.case_id
        col_start = self.log_names.start_timestamp
        col_end = self.log_names.end_timestamp
        col_act = self.log_names.activity
        col_res = self.log_names.resource

        by_pair = defaultdict(list)
        by_activity = defaultdict(list)

        sorted_log = log.sort_values(by=[col_case, col_start])

        for _, group in sorted_log.groupby(col_case, sort=False):
            if len(group) < 2:
                continue

            prev_ends = group[col_end].iloc[:-1].values
            curr_starts = group[col_start].iloc[1:].values
            acts = group[col_act].iloc[1:].values
            resources = group[col_res].iloc[1:].values

            gaps_sec = (curr_starts - prev_ends).astype("timedelta64[s]").astype(np.float64)

            for idx in range(len(gaps_sec)):
                gap = gaps_sec[idx]
                if gap <= 0 or np.isnan(gap):
                    continue
                delay = self._time_unit_conversion(gap, time_unit)
                act = acts[idx]
                res = resources[idx]
                by_pair[(act, res)].append(delay)
                by_activity[act].append(delay)

        # p99.5 filter per key
        def _filter(d):
            return {k: [v for v in vs if v <= np.percentile(vs, 99.5)] for k, vs in d.items()}

        return ExtraneousWaitingTimePolicy(_filter(by_pair), _filter(by_activity))

    # ─────────────────────────────────────────────────────────────────
    # CALENDAR — vectorised with .dt accessors
    # ─────────────────────────────────────────────────────────────────
    def _build_calendar_policy(
        self, log, start_timestamp: str, participation_threshold: float = 0.1
    ) -> CalendarPolicy:
        col_act = self.log_names.activity
        col_res = self.log_names.resource
        ts_col  = self.log_names.start_timestamp

        valid = log[ts_col].notna()
        sub   = log[valid]

        # ── Global matrix (same logic as before) ────────────────────────
        weekdays_all = sub[ts_col].dt.weekday.values
        hours_all    = sub[ts_col].dt.hour.values
        global_counts = np.zeros((7, 24), dtype=np.float64)
        np.add.at(global_counts, (weekdays_all, hours_all), 1)
        non_zero = global_counts > 0
        threshold_global = np.percentile(global_counts[non_zero].flatten(), 20)
        global_avail = global_counts > threshold_global

        # ── RParticipation per resource ──────────────────────────────────
        # pair_counts: MultiIndex Series (resource, activity) -> count
        pair_counts = sub.groupby([col_res, col_act]).size()
        # activity_max[a] = max events by any resource for activity a
        activity_max = pair_counts.groupby(level=col_act).max()

        resource_avail: dict = {}
        resources = sub[col_res].dropna().unique()

        for r in resources:
            if r not in pair_counts:
                continue
            r_pairs     = pair_counts[r]                         # Series a->count
            numerator   = r_pairs.sum()
            denominator = activity_max.reindex(r_pairs.index).sum()
            participation = numerator / denominator if denominator > 0 else 0.0

            if participation >= participation_threshold:
                r_sub      = sub[sub[col_res] == r]
                r_weekdays = r_sub[ts_col].dt.weekday.values
                r_hours    = r_sub[ts_col].dt.hour.values
                r_counts   = np.zeros((7, 24), dtype=np.float64)
                np.add.at(r_counts, (r_weekdays, r_hours), 1)
                nz = r_counts > 0
                if nz.any():
                    thr = np.percentile(r_counts[nz].flatten(), 5)
                    avail = r_counts > thr
                    if avail.any():
                        resource_avail[r] = avail
                    # else: all counts equal the threshold (e.g. single unique value)
                    # — fall back to global to avoid an empty calendar

        print(f"  Per-resource calendars: {len(resource_avail)}/{len(resources)} resources "
              f"above participation threshold ({participation_threshold}).")

        return WeeklyResourceCalendarPolicy(resource_avail, global_avail, start_timestamp)

    # ─────────────────────────────────────────────────────────────────
    # ARRIVALS — minor cleanup (already efficient)
    # ─────────────────────────────────────────────────────────────────
    def _build_naive_arrival_policy(self, log, time_unit: str) -> ArrivalPolicy:
        case_starts = (
            log.groupby(self.log_names.case_id)[self.log_names.start_timestamp]
            .min()
            .sort_values()
        )

        start_times = case_starts.astype("int64") / 1e9  # to epoch seconds
        inter_arrivals = np.diff(start_times.values)
        inter_arrivals = self._time_unit_conversion(inter_arrivals, time_unit)
        inter_arrivals = inter_arrivals[inter_arrivals > 0]

        return EmpiricalArrivalPolicy(inter_arrivals.tolist())
    
    def _build_arrival_policy(self, log, time_unit: str, start_timestamp: str) -> ArrivalPolicy:
        case_starts = (
            log.groupby(self.log_names.case_id)[self.log_names.start_timestamp]
            .min()
            .dropna()
        )
        timestamps = pd.to_datetime(case_starts)

        # Build a 7x24 arrival rate matrix (arrivals per hour per slot)
        rate_matrix = np.zeros((7, 24), dtype=np.float64)
        np.add.at(rate_matrix, (timestamps.dt.weekday.values, timestamps.dt.hour.values), 1)

        # Normalize by the number of weeks observed to get a per-week, per-slot rate
        total_hours = (timestamps.max() - timestamps.min()).total_seconds() / 3600
        n_weeks = max(1.0, total_hours / (7 * 24))
        rate_matrix /= n_weeks

        return WeeklyArrivalPolicy(rate_matrix, start_timestamp, time_unit)
    # ─────────────────────────────────────────────────────────────────
    # RESOURCES — vectorised groupby instead of iterrows()
    # ─────────────────────────────────────────────────────────────────
    def _build_resource_list(self, log):
        col_res = self.log_names.resource
        col_act = self.log_names.activity

        skills = (
            log.groupby(col_res)[col_act]
            .apply(set)
            .to_dict()
        )

        return [
            Resource(id=name, skills=skill_set)
            for name, skill_set in skills.items()
        ]

    # ─────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _extract_activities(self, log):
        return set(log[self.log_names.activity].unique())

    def _extract_start_end_activities(self, log):
        col_case = self.log_names.case_id
        col_start = self.log_names.start_timestamp
        col_act = self.log_names.activity

        sorted_log = log.sort_values(by=[col_case, col_start])
        grouped = sorted_log.groupby(col_case, sort=False)

        starts = set(grouped[col_act].first().values)
        ends = set(grouped[col_act].last().values)
        return starts, ends

    def _extract_resources(self, log):
        return set(log[self.log_names.resource].unique())

    @staticmethod
    def _time_unit_conversion(interval_seconds, time_unit):
        """Works on both scalars and numpy arrays."""
        if time_unit == "minutes":
            return interval_seconds / 60
        elif time_unit == "hours":
            return interval_seconds / 3600
        return interval_seconds