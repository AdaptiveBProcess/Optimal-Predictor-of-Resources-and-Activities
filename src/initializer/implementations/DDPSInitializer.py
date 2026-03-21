import pandas as pd
import numpy as np
from collections import defaultdict, Counter

from environment.simulator.core.log_names import LogColumnNames
from environment.simulator.implementations.empirical.SkillBasedResourcePolicy import SkillBasedResourcePolicy
from environment.simulator.implementations.empirical.EmpiricalArrivalPolicy import EmpiricalArrivalPolicy
from environment.simulator.implementations.distributions.WeeklyArrivalPolicy import WeeklyArrivalPolicy

from environment.simulator.implementations.empirical.WeeklyCalendarPolicy import WeeklyCalendarPolicy
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
        Analytical O(1)-per-gap off-time calculation replacing the minute-level
        date_range loop.  Uses the precomputed weekly availability matrix.
        """
        col_case = self.log_names.case_id
        col_start = self.log_names.start_timestamp
        col_end = self.log_names.end_timestamp
        col_act = self.log_names.activity

        threshold = np.percentile(calendar_policy.raw_matrix.flatten(), 20)
        availability = calendar_policy.raw_matrix > threshold    # shape (7, 24) bool

        # ── Precompute per-weekday off-hours count ───────────────────
        # off_hours_per_day[d] = number of OFF hours on weekday d
        off_hours_per_day = np.array(
            [(~availability[d]).sum() for d in range(7)], dtype=np.float64
        )
        # Total off-seconds in a full week
        off_seconds_per_week = float(off_hours_per_day.sum() * 3600)

        def _off_seconds_in_gap(prev_end: pd.Timestamp, curr_start: pd.Timestamp) -> float:
            """
            Count off-duty seconds between two timestamps using the
            weekly availability matrix — without iterating minute-by-minute.

            Strategy:
              1. Count full weeks → multiply by weekly off total.
              2. Walk the remaining partial hours (at most ~168 h = 1 week).
            """
            total_seconds = (curr_start - prev_end).total_seconds()
            if total_seconds <= 0:
                return 0.0

            total_hours = total_seconds / 3600.0
            full_weeks = int(total_hours // 168)  # 168 h / week
            off = full_weeks * off_seconds_per_week

            # Remaining partial-week portion: iterate by hour (max ~168 iters)
            remainder_start = prev_end + pd.Timedelta(weeks=full_weeks)
            t = remainder_start.replace(minute=0, second=0, microsecond=0)
            if t < remainder_start:
                t += pd.Timedelta(hours=1)

            # Partial first hour
            first_partial = (t - remainder_start).total_seconds()
            if first_partial > 0 and not availability[remainder_start.weekday(), remainder_start.hour]:
                off += first_partial

            # Full hours in the remainder
            while t < curr_start:
                d, h = t.weekday(), t.hour
                next_t = t + pd.Timedelta(hours=1)
                if next_t <= curr_start:
                    if not availability[d, h]:
                        off += 3600.0
                else:
                    # Partial last hour
                    partial = (curr_start - t).total_seconds()
                    if not availability[d, h]:
                        off += partial
                t = next_t

            return off

        # ── Vectorised gap computation per case ──────────────────────
        extraneous_by_activity = defaultdict(list)

        sorted_log = log.sort_values(by=[col_case, col_start])

        for _, group in sorted_log.groupby(col_case, sort=False):
            if len(group) < 2:
                continue

            prev_ends = group[col_end].iloc[:-1].values        # numpy datetime64
            curr_starts = group[col_start].iloc[1:].values
            acts = group[col_act].iloc[1:].values

            # Vectorised gap in seconds (numpy timedelta → float)
            gaps_sec = (curr_starts - prev_ends).astype("timedelta64[s]").astype(np.float64)

            for idx in range(len(gaps_sec)):
                gap = gaps_sec[idx]
                if gap <= 0 or np.isnan(gap):
                    continue

                # Convert numpy datetime64 to Timestamp for the analytical method
                pe = pd.Timestamp(prev_ends[idx])
                cs = pd.Timestamp(curr_starts[idx])

                off = _off_seconds_in_gap(pe, cs)
                extraneous_sec = max(0.0, gap - off*0.01)
                extraneous = self._time_unit_conversion(extraneous_sec, time_unit)
                extraneous_by_activity[acts[idx]].append(extraneous)

        # ── IQR-based outlier filter (Q3 + 3×IQR fence) ─────────────
        for act in extraneous_by_activity:
            delays = extraneous_by_activity[act]
            if delays:
                q1, q3 = np.percentile(delays, [25, 75])
                threshold = q3 + 40 * (q3 - q1)
                extraneous_by_activity[act] = [d for d in delays if d <= threshold]

        return ExtraneousWaitingTimePolicy(extraneous_by_activity)

    # ─────────────────────────────────────────────────────────────────
    # CALENDAR — vectorised with .dt accessors
    # ─────────────────────────────────────────────────────────────────
    def _build_calendar_policy(self, log, start_timestamp: str) -> CalendarPolicy:
        ts = log[self.log_names.start_timestamp]
        valid = ts.notna()

        weekdays = ts[valid].dt.weekday.values   # 0–6
        hours = ts[valid].dt.hour.values          # 0–23

        calendar_counts = np.zeros((7, 24), dtype=np.float64)
        np.add.at(calendar_counts, (weekdays, hours), 1)
        
        non_zero = calendar_counts > 0
        # Global 20th-percentile threshold on raw counts (no per-row normalisation)
        threshold = np.percentile(calendar_counts[non_zero].flatten(), 20)
        availability = calendar_counts > threshold

        return WeeklyCalendarPolicy(availability, calendar_counts, start_timestamp)

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