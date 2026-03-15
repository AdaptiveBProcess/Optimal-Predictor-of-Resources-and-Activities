import numpy as np
import random
import pandas as pd
from collections import defaultdict

from environment.entities import Case
from environment.simulator.policies.WaitingTImePolicy import WaitingTimePolicy

class ExtraneousWaitingTimePolicy(WaitingTimePolicy):
    """
    Empirical extraneous delay policy.

    For each activity, the extraneous delay is estimated as:

        gap            = curr_start - prev_end
        calendar_off   = off-duty seconds inside that gap (from WeeklyCalendarPolicy)
        extraneous     = gap - calendar_off

    Only positive extraneous values are kept — zero/negative values mean
    the gap was fully explained by calendar or contention (SimPy handles those).

    At simulation time, the sampled delay is injected BEFORE the resource
    request in execute_activity, matching the Prosimos timer-event pattern.
    """

    def __init__(
        self,
        extraneous_by_activity: dict,   # {activity: [delay_in_time_units, ...]}
        fallback_delay: float = 0.0,
    ):
        self._distributions: dict = {}
        self._fallback = fallback_delay

        for activity, delays in extraneous_by_activity.items():
            clean = [d for d in delays if d > 0]
            if clean:
                self._distributions[activity] = clean

    def get_waiting_time(self, activity, resource) -> float:
        pool = self._distributions.get(activity)
        if not pool:
            return self._fallback
        return float(random.choice(pool))

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def __str__(self) -> dict:
        """Return per-activity stats for inspection."""
        out = {}
        for act, delays in self._distributions.items():
            arr = np.array(delays)
            out[act] = {
                "n": len(arr),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
                "max": float(arr.max()),
            }
        return str(out)