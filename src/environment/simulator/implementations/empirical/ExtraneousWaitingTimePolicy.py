import random

import numpy as np

from environment.simulator.policies.WaitingTImePolicy import WaitingTimePolicy


class ExtraneousWaitingTimePolicy(WaitingTimePolicy):
    """
    Empirical extraneous waiting time policy stratified by (activity, resource)
    pair, with an activity-only fallback — mirrors the structure of
    EmpiricalResourceActivityProcessingTimePolicy.

    Delays are the raw inter-event gaps (prev_end → curr_start) within a case,
    filtered to positive values and capped at p99.5 by the initializer.
    """

    def __init__(
        self,
        samples_by_activity_resource: dict,  # {(activity, resource_id): [float, ...]}
        samples_by_activity: dict,           # {activity: [float, ...]}
        fallback_delay: float = 0.0,
    ):
        self._by_pair = samples_by_activity_resource
        self._by_activity = samples_by_activity
        self._fallback = fallback_delay

    def get_waiting_time(self, activity, resource) -> float:
        resource_id = resource.id if resource is not None else None
        key = (activity, resource_id)

        if key in self._by_pair and self._by_pair[key]:
            return float(random.choice(self._by_pair[key]))

        if activity in self._by_activity and self._by_activity[activity]:
            return float(random.choice(self._by_activity[activity]))

        return self._fallback

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def __str__(self) -> str:
        out = {}
        for act, delays in self._by_activity.items():
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
