import random
from collections import defaultdict

import numpy as np

from environment.simulator.policies.ProcessingTimePolicy import ProcessingTimePolicy


class EmpiricalResourceActivityProcessingTimePolicy(ProcessingTimePolicy):
    """
    Empirical processing time policy that samples from observed durations
    stratified by (activity, resource) pair.  Falls back to activity-only
    samples when the specific (activity, resource) combination was not seen
    in the training log.
    """

    def __init__(self, samples_by_activity_resource: dict, samples_by_activity: dict):
        # keys: (activity, resource) -> list[float]
        self._by_pair = samples_by_activity_resource
        # fallback keys: activity -> list[float]
        self._by_activity = samples_by_activity

    def get_activity_duration(self, activity, resource=None) -> float:
        resource_id = resource.id if resource is not None else None
        key = (activity, resource_id)

        if key in self._by_pair and self._by_pair[key]:
            return random.choice(self._by_pair[key])

        # Fallback: activity-only distribution
        if activity in self._by_activity and self._by_activity[activity]:
            return random.choice(self._by_activity[activity])

        return 0.0

    def __str__(self) -> str:
        lines = ["EmpiricalResourceActivityProcessingTimePolicy"]

        all_activities = sorted({act for act, _ in self._by_pair})
        for activity in all_activities:
            pairs = {
                res: samples
                for (act, res), samples in self._by_pair.items()
                if act == activity and samples
            }
            for resource_id, samples in sorted(pairs.items()):
                arr = np.array(samples)
                lines.append(
                    f"  ({activity}, {resource_id}): "
                    f"n={len(samples)}, "
                    f"mean={arr.mean():.2f}, "
                    f"min={arr.min():.2f}, "
                    f"max={arr.max():.2f}"
                )

        return "\n".join(lines)
