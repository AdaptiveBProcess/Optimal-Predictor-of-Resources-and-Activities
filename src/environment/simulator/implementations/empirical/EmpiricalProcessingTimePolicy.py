import random

from environment.simulator.policies.ProcessingTimePolicy import ProcessingTimePolicy

import numpy as np

class EmpiricalProcessingTimePolicy(ProcessingTimePolicy):
    def __init__(self, samples_by_activity):
        self.samples = samples_by_activity

    def get_activity_duration(self, activity, resource=None):
        return random.choice(self.samples[activity])

    def __str__(self) -> str:
        lines = ["EmpiricalProcessingTimePolicy"]

        for activity, samples in sorted(self.samples.items()):
            if len(samples) == 0:
                lines.append(f"  {activity}: n=0")
                continue

            arr = np.array(samples)
            mean = arr.mean()
            min_v = arr.min()
            max_v = arr.max()

            lines.append(
                f"  {activity}: "
                f"n={len(samples)}, "
                f"mean={mean:.2f}, "
                f"min={min_v:.2f}, "
                f"max={max_v:.2f}"
            )

        return "\n".join(lines)