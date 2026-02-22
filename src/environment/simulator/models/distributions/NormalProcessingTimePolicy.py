import random
import numpy as np

from environment.simulator.policies.ProcessingTimePolicy import ProcessingTimePolicy
from typing import Dict, Tuple

class NormalProcessingTimePolicy(ProcessingTimePolicy):
    def __init__(self, params_by_activity: Dict[str, Tuple[float, float]]):
        self.params = params_by_activity

    def get_activity_duration(self, activity: str, resource=None) -> float:
        mean, std_dev = self.params.get(activity, (0.0, 0.0))
        # Ensure duration is not negative. In some cases, a normal distribution can yield negative values.
        # We can either re-sample, or return 0, or return a small positive number.
        # For simplicity, we'll return 0 if the sampled value is negative.
        duration = np.random.normal(loc=mean, scale=std_dev)
        return max(0.0, duration)

    def __str__(self) -> str:
        lines = ["NormalProcessingTimePolicy"]
        for activity, (mean, std_dev) in sorted(self.params.items()):
            lines.append(
                f"  {activity}: "
                f"mean={mean:.2f}, "
                f"std_dev={std_dev:.2f}"
            )
        return "".join(lines)
