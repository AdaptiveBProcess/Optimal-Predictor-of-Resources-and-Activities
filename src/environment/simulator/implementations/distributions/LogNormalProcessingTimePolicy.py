import numpy as np
from environment.simulator.policies.ProcessingTimePolicy import ProcessingTimePolicy
from typing import Dict, Tuple


class LogNormalProcessingTimePolicy(ProcessingTimePolicy):
    """
    Samples activity durations from a Log-Normal distribution.

    Parameters are (mu, sigma) in log-space, fitted from observed durations.
    This naturally produces right-skewed durations matching real processes.
    """

    def __init__(self, params_by_activity: Dict[str, Tuple[float, float]]):
        self.params = params_by_activity

    def get_activity_duration(self, activity: str, resource=None) -> float:
        mu, sigma = self.params.get(activity, (0.0, 0.1))
        if sigma <= 0:
            return max(0.0, np.exp(mu))
        return float(np.random.lognormal(mean=mu, sigma=sigma))
