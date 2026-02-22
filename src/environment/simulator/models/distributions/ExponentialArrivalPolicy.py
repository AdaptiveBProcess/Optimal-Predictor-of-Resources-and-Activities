import random
import numpy as np

from environment.simulator.policies.ArrivalPolicy import ArrivalPolicy

class ExponentialArrivalPolicy(ArrivalPolicy):

    def __init__(self, lambda_param: float):
        if lambda_param <= 0:
            raise ValueError("Lambda parameter for ExponentialArrivalPolicy must be positive.")
        self.lambda_param = lambda_param

    def get_next_arrival_time(self) -> float:
        return float(np.random.exponential(1 / self.lambda_param))

    def __str__(self):
        return (
            f"ExponentialArrivalPolicy(lambda={self.lambda_param:.4f})"
        )
