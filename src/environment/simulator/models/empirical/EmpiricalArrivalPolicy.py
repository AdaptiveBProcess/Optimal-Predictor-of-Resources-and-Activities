import random

from environment.simulator.policies.ArrivalPolicy import ArrivalPolicy

class EmpiricalArrivalPolicy(ArrivalPolicy):

    def __init__(self, inter_arrival_times):
        self.inter_arrivals = inter_arrival_times

    def get_next_arrival_time(self) -> float:
        return random.choice(self.inter_arrivals)

    def __str__(self):
        return (
            "EmpiricalArrivalPolicy\n"
            f"  samples={len(self.inter_arrivals)}, "
            f"mean={sum(self.inter_arrivals)/len(self.inter_arrivals):.2f}s"
        )
