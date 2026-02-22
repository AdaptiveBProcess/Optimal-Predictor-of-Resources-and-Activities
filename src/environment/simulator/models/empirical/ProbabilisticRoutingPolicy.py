
import random
from environment.entities.Case import Case
from environment.simulator.policies.RoutingPolicy import RoutingPolicy


class ProbabilisticRoutingPolicy(RoutingPolicy):
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def get_next_activity(self, case_state: "Case", current_activity=None):
        current = current_activity
        choices = self.probabilities.get(current)
        if not choices:
            return None
        return random.choices(
            list(choices.keys()),
            weights=choices.values()
        )[0]
    
    def __str__(self):
        lines = ["ProbabilisticRoutingPolicy:"]
        for src, targets in self.probabilities.items():
            src_label = "START" if src is None else src
            if not targets:
                lines.append(f"  {src_label} -> END")
            else:
                rhs = ", ".join(
                    f"{tgt} ({p:.2f})" for tgt, p in targets.items()
                )
                lines.append(f"  {src_label} -> {rhs}")
        return "\n".join(lines)
    def __repr__(self):
        return f"ProbabilisticRoutingPolicy(n_states={len(self.probabilities)})"
