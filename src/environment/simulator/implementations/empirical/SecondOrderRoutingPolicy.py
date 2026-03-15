import random

from environment.entities.Case import Case
from environment.simulator.policies.RoutingPolicy import RoutingPolicy
from environment.simulator.implementations.empirical.ProbabilisticRoutingPolicy import ProbabilisticRoutingPolicy


class SecondOrderRoutingPolicy(RoutingPolicy):
    """
    Second-order Markov routing policy.

    Chooses the next activity based on both the current and the previous
    activity: P(next | previous, current).

    Keys in `probabilities` are (previous_activity, current_activity) tuples,
    where None stands for the start-of-case sentinel.

    Falls back to a first-order policy when the second-order key is not found
    (e.g. at the very first step, or for unseen bigrams).
    """

    def __init__(self, probabilities: dict, fallback: ProbabilisticRoutingPolicy):
        # probabilities: {(prev, current): {next: weight, ...}, ...}
        self.probabilities = probabilities
        self.fallback = fallback

    def get_activity_probabilities(self, case: Case) -> dict:
        history = case.activity_history
        current = history[-1] if history else None
        previous = history[-2] if len(history) >= 2 else None
        probs = self.probabilities.get((previous, current))
        if probs:
            return probs
        return self.fallback.get_activity_probabilities(case)

    def get_next_activity(self, case: Case):
        history = case.activity_history
        current = history[-1] if history else None
        previous = history[-2] if len(history) >= 2 else None

        choices = self.probabilities.get((previous, current))
        if choices:
            return random.choices(list(choices.keys()), weights=choices.values())[0]

        # Fallback to first-order when bigram was never observed
        return self.fallback.get_next_activity(case)

    def __repr__(self):
        return f"SecondOrderRoutingPolicy(n_bigrams={len(self.probabilities)})"
