from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.entities.Case import Case


@dataclass
class CaseRewardContext:
    """
    All the information a reward function might need to score a completed case.

    This context is built by the environment at case-completion time and passed
    to the reward function.  Adding new fields here (rather than changing the
    RewardFunction signature) keeps every implementation forward-compatible.
    """
    cycle_time: float
    sla_threshold: float
    num_events: int
    start_time: float
    end_time: float


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.

    A reward function receives a CaseRewardContext for each completed case
    and returns a scalar reward signal that the RL agent will optimize.
    """

    @abstractmethod
    def compute(self, ctx: CaseRewardContext) -> float:
        """
        Computes the reward for a single completed case.

        :param ctx: Context with all case-level metrics.
        :return: A scalar reward value.
        """
        pass


class SLARewardFunction(RewardFunction):
    """
    Two-part reward from the thesis (Section 'Reward Function').

    Intermediate (per-case directional signal):
        r(σ) = K / ct(σ)

    Terminal (SLA compliance gate):
        R(σ) = +r(σ)   if ct(σ) < T
        R(σ) = -K       if ct(σ) >= T

    Parameters
    ----------
    K : float
        Scaling constant that controls reward magnitude.
        Defaults to 1.0.
    """

    def __init__(self, K: float = 1.0):
        self.K = K

    def compute(self, ctx: CaseRewardContext) -> float:
        ct = ctx.cycle_time
        if ct <= 0:
            ct = 1e-6  # guard against zero-duration cases

        if ct < ctx.sla_threshold:
            return self.K / ct
        else:
            return -self.K


class BinaryRewardFunction(RewardFunction):
    """
    Simple binary reward (the previous default).

    +1 if the case meets the SLA threshold, 0 otherwise.
    """

    def compute(self, ctx: CaseRewardContext) -> float:
        return 1.0 if ctx.cycle_time <= ctx.sla_threshold else 0.0
