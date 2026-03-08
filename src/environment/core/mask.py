from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from environment.entities.Case import Case
    from environment.entities.Resource import Resource
    from environment.simulator.policies.RoutingPolicy import RoutingPolicy


# ── Context objects ──────────────────────────────────────────────────────────

@dataclass
class ActivityMaskContext:
    """
    Everything an activity-masking strategy needs to decide which
    activities are feasible at a given decision point.
    """
    probabilities: dict[str, float]   # {activity_name: branching_probability}
    all_activities: list[str]         # ordered activity list (index ↔ position)


@dataclass
class ResourceMaskContext:
    """
    Everything a resource-masking strategy needs to decide which
    resources are feasible for a chosen activity.
    """
    activity_name: Optional[str]      # None means END
    all_resources: list["Resource"]   # ordered resource list (index ↔ position)


# ── Abstract base classes ────────────────────────────────────────────────────

class ActivityMaskFunction(ABC):
    """
    Determines which activities the agent is allowed to select.

    Receives an ActivityMaskContext and returns a binary numpy mask
    of shape (num_activities,).
    """

    @abstractmethod
    def compute(self, ctx: ActivityMaskContext) -> np.ndarray:
        """
        :param ctx: Context with branching probabilities and activity list.
        :return: Binary float32 mask — 1.0 for allowed activities, 0.0 otherwise.
        """
        pass


class ResourceMaskFunction(ABC):
    """
    Determines which resources the agent is allowed to select for a
    given activity.

    Receives a ResourceMaskContext and returns a binary numpy mask
    of shape (num_resources,).
    """

    @abstractmethod
    def compute(self, ctx: ResourceMaskContext) -> np.ndarray:
        """
        :param ctx: Context with activity name and resource list.
        :return: Binary float32 mask — 1.0 for allowed resources, 0.0 otherwise.
        """
        pass


# ── Implementations ──────────────────────────────────────────────────────────

class NucleusMaskFunction(ActivityMaskFunction):
    """
    Top-K / Top-P (nucleus) filtering over branching probabilities.

    This is the masking strategy described in the thesis (Section 'Mask'):
    the action space is limited to the most probable activities according to
    the routing policy, controlled by ``k`` and ``p`` parameters.

    Parameters
    ----------
    k : int or None
        Keep at most the top-k activities by probability.  None disables top-k.
    p : float
        Nucleus probability threshold.  Activities are included in decreasing
        probability order until cumulative probability >= p.  Set to 1.0 to
        disable nucleus filtering.
    """

    def __init__(self, k: Optional[int] = None, p: float = 0.9):
        self.k = k
        self.p = p

    def compute(self, ctx: ActivityMaskContext) -> np.ndarray:
        num_activities = len(ctx.all_activities)
        mask = np.zeros(num_activities, dtype=np.float32)

        if not ctx.probabilities:
            mask[-1] = 1.0  # only END is allowed
            return mask

        for act_name, prob in ctx.probabilities.items():
            try:
                idx = ctx.all_activities.index(act_name)
                mask[idx] = prob
            except ValueError:
                continue

        # Top-K filtering
        if self.k is not None and 0 < self.k < num_activities:
            top_k_indices = np.argsort(mask)[-self.k:]
            new_mask = np.zeros_like(mask)
            new_mask[top_k_indices] = mask[top_k_indices]
            mask = new_mask

        # Top-P (nucleus) filtering
        if self.p < 1.0:
            sorted_indices = np.argsort(mask)[::-1]
            sorted_probs = mask[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)

            cutoff_idx = np.where(cumulative_probs >= self.p)[0]
            if len(cutoff_idx) > 0:
                cutoff = cutoff_idx[0]
                top_p_indices = sorted_indices[:cutoff + 1]
                new_mask = np.zeros_like(mask)
                new_mask[top_p_indices] = mask[top_p_indices]
                mask = new_mask

        binary_mask = (mask > 0).astype(np.float32)

        # Ensure at least one activity is allowed (fallback to END)
        if binary_mask.sum() == 0:
            binary_mask[-1] = 1.0

        return binary_mask


class SkillBasedMaskFunction(ResourceMaskFunction):
    """
    Masks resources based on whether they possess the skill for the
    requested activity.  For END actions (activity_name is None), all
    resources are allowed.
    """

    def compute(self, ctx: ResourceMaskContext) -> np.ndarray:
        num_resources = len(ctx.all_resources)

        if ctx.activity_name is None:
            return np.ones(num_resources, dtype=np.float32)

        mask = np.zeros(num_resources, dtype=np.float32)
        for i, res in enumerate(ctx.all_resources):
            if ctx.activity_name in res.skills:
                mask[i] = 1.0

        # Fallback: allow all if no resource has the skill
        if mask.sum() == 0:
            return np.ones(num_resources, dtype=np.float32)

        return mask
