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
    @abstractmethod
    def compute(self, ctx: ActivityMaskContext) -> np.ndarray:
        pass


class ResourceMaskFunction(ABC):
    @abstractmethod
    def compute(self, ctx: ResourceMaskContext) -> np.ndarray:
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
    end_threshold : float
        Minimum branching probability for the END (None) action to be allowed.
        If the probability of None is below this threshold, it is masked out
        and the agent must continue the trace.  This prevents the agent from
        exploiting low-probability early terminations.
        Default is 0.5 (END must be the majority transition to be allowed).
        Set to 0.0 to disable this filter (allow END whenever it has any probability).
    """

    def __init__(
        self,
        k: Optional[int] = None,
        p: float = 0.9,
        end_threshold: float = 0.5,
    ):
        self.k = k
        self.p = p
        self.end_threshold = end_threshold

    def compute(self, ctx: ActivityMaskContext) -> np.ndarray:
        num_activities = len(ctx.all_activities)
        mask = np.zeros(num_activities, dtype=np.float32)

        if not ctx.probabilities:
            mask[-1] = 1.0  # only END is allowed
            return mask

        # ── Step 0: Filter END (None) based on threshold ────────────
        end_prob = ctx.probabilities.get(None, 0.0)
        filtered_probs = {}
        for act_name, prob in ctx.probabilities.items():
            if act_name is None:
                # Only include END if its probability meets the threshold
                if end_prob >= self.end_threshold:
                    filtered_probs[act_name] = prob
                # Otherwise: skip it, agent must continue
            else:
                filtered_probs[act_name] = prob

        # If filtering removed everything (only END existed but below threshold),
        # this shouldn't happen in a well-formed process, but handle gracefully
        if not filtered_probs:
            # Allow END as fallback — the process has no other option
            mask[-1] = 1.0
            return mask

        # ── Step 1: Build probability mask ──────────────────────────
        for act_name, prob in filtered_probs.items():
            try:
                idx = ctx.all_activities.index(act_name)
                mask[idx] = prob
            except ValueError:
                continue

        # ── Step 2: Top-K filtering ─────────────────────────────────
        if self.k is not None and 0 < self.k < num_activities:
            top_k_indices = np.argsort(mask)[-self.k:]
            new_mask = np.zeros_like(mask)
            new_mask[top_k_indices] = mask[top_k_indices]
            mask = new_mask

        # ── Step 3: Top-P (nucleus) filtering ───────────────────────
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

        # ── Step 4: Binarize ────────────────────────────────────────
        binary_mask = (mask > 0).astype(np.float32)

        # Ensure at least one activity is allowed
        if binary_mask.sum() == 0:
            # Fallback: allow the highest-probability non-END activity
            for act_name, prob in sorted(filtered_probs.items(), key=lambda x: -x[1]):
                if act_name is not None:
                    try:
                        idx = ctx.all_activities.index(act_name)
                        binary_mask[idx] = 1.0
                        break
                    except ValueError:
                        continue
            # If still nothing (no non-END activities), allow END
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