import zlib
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
    last_activity: Optional[str] = None  # case's last executed activity (None = case start)
    case_id: Optional[str] = None        # identity of the deciding case (None = unknown)


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
    p_min_end : float
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
        p_min_end: float = 0.2,
    ):
        self.k = k
        self.p = p
        self.p_min_end = p_min_end


    def compute(self, ctx: ActivityMaskContext) -> np.ndarray:
        num_activities = len(ctx.all_activities)
        mask = np.zeros(num_activities, dtype=np.float32)

        
        if not ctx.probabilities:
            mask[-1] = 1.0  # only END is allowed
            return mask

        # ── Step 0: Filter END (None) based on argmax OR threshold ──
        end_prob = ctx.probabilities.get(None, 0.0)
        filtered_probs = {}

        # Determine if END should be allowed
        allow_end = False
        if end_prob > 0:
            non_end_probs = [v for k, v in ctx.probabilities.items() if k is not None]
            max_non_end = max(non_end_probs) if non_end_probs else 0.0
            # Allow END if it's the most probable transition OR above threshold
            allow_end = (end_prob > max_non_end) or (end_prob >= self.p_min_end)

        for act_name, prob in ctx.probabilities.items():
            if act_name is None:
                if allow_end:
                    filtered_probs[act_name] = prob
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


_UNKNOWN = object()


class TransitionRestrictionMaskFunction(ActivityMaskFunction):
    """
    Decorator over any ActivityMaskFunction that forbids specific
    ``last_activity -> next_activity`` transitions (outcome restriction /
    what-if analysis).  Works for every selector because all of them consume
    the env's activity mask.

    Parameters
    ----------
    base : ActivityMaskFunction
        Mask applied first (e.g. NucleusMaskFunction).
    forbidden : dict
        ``{source_activity: [forbidden_targets]}``.
        Source ``"*"`` = from any activity; source ``None`` = case start.
        Target ``"END"`` (or ``None``) = forbid ending the case.

    Example
    -------
    >>> TransitionRestrictionMaskFunction(NucleusMaskFunction(k=3), {"Radicar": ["Cancelar"]})

    coverage : float in [0, 1]
        Fraction of cases subject to the restrictions (default 1.0 = all).
        Each case is assigned ONCE, on its first decision: with probability
        ``coverage`` it is "restricted" and every rule applies to it at every
        decision; otherwise no rule ever applies to it.  The draw is a
        deterministic function of ``(seed, case_id)`` and does not touch the
        global RNG, so the same cases are restricted under every policy for a
        given seed (common random numbers) and the other random streams are
        unaffected.
    seed : int
        Seed for the per-case assignment (vary it across runs).

    Counters ``n_blocked`` (decisions where a forbidden option was actually
    removed from the mask) and ``n_fallback`` (decisions where the mask became
    empty and an alternative was re-enabled) let the analyst confirm the
    restriction fired.  ``n_cases_seen`` / ``n_cases_restricted`` report the
    realised coverage; ``restricted_case_ids`` lists the restricted cases.
    """

    ANY = "*"
    END_ALIAS = "END"

    def __init__(
        self,
        base: ActivityMaskFunction,
        forbidden: dict,
        coverage: float = 1.0,
        seed: int = 0,
    ):
        if not 0.0 <= coverage <= 1.0:
            raise ValueError(f"coverage must be in [0, 1], got {coverage}")
        self.base = base
        self.forbidden = {
            src: {None if t == self.END_ALIAS else t for t in targets}
            for src, targets in forbidden.items()
        }
        self.coverage = float(coverage)
        self.seed = int(seed)
        self.reset_counters()

    def reset_counters(self) -> None:
        self.n_blocked = 0
        self.n_fallback = 0
        self._case_flags: dict = {}

    @property
    def n_cases_seen(self) -> int:
        return len(self._case_flags)

    @property
    def n_cases_restricted(self) -> int:
        return sum(self._case_flags.values())

    @property
    def restricted_case_ids(self) -> list:
        return [cid for cid, flag in self._case_flags.items() if flag]

    def _case_is_restricted(self, case_id) -> bool:
        if case_id is None:
            # No identity available: fall back to restricting (old behaviour).
            return True
        flag = self._case_flags.get(case_id)
        if flag is None:
            if self.coverage >= 1.0:
                flag = True
            elif self.coverage <= 0.0:
                flag = False
            else:
                h = zlib.crc32(str(case_id).encode("utf-8"))
                flag = bool(np.random.default_rng([self.seed, h]).random() < self.coverage)
            self._case_flags[case_id] = flag
        return flag

    def validate(self, all_activities: list) -> None:
        """
        Resolve names against the simulator's activities and raise ValueError
        if any is unknown (catches typos).  Matching ignores surrounding
        whitespace (some logs store e.g. ' W_Handle leads').
        """
        by_stripped = {a.strip(): a for a in all_activities if a is not None}

        def resolve(name):
            if name in (self.ANY, None):
                return name
            if name in by_stripped.values():
                return name
            return by_stripped.get(str(name).strip(), _UNKNOWN)

        unknown = []
        resolved = {}
        for src, targets in self.forbidden.items():
            r_src = resolve(src)
            if r_src is _UNKNOWN:
                unknown.append(src)
            r_targets = set()
            for t in targets:
                r_t = resolve(t)
                if r_t is _UNKNOWN:
                    unknown.append(t)
                else:
                    r_targets.add(r_t)
            resolved.setdefault(r_src, set()).update(r_targets)
        if unknown:
            valid = sorted(str(a).strip() for a in all_activities if a is not None)
            raise ValueError(
                f"Unknown activity name(s) in restrictions: {unknown}. "
                f"Valid activities: {valid} (use '{self.END_ALIAS}' for case end, "
                f"'{self.ANY}' as source for any activity)."
            )
        self.forbidden = resolved

    def _targets_for(self, last_activity) -> set:
        return self.forbidden.get(self.ANY, set()) | self.forbidden.get(last_activity, set())

    def compute(self, ctx: ActivityMaskContext) -> np.ndarray:
        mask = np.asarray(self.base.compute(ctx), dtype=np.float32).copy()
        if not self._case_is_restricted(ctx.case_id):
            return mask
        targets = self._targets_for(ctx.last_activity)
        if not targets:
            return mask

        forbidden_idx = [i for i, a in enumerate(ctx.all_activities) if a in targets]
        if forbidden_idx and mask[forbidden_idx].sum() > 0:
            self.n_blocked += 1
        mask[forbidden_idx] = 0.0

        if mask.sum() == 0:
            # Re-enable ONE allowed observed transition (even if the base mask had
            # filtered it out) so the simulation never deadlocks.  It is sampled
            # proportionally to the routing probabilities: usually the most
            # probable one, but not always (avoids a greedy, deterministic choice).
            self.n_fallback += 1
            candidates = [
                (act, p) for act, p in ctx.probabilities.items()
                if act not in targets and act in ctx.all_activities and p > 0
            ]
            if candidates:
                probs = np.array([p for _, p in candidates], dtype=float)
                pick = candidates[np.random.choice(len(candidates), p=probs / probs.sum())][0]
                mask[ctx.all_activities.index(pick)] = 1.0
            else:
                mask[-1] = 1.0  # END as last resort
        return mask


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