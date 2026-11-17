"""Helpers for outcome restrictions that look at the original event log."""

from __future__ import annotations

import pandas as pd

from environment.core.mask import NucleusMaskFunction, TransitionRestrictionMaskFunction


def log_compliance_share(
    log_df: pd.DataFrame,
    restrictions: dict,
    *,
    case_col: str = "case_id",
    activity_col: str = "activity",
    start_col: str = "start_time",
    end_col: str = "end_time",
) -> tuple[float, dict]:
    """
    Share of cases in ``log_df`` that already obey every rule in ``restrictions``.

    Uses the same semantics as ``TransitionRestrictionMaskFunction``: a case
    violates the rules if any consecutive pair ``(prev, next)`` has ``next`` in
    ``forbidden[prev] | forbidden["*"]``, where the case start is ``prev=None``
    and the case end is ``next=None`` ("END").  Activity names are resolved
    (and validated) exactly as in the simulator.

    Returns ``(share_compliant, per_rule)`` where ``per_rule`` maps
    ``"src -> target"`` to the share of cases that contain that transition.
    """
    rules = TransitionRestrictionMaskFunction(NucleusMaskFunction(), restrictions)
    rules.validate(list(log_df[activity_col].dropna().unique()) + [None])

    df = log_df.sort_values([case_col, start_col, end_col], kind="stable")
    n_cases = 0
    n_compliant = 0
    hits: dict[str, int] = {}
    for _, acts in df.groupby(case_col, sort=False)[activity_col]:
        trace = [None] + list(acts) + [None]  # start marker, trace, END
        n_cases += 1
        violated = set()
        for prev, nxt in zip(trace[:-1], trace[1:]):
            if prev is None and nxt is None:
                continue
            if nxt in rules._targets_for(prev):
                violated.add(f"{'START' if prev is None else prev} -> {'END' if nxt is None else nxt}")
        for key in violated:
            hits[key] = hits.get(key, 0) + 1
        n_compliant += not violated

    share = n_compliant / n_cases if n_cases else 0.0
    per_rule = {k: v / n_cases for k, v in sorted(hits.items(), key=lambda kv: -kv[1])}
    return share, per_rule
