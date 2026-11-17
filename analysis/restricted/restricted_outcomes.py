"""
Outcome-restriction analysis: compare policies within each restricted scenario
(and against their unrestricted `flex` counterparts).

Run from the project root after `python src/run_restricted_evaluation.py`:

    python analysis/restricted/restricted_outcomes.py

Metric definitions follow analysis/cancellations/cancelation.ipynb (events/case,
focus-end share, TVD of endings, canceled share) and
analysis/trace_similarity/main.ipynb (variant coverage = share of simulated
cases whose activity sequence occurs in the training log). SLA thresholds come
from PolicyEvaluator on the original training log, as in the paper.
CIR is computed against DM-RR *of the same environment* (restricted vs.
restricted, unrestricted vs. unrestricted).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from evaluation.metrics.functions.aggregation import mean_and_ci  # noqa: E402
from evaluation.metrics.functions.compliance import compute_compliance_rate  # noqa: E402
from evaluation.metrics.functions.cycle_time import compute_cycle_times  # noqa: E402
from evaluation.metrics.policy_evaluator import PolicyEvaluator  # noqa: E402
from run_restricted_evaluation import SCENARIOS  # noqa: E402

RESTRICTED_DIR = Path("data/evaluation_results_restricted")
UNRESTRICTED_DIR = Path("data/evaluation_results/flex")
POLICIES = ["DM-RR", "DM-DRL", "DRL-DRL"]
SLA_PCTS = [95, 90, 75, 50]
CANCEL_ACTIVITIES = {
    "AcademicCredentials": {"Notificacion estudiante cancelacion soli", "Cancelar Solicitud", "Cancelar curso"},
}
LABEL = {"AcademicCredentials": "AC_CRE", "BPIC_2012": "BPIC12_W", "BPIC_2017": "BPIC17_W"}


def load(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["activity"] = df["activity"].astype(str).str.strip()
    return df


def case_table(df: pd.DataFrame) -> pd.DataFrame:
    s = df.sort_values(["case_id", "start_time", "end_time"], kind="stable")
    g = s.groupby("case_id", sort=True)
    e = df.sort_values(["case_id", "end_time", "start_time"], kind="stable").groupby("case_id", sort=True)
    return pd.DataFrame({
        "length": g.size(),
        "trace": g["activity"].agg(tuple),
        "end_activity": e["activity"].last(),
    })


def violations(cases: pd.Series, restrictions: dict) -> int:
    forbidden = {(src, None if t == "END" else t) for src, ts in restrictions.items() for t in ts}
    any_src = {t for s, t in forbidden if s == "*"}
    n = 0
    for tr in cases:
        seq = list(tr) + [None]
        for a, b in zip(seq[:-1], seq[1:]):
            if b in any_src or (a, b) in forbidden:
                n += 1
    return n


class Reference:
    def __init__(self, log: str):
        path = f"data/logs/{log}/{log}_train.csv"
        ev = PolicyEvaluator(path, {"case": "case_id", "start": "start_time", "end": "end_time"},
                             sla_percentiles=SLA_PCTS)
        self.thresholds = ev.sla_thresholds
        cases = case_table(load(path))
        self.end_dist = cases["end_activity"].value_counts(normalize=True)
        self.focus_end = self.end_dist.idxmax()
        self.variants = set(cases["trace"])
        self.cases = cases
        raw = pd.read_csv(path)
        self.ct = compute_cycle_times(raw, "case_id", "start_time", "end_time")


def run_metrics(path, log: str, ref: Reference, restrictions: dict | None) -> dict:
    df = load(path)
    cases = case_table(df)
    ct = compute_cycle_times(pd.read_csv(path), "case_id", "start_time", "end_time")
    end_dist = cases["end_activity"].value_counts(normalize=True)
    acts = ref.end_dist.index.union(end_dist.index)
    m = {
        "n_cases": len(cases),
        "events_per_case": cases["length"].mean(),
        "focus_end_share": end_dist.get(ref.focus_end, 0.0),
        "dominant_end": end_dist.idxmax(),
        "dominant_end_share": end_dist.max(),
        "tvd_end": 0.5 * (ref.end_dist.reindex(acts, fill_value=0) - end_dist.reindex(acts, fill_value=0)).abs().sum(),
        "var_cov_cases": cases["trace"].map(lambda v: v in ref.variants).mean(),
        "n_distinct_variants": cases["trace"].nunique(),
    }
    if log in CANCEL_ACTIVITIES:
        m["canceled_share"] = cases["end_activity"].isin(CANCEL_ACTIVITIES[log]).mean()
    for label, T in ref.thresholds.items():
        m[f"cr_{label}"] = compute_compliance_rate(ct, T)
    if restrictions:
        m["violations"] = violations(cases["trace"], restrictions)
    return m


def restriction_counts(scenario_dir: Path) -> pd.DataFrame:
    """Per-run blocked/fallback counts, from <log>/<policy>/restriction_counts.csv."""
    frames = []
    for p in scenario_dir.glob("*/*/restriction_counts.csv"):
        d = pd.read_csv(p)
        d["policy"] = p.parent.name
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    refs: dict[str, Reference] = {}
    rows = []
    for scen, (log, restr, _coverage) in SCENARIOS.items():
        sdir = RESTRICTED_DIR / scen
        if not sdir.exists():
            print(f"[skip] {scen}: not run yet")
            continue
        ref = refs.setdefault(log, Reference(log))
        counts = restriction_counts(sdir)
        # original log row (reference)
        om = run_metrics(f"data/logs/{log}/{log}_train.csv", log, ref, restr)
        rows.append({"scenario": scen, "log": LABEL[log], "env": "original", "policy": "Original (train)",
                     "run_id": 0, **om})
        for env, base in (("restricted", sdir), ("unrestricted", UNRESTRICTED_DIR)):
            for pol in POLICIES:
                files = sorted(glob.glob(str(base / log / pol / "simulated_logs" / "sim_run_*.csv")))
                for i, f in enumerate(files, 1):
                    m = run_metrics(f, log, ref, restr)
                    if env == "restricted" and not counts.empty:
                        c = counts[(counts.policy == pol) & (counts.run_id == i)]
                        if len(c):
                            m["blocked"] = float(c["restr_blocked"].iloc[0])
                            m["fallback"] = float(c["restr_fallback"].iloc[0])
                            m["fallback_rate"] = m["fallback"] / max(float(c["num_decisions"].iloc[0]), 1)
                    rows.append({"scenario": scen, "log": LABEL[log], "env": env, "policy": pol, "run_id": i, **m})

    runs = pd.DataFrame(rows)
    # CIR vs DM-RR of the same environment (mean CR of DM-RR as reference)
    for label in [f"T{p}" for p in SLA_PCTS]:
        ref_cr = runs[runs.policy == "DM-RR"].groupby(["scenario", "env"])[f"cr_{label}"].mean()
        runs[f"cir_{label}_vs_dmrr"] = [
            (r[f"cr_{label}"] - ref_cr.get((r.scenario, r.env), np.nan)) / ref_cr.get((r.scenario, r.env), np.nan)
            if r.env != "original" else np.nan
            for _, r in runs.iterrows()
        ]
    RESTRICTED_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RESTRICTED_DIR / "summary_runs.csv", index=False)

    num_cols = [c for c in runs.columns if c not in ("scenario", "log", "env", "policy", "run_id", "dominant_end")
                and pd.api.types.is_numeric_dtype(runs[c])]
    agg_rows = []
    for key, g in runs.groupby(["scenario", "log", "env", "policy"], sort=False):
        row = dict(zip(["scenario", "log", "env", "policy"], key), n_runs=len(g),
                   dominant_end=g["dominant_end"].mode().iloc[0])
        for c in num_cols:
            v = g[c].dropna()
            if len(v):
                row[f"{c}_mean"], row[f"{c}_ci95"] = mean_and_ci(v.astype(float).tolist())
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(RESTRICTED_DIR / "summary_aggregated.csv", index=False)

    # compact console view
    def f(r, c, pct=True, d=1):
        m, ci = r.get(f"{c}_mean"), r.get(f"{c}_ci95")
        if m is None or pd.isna(m):
            return "-"
        k = 100 if pct else 1
        return f"{m*k:.{d}f}" if (pd.isna(ci) or ci == 0) else f"{m*k:.{d}f}±{ci*k:.{d}f}"

    view = pd.DataFrame([{
        "scenario": r.scenario, "env": r.env, "policy": r.policy,
        "CR75%": f(r, "cr_T75"), "CIR75 vs DM-RR%": f(r, "cir_T75_vs_dmrr"),
        "ev/case": f(r, "events_per_case", False, 2), "focus_end%": f(r, "focus_end_share"),
        "dom_end": str(r.dominant_end)[:28], "dom%": f(r, "dominant_end_share"),
        "TVD": f(r, "tvd_end", False, 3), "var_cov%": f(r, "var_cov_cases"),
        "cancel%": f(r, "canceled_share"), "blocked": f(r, "blocked", False, 0),
        "fallback%": f(r, "fallback_rate", True, 2), "viol": f(r, "violations", False, 0),
    } for _, r in agg.iterrows()])
    with pd.option_context("display.width", 260, "display.max_columns", 30, "display.max_colwidth", 30):
        for scen, g in view.groupby("scenario", sort=False):
            print(f"\n=== {scen}  {json.dumps(SCENARIOS[scen][1], ensure_ascii=False)}")
            print(g.drop(columns="scenario").to_string(index=False))
    print(f"\n-> {RESTRICTED_DIR / 'summary_runs.csv'}\n-> {RESTRICTED_DIR / 'summary_aggregated.csv'}")


if __name__ == "__main__":
    main()
