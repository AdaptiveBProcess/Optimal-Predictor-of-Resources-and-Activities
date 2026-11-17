"""
Outcome-restriction scenarios: run a set of policies under transition
restrictions (what-if analysis) and write results per scenario.

Run from the project root:

    python src/run_restricted_evaluation.py                       # all scenarios
    python src/run_restricted_evaluation.py --scenarios AC_R_never_cancel
    python src/run_restricted_evaluation.py --K 2                 # quick smoke test

Output layout:
    data/evaluation_results_restricted/<scenario>/<log>/<policy>/   (simulated logs, results.json, restrictions.json)
    data/evaluation_results_restricted/<scenario>/{runs_long,runs_wide,aggregated}.csv

Each scenario is (log, rules, coverage).  Coverage is the fraction of cases the
rules apply to, assigned once per case (a restricted case obeys every rule, the
rest obey none).  It is either a number in [0, 1] or ``log_share``, which uses
the share of original-log cases that already obey the rules.

All policies in a scenario share the same restriction, and SLA thresholds come
from the original training log (same PolicyEvaluator as the paper's matrix).
Checkpoints/masks are the paper's "flex" configuration (nucleus mask, no KL).
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from evaluation.csv_export import write_all_csvs
from evaluation.experiment import evaluate_policy_on_log
from evaluation.restrictions import log_compliance_share
from run_matrix_evaluation import _checkpoint_for, _ckpt, _evaluator_for_log


def log_share(log_path: str, rules: dict) -> float:
    """Coverage = share of cases in the original log that already obey every rule."""
    share, _ = log_compliance_share(pd.read_csv(log_path), rules)
    return share


# Paper "flex" configuration (nucleus mask, kl0 checkpoints)
LOGS: dict[str, dict] = {
    "AcademicCredentials": {
        "path": "data/logs/AcademicCredentials/AcademicCredentials_train.csv",
        "checkpoint_full": _ckpt("AcademicCredentials_DDPS_p75_300_400_tp90_tk2_pe30_kl0_full"),
        "checkpoint_resource_only": _ckpt("AcademicCredentials_DDPS_p75_300_400_tp90_tk2_pe30_kl0_resource_only"),
        "masks": dict(top_k=2, top_p=0.9, p_min_end=0.3),
    },
    "BPIC_2012": {
        "path": "data/logs/BPIC_2012/BPIC_2012_train.csv",
        "checkpoint_full": _ckpt("BPIC_2012_DDPS_p75_300_3000_tp90_tk2_pe30_kl0_full"),
        "checkpoint_resource_only": _ckpt("BPIC_2012_DDPS_p75_300_3000_tp90_tk2_pe30_kl0_resource_only"),
        "masks": dict(top_k=2, top_p=0.9, p_min_end=0.3),
    },
    "BPIC_2017": {
        "path": "data/logs/BPIC_2017/BPIC_2017_train.csv",
        "checkpoint_full": _ckpt("BPIC_2017_DDPS_p75_300_7400_tp90_tk3_pe10_kl0_full"),
        "checkpoint_resource_only": _ckpt("BPIC_2017_DDPS_p75_300_7400_tp90_tk3_pe10_kl0_resource_only"),
        "masks": dict(top_k=3, top_p=0.9, p_min_end=0.1),
    },
}

# scenario name -> (log, rules, coverage).  Listed in priority order.
SCENARIOS: dict[str, tuple] = {
    # BPIC17: after an offer call the case must move on (no END, no repeated call).
    "BPIC17_R_validate_after_call_offers": ("BPIC_2017", {
        "W_Call after offers": ["END", "W_Call after offers"],
    }, log_share),
    # BPIC12: a lead must be followed up.
    "BPIC12_R1_no_end_after_leads": ("BPIC_2012", {
        "W_Afhandelen leads": ["END"],
    }, log_share),
    # AC: never cancel, and a case cannot end right after course homologation.
    # Applied to as many cases as obey these rules in the original log (28.6%).
    "AC_R_never_cancel": ("AcademicCredentials", {
        "*": ["Cancelar Solicitud", "Cancelar curso", "Notificacion estudiante cancelacion soli"],
        "Homologacion por grupo de cursos": ["END"],
    }, log_share),
}

POLICIES = ["DM-RR", "DM-DRL", "DRL-DRL"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OPRA outcome-restriction scenarios")
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    p.add_argument("--policies", nargs="+", default=POLICIES)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--sla-percentiles", type=int, nargs="+", default=[95, 90, 75, 50])
    p.add_argument("--train-percentile", type=int, default=75)
    p.add_argument("--output-dir", default="data/evaluation_results_restricted")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    unknown = [s for s in args.scenarios if s not in SCENARIOS]
    if unknown:
        print(f"Unknown scenarios: {unknown}. Known: {list(SCENARIOS)}", file=sys.stderr)
        return 1

    for i, name in enumerate(args.scenarios, 1):
        log_name, restrictions, coverage = SCENARIOS[name]
        cfg = LOGS[log_name]
        if callable(coverage):
            coverage = coverage(cfg["path"], restrictions)
        out_dir = f"{args.output_dir}/{name}"
        print(
            f"\n{'=' * 70}\n[{i}/{len(args.scenarios)}] {name}  ({log_name})  "
            f"coverage={coverage:.1%}  {restrictions}\n{'=' * 70}"
        )
        evaluator = _evaluator_for_log(cfg, args.sla_percentiles)
        records: list[dict] = []
        t0 = time.time()
        for policy in args.policies:
            _, recs = evaluate_policy_on_log(
                log_path=cfg["path"],
                policy_name=policy,
                K=args.K,
                sla_percentiles=tuple(args.sla_percentiles),
                train_percentile=args.train_percentile,
                checkpoint=_checkpoint_for(policy, cfg),
                masks_cfg=cfg["masks"],
                output_dir=out_dir,
                log_name=log_name,
                seed=args.seed,
                evaluator=evaluator,
                restrictions=restrictions,
                restriction_coverage=coverage,
            )
            records.extend(recs)
        paths = write_all_csvs(records, out_dir)
        print(f"[{name}] done in {(time.time() - t0) / 60:.1f} min -> {paths}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
