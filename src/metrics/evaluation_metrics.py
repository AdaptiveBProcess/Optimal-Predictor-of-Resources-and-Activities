"""
OPRA Evaluation Metrics.

Computes the full evaluation suite defined in the thesis (Section 8):
  - Performance: CR, CIR, avg/median/std cycle time, resource utilization CV
  - Similarity: NGD, AED, CED, RED, CWD, CAR, CTD (via log-distance-measures)

Usage:
    evaluator = PolicyEvaluator(original_log_path, sla_thresholds, log_names)
    results = evaluator.evaluate_simulated_log(simulated_log_path)
    evaluator.evaluate_policy(run_dir)  # evaluates all K=10 runs
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------- #
#  Performance Metrics (Section 8.2.1 of the thesis)
# ----------------------------------------------------------------------- #

@dataclass
class PerformanceResult:
    """Performance metrics for a single simulated log against reference thresholds."""
    # Per-threshold compliance
    compliance_rates: Dict[str, float]  # e.g. {"T95": 0.98, "T90": 0.95, ...}
    compliance_improvement_ratios: Dict[str, float]  # CIR relative to original

    # Cycle time statistics
    avg_cycle_time: float
    median_cycle_time: float
    std_cycle_time: float
    min_cycle_time: float
    max_cycle_time: float
    p75_cycle_time: float
    p90_cycle_time: float
    p95_cycle_time: float

    # Resource utilization balance
    resource_utilization_cv: Optional[float] = None

    # Metadata
    num_cases: int = 0
    log_path: str = ""


@dataclass
class SimilarityResult:
    """Similarity metrics comparing a simulated log to the original (Section 8.2.2)."""
    ngd: Optional[float] = None   # N-gram distance (control-flow)
    aed: Optional[float] = None   # Absolute event distribution (temporal)
    ced: Optional[float] = None   # Circadian event distribution (temporal)
    red: Optional[float] = None   # Relative event distribution (temporal)
    cwd: Optional[float] = None   # Circadian workforce distribution (resource)
    car: Optional[float] = None   # Case arrival rate (congestion)
    ctd: Optional[float] = None   # Cycle time distribution (congestion)


@dataclass
class AggregatedResults:
    """Mean ± 95% CI across K simulation runs for one policy-log combination."""
    policy_name: str
    log_name: str
    num_runs: int

    # Performance: mean ± ci
    compliance_rates_mean: Dict[str, float] = field(default_factory=dict)
    compliance_rates_ci: Dict[str, float] = field(default_factory=dict)
    cir_mean: Dict[str, float] = field(default_factory=dict)
    cir_ci: Dict[str, float] = field(default_factory=dict)

    avg_cycle_time_mean: float = 0.0
    avg_cycle_time_ci: float = 0.0
    median_cycle_time_mean: float = 0.0
    median_cycle_time_ci: float = 0.0
    std_cycle_time_mean: float = 0.0
    resource_utilization_cv_mean: Optional[float] = None

    # Similarity: mean ± ci
    similarity_mean: Dict[str, float] = field(default_factory=dict)
    similarity_ci: Dict[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------------- #
#  Computation Functions
# ----------------------------------------------------------------------- #

def compute_cycle_times(log_df: pd.DataFrame, case_col: str, start_col: str, end_col: str) -> np.ndarray:
    """Compute end-to-end cycle time per case (in seconds)."""
    cycle_times = []
    for _, group in log_df.groupby(case_col):
        st = pd.to_datetime(group[start_col], format='ISO8601').min()
        et = pd.to_datetime(group[end_col], format='ISO8601').max()
        cycle_times.append((et - st).total_seconds())
    return np.array(cycle_times)


def compute_compliance_rate(cycle_times: np.ndarray, threshold: float) -> float:
    """CR(L, T) = |{σ ∈ L : ct(σ) < T}| / |L|"""
    if len(cycle_times) == 0:
        return 0.0
    return float(np.mean(cycle_times < threshold))


def compute_cir(sim_cr: float, ref_cr: float) -> float:
    """CIR = (CR_sim - CR_ref) / CR_ref"""
    if ref_cr == 0:
        return float('inf') if sim_cr > 0 else 0.0
    return (sim_cr - ref_cr) / ref_cr


def compute_resource_utilization_cv(
    log_df: pd.DataFrame, 
    resource_col: str, 
    start_col: str, 
    end_col: str,
) -> Optional[float]:
    """
    CV of resource utilization = SD(utilizations) / mean(utilizations).
    Utilization = total busy time / total available time for each resource.
    Available time is approximated as (max_end - min_start) of the entire log.
    """
    if resource_col not in log_df.columns:
        return None

    log_df = log_df.copy()
    log_df[start_col] = pd.to_datetime(log_df[start_col], format='ISO8601')
    log_df[end_col] = pd.to_datetime(log_df[end_col], format='ISO8601')

    horizon_start = log_df[start_col].min()
    horizon_end = log_df[end_col].max()
    total_horizon = (horizon_end - horizon_start).total_seconds()

    if total_horizon <= 0:
        return None

    utilizations = []
    for _, group in log_df.groupby(resource_col):
        busy_time = (group[end_col] - group[start_col]).dt.total_seconds().sum()
        utilizations.append(busy_time / total_horizon)

    utilizations = np.array(utilizations)
    mean_u = np.mean(utilizations)
    if mean_u <= 0:
        return None
    return float(np.std(utilizations) / mean_u)


def compute_performance_metrics(
    sim_log_df: pd.DataFrame,
    ref_cycle_times: np.ndarray,
    sla_thresholds: Dict[str, float],
    ref_compliance_rates: Dict[str, float],
    case_col: str = "case",
    start_col: str = "start",
    end_col: str = "end",
    resource_col: str = "resource",
) -> PerformanceResult:
    """Compute all performance metrics for one simulated log."""
    sim_ct = compute_cycle_times(sim_log_df, case_col, start_col, end_col)

    compliance_rates = {}
    cir_values = {}
    for label, threshold in sla_thresholds.items():
        cr = compute_compliance_rate(sim_ct, threshold)
        compliance_rates[label] = cr
        cir_values[label] = compute_cir(cr, ref_compliance_rates.get(label, 0.0))

    util_cv = compute_resource_utilization_cv(sim_log_df, resource_col, start_col, end_col)

    return PerformanceResult(
        compliance_rates=compliance_rates,
        compliance_improvement_ratios=cir_values,
        avg_cycle_time=float(np.mean(sim_ct)),
        median_cycle_time=float(np.median(sim_ct)),
        std_cycle_time=float(np.std(sim_ct)),
        min_cycle_time=float(np.min(sim_ct)) if len(sim_ct) > 0 else 0.0,
        max_cycle_time=float(np.max(sim_ct)) if len(sim_ct) > 0 else 0.0,
        p75_cycle_time=float(np.percentile(sim_ct, 75)) if len(sim_ct) > 0 else 0.0,
        p90_cycle_time=float(np.percentile(sim_ct, 90)) if len(sim_ct) > 0 else 0.0,
        p95_cycle_time=float(np.percentile(sim_ct, 95)) if len(sim_ct) > 0 else 0.0,
        resource_utilization_cv=util_cv,
        num_cases=len(sim_ct),
    )


def compute_similarity_metrics(
    original_log_path: str,
    simulated_log_path: str,
) -> SimilarityResult:
    """
    Compute all 7 similarity metrics using the log-distance-measures package.
    
    Requires: pip install log-distance-measures
    
    Falls back gracefully if the package is not installed.
    """
    result = SimilarityResult()

    try:
        from log_distance_measures.config import EventLogIDs
        from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
        from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
        from log_distance_measures.circadian_event_distribution import circadian_event_distribution_distance
        from log_distance_measures.relative_event_distribution import relative_event_distribution_distance
        from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance
        from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
        from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
    except ImportError:
        print("  WARNING: log-distance-measures not installed. Skipping similarity metrics.")
        print("  Install with: pip install log-distance-measures")
        return result

    # Load logs
    original = pd.read_csv(original_log_path)
    simulated = pd.read_csv(simulated_log_path)

    # Configure column mappings for the library
    # NOTE: Adjust these EventLogIDs to match your column names
    original_ids = EventLogIDs(
        case="case_id",
        activity="activity",
        resource="resource",
        start_time="start_time",
        end_time="end_time",
    )
    simulated_ids = EventLogIDs(
        case="case",
        activity="activity",
        resource="resource",
        start_time="start",
        end_time="end",
    )

    # Ensure datetime columns (utc=True for consistency with log-distance-measures)
    for col in [original_ids.start_time, original_ids.end_time]:
        original[col] = pd.to_datetime(original[col], format='ISO8601', utc=True)
    for col in [simulated_ids.start_time, simulated_ids.end_time]:
        simulated[col] = pd.to_datetime(simulated[col], format='ISO8601', utc=True)

    try:
        result.ngd = n_gram_distribution_distance(original, original_ids, simulated, simulated_ids)
    except Exception as e:
        print(f"  NGD computation failed: {e}")

    try:
        result.aed = absolute_event_distribution_distance(original, original_ids, simulated, simulated_ids)
    except Exception as e:
        print(f"  AED computation failed: {e}")

    try:
        result.ced = circadian_event_distribution_distance(original, original_ids, simulated, simulated_ids)
    except Exception as e:
        print(f"  CED computation failed: {e}")

    try:
        result.red = relative_event_distribution_distance(original, original_ids, simulated, simulated_ids)
    except Exception as e:
        print(f"  RED computation failed: {e}")

    try:
        result.cwd = circadian_workforce_distribution_distance(original, original_ids, simulated, simulated_ids)
    except Exception as e:
        print(f"  CWD computation failed: {e}")

    try:
        result.car = case_arrival_distribution_distance(original, original_ids, simulated, simulated_ids)
    except Exception as e:
        print(f"  CAR computation failed: {e}")

    try:
        result.ctd = cycle_time_distribution_distance(original, original_ids, simulated, simulated_ids, bin_size=pd.Timedelta(hours=1))
    except Exception as e:
        print(f"  CTD computation failed: {e}")

    return result


# ----------------------------------------------------------------------- #
#  Aggregation across K runs (Section 8.3)
# ----------------------------------------------------------------------- #

def mean_and_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute mean and 95% CI using t-distribution."""
    from scipy import stats
    n = len(values)
    if n < 2:
        return (np.mean(values), 0.0)
    mean = np.mean(values)
    se = stats.sem(values)
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    ci = t_val * se
    return float(mean), float(ci)


def aggregate_results(
    performance_results: List[PerformanceResult],
    similarity_results: List[SimilarityResult],
    policy_name: str,
    log_name: str,
) -> AggregatedResults:
    """Aggregate K runs into mean ± 95% CI."""
    agg = AggregatedResults(
        policy_name=policy_name,
        log_name=log_name,
        num_runs=len(performance_results),
    )

    if not performance_results:
        return agg

    # --- Compliance rates ---
    threshold_labels = list(performance_results[0].compliance_rates.keys())
    for label in threshold_labels:
        crs = [r.compliance_rates[label] for r in performance_results]
        mean, ci = mean_and_ci(crs)
        agg.compliance_rates_mean[label] = mean
        agg.compliance_rates_ci[label] = ci

        cirs = [r.compliance_improvement_ratios[label] for r in performance_results]
        mean, ci = mean_and_ci(cirs)
        agg.cir_mean[label] = mean
        agg.cir_ci[label] = ci

    # --- Cycle time ---
    avg_cts = [r.avg_cycle_time for r in performance_results]
    agg.avg_cycle_time_mean, agg.avg_cycle_time_ci = mean_and_ci(avg_cts)

    med_cts = [r.median_cycle_time for r in performance_results]
    agg.median_cycle_time_mean, agg.median_cycle_time_ci = mean_and_ci(med_cts)

    std_cts = [r.std_cycle_time for r in performance_results]
    agg.std_cycle_time_mean = float(np.mean(std_cts))

    # --- Resource utilization ---
    util_cvs = [r.resource_utilization_cv for r in performance_results if r.resource_utilization_cv is not None]
    if util_cvs:
        agg.resource_utilization_cv_mean = float(np.mean(util_cvs))

    # --- Similarity ---
    sim_keys = ["ngd", "aed", "ced", "red", "cwd", "car", "ctd"]
    for key in sim_keys:
        vals = [getattr(r, key) for r in similarity_results if getattr(r, key) is not None]
        if vals:
            mean, ci = mean_and_ci(vals)
            agg.similarity_mean[key] = mean
            agg.similarity_ci[key] = ci

    return agg


# ----------------------------------------------------------------------- #
#  High-level evaluator
# ----------------------------------------------------------------------- #

class PolicyEvaluator:
    """
    Evaluates a trained policy by running K simulations and computing
    all performance + similarity metrics from the thesis.
    """

    def __init__(
        self,
        original_log_path: str,
        original_log_names: dict,  # {"case": ..., "activity": ..., ...}
        sla_percentiles: List[int] = [95, 90, 75, 50],
    ):
        self.original_log_path = original_log_path
        self.original_log_names = original_log_names

        # Load original log and compute reference values
        self.original_df = pd.read_csv(original_log_path)
        self.ref_cycle_times = compute_cycle_times(
            self.original_df,
            original_log_names["case"],
            original_log_names["start"],
            original_log_names["end"],
        )

        # Compute SLA thresholds
        self.sla_thresholds = {}
        self.ref_compliance_rates = {}
        for p in sla_percentiles:
            label = f"T{p}"
            threshold = float(np.percentile(self.ref_cycle_times, p))
            self.sla_thresholds[label] = threshold
            self.ref_compliance_rates[label] = compute_compliance_rate(
                self.ref_cycle_times, threshold
            )

        print(f"Original log: {len(self.ref_cycle_times)} cases")
        print(f"SLA thresholds: {self.sla_thresholds}")
        print(f"Reference CRs: {self.ref_compliance_rates}")

    def evaluate_single_log(
        self,
        sim_log_path: str,
        sim_case_col: str = "case",
        sim_start_col: str = "start",
        sim_end_col: str = "end",
        sim_resource_col: str = "resource",
    ) -> Tuple[PerformanceResult, SimilarityResult]:
        """Evaluate one simulated log file."""
        sim_df = pd.read_csv(sim_log_path)

        perf = compute_performance_metrics(
            sim_df,
            self.ref_cycle_times,
            self.sla_thresholds,
            self.ref_compliance_rates,
            case_col=sim_case_col,
            start_col=sim_start_col,
            end_col=sim_end_col,
            resource_col=sim_resource_col,
        )
        perf.log_path = sim_log_path

        sim = compute_similarity_metrics(self.original_log_path, sim_log_path)

        return perf, sim

    def evaluate_policy(
        self,
        simulated_log_paths: List[str],
        policy_name: str,
        log_name: str,
    ) -> AggregatedResults:
        """Evaluate K simulated logs for one policy and aggregate."""
        perf_results = []
        sim_results = []

        for path in simulated_log_paths:
            print(f"  Evaluating: {path}")
            perf, sim = self.evaluate_single_log(path)
            perf_results.append(perf)
            sim_results.append(sim)

        agg = aggregate_results(perf_results, sim_results, policy_name, log_name)
        return agg

    def save_results(self, results: AggregatedResults, output_dir: str):
        """Save aggregated results to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(
            output_dir,
            f"{results.policy_name}_{results.log_name}_results.json"
        )
        with open(path, "w") as f:
            json.dump(asdict(results), f, indent=2, default=str)
        print(f"  Results saved: {path}")

    def print_results(self, results: AggregatedResults):
        """Pretty-print aggregated results."""
        print(f"\n{'='*60}")
        print(f"Policy: {results.policy_name} | Log: {results.log_name} | Runs: {results.num_runs}")
        print(f"{'='*60}")

        print("\nSLA Compliance Rates (mean ± 95% CI):")
        for label in results.compliance_rates_mean:
            cr = results.compliance_rates_mean[label]
            ci = results.compliance_rates_ci.get(label, 0)
            cir = results.cir_mean.get(label, 0)
            cir_ci = results.cir_ci.get(label, 0)
            print(f"  {label}: CR = {cr:.2%} ± {ci:.2%}  |  CIR = {cir:+.2%} ± {cir_ci:.2%}")

        print(f"\nCycle Time:")
        print(f"  Mean:   {results.avg_cycle_time_mean:.1f} ± {results.avg_cycle_time_ci:.1f}")
        print(f"  Median: {results.median_cycle_time_mean:.1f} ± {results.median_cycle_time_ci:.1f}")
        print(f"  Std:    {results.std_cycle_time_mean:.1f}")

        if results.resource_utilization_cv_mean is not None:
            print(f"  Resource Util CV: {results.resource_utilization_cv_mean:.4f}")

        if results.similarity_mean:
            print(f"\nSimilarity Metrics (mean ± 95% CI):")
            for key, val in results.similarity_mean.items():
                ci = results.similarity_ci.get(key, 0)
                print(f"  {key.upper()}: {val:.4f} ± {ci:.4f}")
