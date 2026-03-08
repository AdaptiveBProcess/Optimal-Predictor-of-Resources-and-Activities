"""
Training Metrics Tracker for OPRA.

Records per-episode and per-update metrics during DRL training,
and persists them as CSV + JSON for analysis and plotting.

Usage:
    tracker = TrainingMetricsTracker(log_dir="data/training_runs/run_01")
    tracker.log_episode(episode=1, metrics={...})
    tracker.log_update(update=1, metrics={...})
    tracker.save()
"""

import os
import json
import csv
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics collected at the end of each training episode (one full simulation)."""

    episode: int
    total_reward: float
    num_steps: int  # decision points in this episode

    # --- SLA Compliance (primary objective) ---
    num_cases: int
    num_compliant: int
    sla_compliance_rate: float  # CR = num_compliant / num_cases

    # --- Cycle Time Stats ---
    avg_cycle_time: float
    median_cycle_time: float
    std_cycle_time: float
    min_cycle_time: float
    max_cycle_time: float
    p75_cycle_time: float
    p90_cycle_time: float
    p95_cycle_time: float

    # --- Wall-clock ---
    episode_duration_sec: float  # how long the episode took (seconds)

    # --- Optional: resource utilization ---
    resource_utilization_cv: Optional[float] = None  # CV of utilization across resources


@dataclass
class UpdateMetrics:
    """Metrics collected after each PPO policy update."""

    update: int
    episode: int  # which episode triggered this update
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approx_kl: Optional[float] = None
    clip_fraction: Optional[float] = None


class TrainingMetricsTracker:
    """
    Accumulates training metrics and persists them.

    Produces:
        - episode_metrics.csv  (one row per episode)
        - update_metrics.csv   (one row per PPO update)
        - summary.json         (best episode, final stats, hyperparams)
    """

    def __init__(self, log_dir: str, hyperparams: Optional[Dict] = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.episode_history: List[EpisodeMetrics] = []
        self.update_history: List[UpdateMetrics] = []
        self.hyperparams = hyperparams or {}

        # Running best
        self._best_compliance = -1.0
        self._best_episode = -1

    # ------------------------------------------------------------------ #
    #  Logging helpers
    # ------------------------------------------------------------------ #

    def log_episode(self, metrics: EpisodeMetrics):
        """Record one episode's metrics."""
        self.episode_history.append(metrics)

        if metrics.sla_compliance_rate > self._best_compliance:
            self._best_compliance = metrics.sla_compliance_rate
            self._best_episode = metrics.episode

    def log_update(self, metrics: UpdateMetrics):
        """Record one PPO update's metrics."""
        self.update_history.append(metrics)

    # ------------------------------------------------------------------ #
    #  Computed summaries
    # ------------------------------------------------------------------ #

    def recent_avg(self, window: int = 10, key: str = "sla_compliance_rate") -> float:
        """Moving average of a metric over the last `window` episodes."""
        if not self.episode_history:
            return 0.0
        recent = self.episode_history[-window:]
        return float(np.mean([getattr(m, key) for m in recent]))

    def improvement_over_baseline(self, baseline_cr: float) -> Optional[float]:
        """CIR of the latest episode relative to a baseline compliance rate."""
        if not self.episode_history or baseline_cr == 0:
            return None
        latest_cr = self.episode_history[-1].sla_compliance_rate
        return (latest_cr - baseline_cr) / baseline_cr

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def save(self):
        """Write all accumulated metrics to disk."""
        self._save_csv(
            os.path.join(self.log_dir, "episode_metrics.csv"),
            self.episode_history,
        )
        self._save_csv(
            os.path.join(self.log_dir, "update_metrics.csv"),
            self.update_history,
        )
        self._save_summary()

    def _save_csv(self, path: str, records: list):
        if not records:
            return
        fieldnames = list(asdict(records[0]).keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow(asdict(rec))

    def _save_summary(self):
        summary = {
            "total_episodes": len(self.episode_history),
            "total_updates": len(self.update_history),
            "best_episode": self._best_episode,
            "best_sla_compliance_rate": self._best_compliance,
            "hyperparams": self.hyperparams,
        }
        if self.episode_history:
            last = self.episode_history[-1]
            summary["final_sla_compliance_rate"] = last.sla_compliance_rate
            summary["final_avg_cycle_time"] = last.avg_cycle_time
            summary["final_total_reward"] = last.total_reward

        with open(os.path.join(self.log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------ #
    #  Console printing
    # ------------------------------------------------------------------ #

    def print_episode_summary(self, metrics: EpisodeMetrics, baseline_cr: Optional[float] = None):
        """Pretty-print a single episode's results to console."""
        cir_str = ""
        if baseline_cr and baseline_cr > 0:
            cir = (metrics.sla_compliance_rate - baseline_cr) / baseline_cr
            cir_str = f"  CIR vs baseline: {cir:+.2%}"

        print(
            f"[Episode {metrics.episode:>4d}] "
            f"Reward={metrics.total_reward:>7.2f}  "
            f"CR={metrics.sla_compliance_rate:.2%}  "
            f"AvgCT={metrics.avg_cycle_time:.1f}  "
            f"MedCT={metrics.median_cycle_time:.1f}  "
            f"Steps={metrics.num_steps}  "
            f"Cases={metrics.num_cases}  "
            f"Time={metrics.episode_duration_sec:.1f}s"
            f"{cir_str}"
        )

    def print_update_summary(self, metrics: UpdateMetrics):
        print(
            f"  [Update {metrics.update:>3d}] "
            f"PolicyLoss={metrics.policy_loss:.4f}  "
            f"ValueLoss={metrics.value_loss:.4f}  "
            f"Entropy={metrics.entropy:.4f}  "
            f"TotalLoss={metrics.total_loss:.4f}"
        )


# -------------------------------------------------------------------- #
#  Helper: compute episode metrics from a completed simulation
# -------------------------------------------------------------------- #

def compute_episode_metrics(
    episode: int,
    total_reward: float,
    num_steps: int,
    cycle_times: List[float],
    sla_threshold: float,
    episode_duration_sec: float,
    resource_utilizations: Optional[List[float]] = None,
) -> EpisodeMetrics:
    """
    Build an EpisodeMetrics from raw simulation outputs.

    Args:
        cycle_times: list of cycle times (in seconds) for each completed case.
        sla_threshold: the T used for compliance.
        resource_utilizations: optional list of per-resource utilization ratios.
    """
    ct = np.array(cycle_times) if cycle_times else np.array([0.0])
    num_cases = len(ct)
    num_compliant = int(np.sum(ct < sla_threshold))

    util_cv = None
    if resource_utilizations and len(resource_utilizations) > 1:
        mean_u = np.mean(resource_utilizations)
        if mean_u > 0:
            util_cv = float(np.std(resource_utilizations) / mean_u)

    return EpisodeMetrics(
        episode=episode,
        total_reward=total_reward,
        num_steps=num_steps,
        num_cases=num_cases,
        num_compliant=num_compliant,
        sla_compliance_rate=num_compliant / max(num_cases, 1),
        avg_cycle_time=float(np.mean(ct)),
        median_cycle_time=float(np.median(ct)),
        std_cycle_time=float(np.std(ct)),
        min_cycle_time=float(np.min(ct)),
        max_cycle_time=float(np.max(ct)),
        p75_cycle_time=float(np.percentile(ct, 75)),
        p90_cycle_time=float(np.percentile(ct, 90)),
        p95_cycle_time=float(np.percentile(ct, 95)),
        episode_duration_sec=episode_duration_sec,
        resource_utilization_cv=util_cv,
    )
