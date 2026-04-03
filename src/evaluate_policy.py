"""
OPRA Evaluation Script.

Loads a trained model checkpoint and runs K independent simulations
to evaluate the policy using the full metrics suite from the thesis.

Usage:
    python src/evaluate_policy.py --checkpoint data/training_runs/run_01/checkpoints/best_model.pt
    python src/evaluate_policy.py --checkpoint best_model.pt --K 10 --policy_name DRL-AR
"""

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from environment.simulator.adapters.event_log_to_csv import export_event_log_to_csv
from initializer.implementations.DDPSInitializer import DDPSInitializer
from environment.simulator.core.setup import SimulationSetup
from environment.core.env import BusinessProcessEnvironment
from environment.core.mask import NucleusMaskFunction
from environment.simulator.core.log_names import LogColumnNames
from environment.simulator.core.engine import SimulatorEngine
from agent.agent import PPOAgent

from metrics.evaluation.policy_evaluator import PolicyEvaluator
from train import run_single_episode, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="OPRA Policy Evaluation")
    parser.add_argument("--log_path", type=str, default="data/logs/LoanApp/LoanApp.csv")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--K", type=int, default=10, help="Number of evaluation runs")
    parser.add_argument("--max_cases", type=int, default=None, help="Cases per run (default: same as original log)")
    parser.add_argument("--percentile", type=int, default=95, help="SLA percentile threshold")
    parser.add_argument("--policy_name", type=str, default="DRL-AR")
    parser.add_argument("--log_name", type=str, default="LoanApp")
    parser.add_argument("--output_dir", type=str, default="data/evaluation_results")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--p_min_end", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def run_evaluation():
    args = parse_args()

    # --- Load original log ---
    log = pd.read_csv(args.log_path)
    log_names = LogColumnNames(
        case_id="case_id",
        activity="activity",
        resource="resource",
        start_timestamp="start_time",
        end_timestamp="end_time",
    )

    # --- Setup ---
    initializer = DDPSInitializer()
    start_timestamp = log[log_names.start_timestamp].min()
    time_unit = "seconds"
    setup: SimulationSetup = initializer.build(log, log_names, start_timestamp, time_unit)

    # --- Determine max_cases ---
    num_original_cases = log[log_names.case_id].nunique()
    max_cases = args.max_cases or num_original_cases
    print(f"Original log: {num_original_cases} cases. Simulating {max_cases} per run.")

    # --- SLA threshold ---
    original_cycle_times = []
    for case_id, group in log.groupby(log_names.case_id):
        st = pd.to_datetime(group[log_names.start_timestamp], format="mixed").min()
        et = pd.to_datetime(group[log_names.end_timestamp], format="mixed").max()
        original_cycle_times.append((et - st).total_seconds())
    sla_threshold = np.percentile(original_cycle_times, args.percentile)

    # --- Create evaluator (handles reference CRs and all thresholds) ---
    evaluator = PolicyEvaluator(
        original_log_path=args.log_path,
        original_log_names={
            "case": log_names.case_id,
            "start": log_names.start_timestamp,
            "end": log_names.end_timestamp,
        },
        sla_percentiles=[95, 90, 75, 50],
    )

    # --- Build agent and load checkpoint ---
    simulator = SimulatorEngine(setup)
    env = BusinessProcessEnvironment(
        simulator,
        sla_threshold=sla_threshold,
        max_cases=max_cases,
        activity_mask_function=NucleusMaskFunction(k=args.top_k, p=args.top_p, p_min_end=args.p_min_end),
    )
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        num_activities=simulator.num_activities,
        num_resources=simulator.num_resources,
    )
    load_checkpoint(agent, args.checkpoint)

    # --- Run K evaluation simulations ---
    sim_log_dir = os.path.join(args.output_dir, args.log_name, args.policy_name, "simulated_logs")
    os.makedirs(sim_log_dir, exist_ok=True)

    simulated_log_paths = []

    print(f"\nRunning {args.K} evaluation simulations with policy '{args.policy_name}'...")

    for k in range(1, args.K + 1):
        seed = args.seed + k
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Fresh simulator and env for each run
        simulator_k = SimulatorEngine(setup)
        env_k = BusinessProcessEnvironment(
            simulator_k,
            sla_threshold=sla_threshold,
            max_cases=max_cases,
            activity_mask_function=NucleusMaskFunction(k=args.top_k, p=args.top_p, p_min_end=args.p_min_end),
        )

        t0 = time.time()
        total_reward, num_steps, cycle_times = run_single_episode(
            env=env_k,
            simulator=simulator_k,
            agent=agent,
            deterministic=False,  # Stochastic evaluation
        )
        duration = time.time() - t0

        # Convert to absolute timestamps for similarity metrics
        simulator_k._convert_event_log_to_absolute_time()

        # Export simulated log
        log_path = os.path.join(sim_log_dir, f"sim_run_{k:02d}.csv")
        export_event_log_to_csv(simulator_k.event_log, log_path)
        simulated_log_paths.append(log_path)

        ct = np.array(cycle_times)
        cr = float(np.mean(ct < sla_threshold)) if len(ct) > 0 else 0.0
        print(
            f"  Run {k:>2d}/{args.K}: "
            f"Cases={len(ct)}, Steps={num_steps}, "
            f"Reward={total_reward:.2f}, CR(p{args.percentile})={cr:.2%}, "
            f"AvgCT={np.mean(ct):.1f}, Time={duration:.1f}s"
        )

    # --- Compute all evaluation metrics ---
    print(f"\nComputing evaluation metrics across {args.K} runs...")
    results = evaluator.evaluate_policy(
        simulated_log_paths=simulated_log_paths,
        policy_name=args.policy_name,
        log_name=args.log_name,
    )

    # --- Print and save ---
    evaluator.print_results(results)
    policy_dir = os.path.join(args.output_dir, args.log_name, args.policy_name)
    evaluator.save_results(results, policy_dir)

    print(f"\nEvaluation complete. Results in: {args.output_dir}")


if __name__ == "__main__":
    run_evaluation()
