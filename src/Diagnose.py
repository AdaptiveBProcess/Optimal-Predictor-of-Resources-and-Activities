"""
Diagnostic: Compare normal vs RL simulation to identify the time inflation bug.

This script runs both modes on the same setup and compares:
1. Cycle times (normal vs RL with random actions)
2. Number of pending decisions that accumulate
3. SimPy clock progression between decision points

Run from project root:
    python src/diagnose_time.py
"""

import random
import numpy as np
import pandas as pd

from initializer.implementations.DESInitializer import DESInitializer
from environment.simulator.core.setup import SimulationSetup
from environment.simulator.core.log_names import LogColumnNames
from environment.simulator.core.engine import SimulatorEngine


def run_normal_simulation(setup, max_cases=50):
    """Run standard DES (no RL)."""
    random.seed(42)
    np.random.seed(42)
    simulator = SimulatorEngine(setup)
    event_log = simulator.simulate(max_cases=max_cases)
    
    # Compute cycle times
    cases = {}
    for e in event_log:
        cid = e["case"]
        if cid not in cases:
            cases[cid] = {"start": e["start"], "end": e["end"]}
        else:
            cases[cid]["start"] = min(cases[cid]["start"], e["start"])
            cases[cid]["end"] = max(cases[cid]["end"], e["end"])
    
    cycle_times = [v["end"] - v["start"] for v in cases.values()]
    return cycle_times, event_log, simulator


def run_rl_simulation_random(setup, max_cases=50):
    """
    Run RL simulation with RANDOM valid actions.
    This isolates the simulator behavior from the agent.
    """
    random.seed(42)
    np.random.seed(42)
    simulator = SimulatorEngine(setup)
    simulator.reset(max_cases=max_cases)
    simulator.is_rl_mode = True
    
    # Track diagnostics
    time_gaps = []       # SimPy time between consecutive decisions
    pending_counts = []  # How many decisions are pending when we process one
    
    step = 0
    prev_time = 0.0
    
    while True:
        # Advance simulation until a decision is needed or done
        completed = simulator.run_until_decision()
        
        if simulator.all_done.triggered and not simulator.pending_decisions:
            break
        
        if not simulator.pending_decisions:
            continue
        
        # --- Diagnostic: how many are pending? ---
        pending_counts.append(len(simulator.pending_decisions))
        
        # --- Diagnostic: time gap ---
        current_time = simulator.env.now
        time_gaps.append(current_time - prev_time)
        prev_time = current_time
        
        # --- Process ONE decision (like the current training loop does) ---
        case = simulator.pending_decisions[0]["case"]
        
        # Pick a random valid activity from routing policy
        activity = simulator.setup.routing_policy.get_next_activity(case)
        if activity is None:
            # End of trace — apply None to finish the case
            simulator.apply_decision(None, None)
            step += 1
            continue
        
        # Pick a random valid resource
        resource = simulator.setup.resource_policy.select_resource(activity, case)
        simulator.apply_decision(activity, resource)
        step += 1
        
        if step % 100 == 0:
            print(f"  Step {step}: time={current_time:.0f}, pending={pending_counts[-1]}, gap={time_gaps[-1]:.0f}")
    
    # Compute cycle times
    cases = {}
    for e in simulator.event_log:
        cid = e["case"]
        if cid not in cases:
            cases[cid] = {"start": e["start"], "end": e["end"]}
        else:
            cases[cid]["start"] = min(cases[cid]["start"], e["start"])
            cases[cid]["end"] = max(cases[cid]["end"], e["end"])
    
    cycle_times = [v["end"] - v["start"] for v in cases.values()]
    return cycle_times, simulator.event_log, pending_counts, time_gaps


def run_rl_simulation_batch(setup, max_cases=50):
    """
    Run RL simulation processing ALL pending decisions before advancing.
    This should produce similar cycle times to normal simulation.
    """
    random.seed(42)
    np.random.seed(42)
    simulator = SimulatorEngine(setup)
    simulator.reset(max_cases=max_cases)
    simulator.is_rl_mode = True
    
    step = 0
    
    while True:
        completed = simulator.run_until_decision()
        
        if simulator.all_done.triggered and not simulator.pending_decisions:
            break
        
        # --- Process ALL pending decisions before advancing ---
        while simulator.pending_decisions:
            case = simulator.pending_decisions[0]["case"]
            
            activity = simulator.setup.routing_policy.get_next_activity(case)
            if activity is None:
                simulator.apply_decision(None, None)
                step += 1
                continue
            
            resource = simulator.setup.resource_policy.select_resource(activity, case)
            simulator.apply_decision(activity, resource)
            step += 1
        
        # Now all pending decisions are resolved, loop back to advance simulation
    
    # Compute cycle times
    cases = {}
    for e in simulator.event_log:
        cid = e["case"]
        if cid not in cases:
            cases[cid] = {"start": e["start"], "end": e["end"]}
        else:
            cases[cid]["start"] = min(cases[cid]["start"], e["start"])
            cases[cid]["end"] = max(cases[cid]["end"], e["end"])
    
    cycle_times = [v["end"] - v["start"] for v in cases.values()]
    return cycle_times, simulator.event_log


def main():
    # --- Setup ---
    log = pd.read_csv("data/logs/LoanApp/LoanApp.csv")
    log_names = LogColumnNames(
        case_id="case_id", activity="activity", resource="resource",
        start_timestamp="start_time", end_timestamp="end_time",
    )
    initializer = DESInitializer()
    start_timestamp = log[log_names.start_timestamp].min()
    setup = initializer.build(log, log_names, start_timestamp, "seconds")
    
    max_cases = 50
    
    # --- Original log stats ---
    original_cts = []
    for _, group in log.groupby(log_names.case_id):
        st = pd.to_datetime(group[log_names.start_timestamp]).min()
        et = pd.to_datetime(group[log_names.end_timestamp]).max()
        original_cts.append((et - st).total_seconds())
    original_cts = np.array(original_cts)
    
    print("=" * 70)
    print("ORIGINAL LOG")
    print(f"  Cases: {len(original_cts)}")
    print(f"  Avg CT: {np.mean(original_cts):.0f}s ({np.mean(original_cts)/3600:.1f}h)")
    print(f"  Median CT: {np.median(original_cts):.0f}s ({np.median(original_cts)/3600:.1f}h)")
    print(f"  p95 CT: {np.percentile(original_cts, 95):.0f}s")
    
    # --- Normal simulation ---
    print("\n" + "=" * 70)
    print("NORMAL SIMULATION (no RL)")
    normal_cts, normal_log, _ = run_normal_simulation(setup, max_cases)
    normal_cts = np.array(normal_cts)
    print(f"  Cases: {len(normal_cts)}")
    print(f"  Events: {len(normal_log)}")
    print(f"  Avg CT: {np.mean(normal_cts):.0f}s ({np.mean(normal_cts)/3600:.1f}h)")
    print(f"  Median CT: {np.median(normal_cts):.0f}s ({np.median(normal_cts)/3600:.1f}h)")
    print(f"  Avg events/case: {len(normal_log)/len(normal_cts):.1f}")
    
    # --- RL simulation (one-at-a-time, like current code) ---
    print("\n" + "=" * 70)
    print("RL SIMULATION - ONE DECISION AT A TIME (current behavior)")
    rl_cts, rl_log, pending_counts, time_gaps = run_rl_simulation_random(setup, max_cases)
    rl_cts = np.array(rl_cts)
    print(f"  Cases: {len(rl_cts)}")
    print(f"  Events: {len(rl_log)}")
    print(f"  Avg CT: {np.mean(rl_cts):.0f}s ({np.mean(rl_cts)/3600:.1f}h)")
    print(f"  Median CT: {np.median(rl_cts):.0f}s ({np.median(rl_cts)/3600:.1f}h)")
    print(f"  Avg events/case: {len(rl_log)/max(len(rl_cts),1):.1f}")
    print(f"  Max pending decisions at once: {max(pending_counts) if pending_counts else 0}")
    print(f"  Avg pending decisions: {np.mean(pending_counts) if pending_counts else 0:.1f}")
    print(f"  Avg time gap between decisions: {np.mean(time_gaps) if time_gaps else 0:.0f}s")
    
    # --- RL simulation (batch all pending, proposed fix) ---
    print("\n" + "=" * 70)
    print("RL SIMULATION - BATCH ALL PENDING (proposed fix)")
    batch_cts, batch_log = run_rl_simulation_batch(setup, max_cases)
    batch_cts = np.array(batch_cts)
    print(f"  Cases: {len(batch_cts)}")
    print(f"  Events: {len(batch_log)}")
    print(f"  Avg CT: {np.mean(batch_cts):.0f}s ({np.mean(batch_cts)/3600:.1f}h)")
    print(f"  Median CT: {np.median(batch_cts):.0f}s ({np.median(batch_cts)/3600:.1f}h)")
    print(f"  Avg events/case: {len(batch_log)/max(len(batch_cts),1):.1f}")
    
    # --- Comparison ---
    print("\n" + "=" * 70)
    print("COMPARISON (inflation factor = RL / Normal)")
    if len(normal_cts) > 0 and np.mean(normal_cts) > 0:
        print(f"  One-at-a-time inflation: {np.mean(rl_cts)/np.mean(normal_cts):.1f}x")
        print(f"  Batch inflation:         {np.mean(batch_cts)/np.mean(normal_cts):.1f}x")
    
    sla = np.percentile(original_cts, 95)
    print(f"\n  SLA threshold (p95): {sla:.0f}s")
    print(f"  CR normal:         {np.mean(normal_cts < sla):.2%}")
    print(f"  CR RL one-at-a-time: {np.mean(rl_cts < sla):.2%}")
    print(f"  CR RL batch:       {np.mean(batch_cts < sla):.2%}")


if __name__ == "__main__":
    main()