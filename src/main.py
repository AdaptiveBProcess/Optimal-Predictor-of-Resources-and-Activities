import pandas as pd
import numpy as np
import random

from environment.simulator.adapters.event_log_to_csv import export_event_log_to_csv
from initializer.implementations.DESInitializer import DESInitializer
from environment.simulator.core.setup import SimulationSetup
from environment.environment import BusinessProcessEnvironment
from environment.simulator.core.log_names import LogColumnNames
from environment.simulator.core.engine import SimulatorEngine

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# 1. Load data and initialize setup
log = pd.read_csv("data/logs/PurchasingExample/PurchasingExample.csv")

log_names = LogColumnNames(
    case_id="caseid",
    activity="Activity_1",
    resource="Resource_1",
    start_timestamp="start_timestamp",
    end_timestamp="end_timestamp",
)

initializer = DESInitializer()
start_timestamp = log[log_names.start_timestamp].min()
time_unit = "seconds"

setup: SimulationSetup = initializer.build(log, log_names, start_timestamp, time_unit)
simulator = SimulatorEngine(setup)

# 2. Calculate SLA threshold (e.g., p75 cycle time)
cycle_times = []
for case_id, group in log.groupby(log_names.case_id):
    st = pd.to_datetime(group[log_names.start_timestamp]).min()
    et = pd.to_datetime(group[log_names.end_timestamp]).max()
    cycle_times.append((et - st).total_seconds())
sla_threshold = np.percentile(cycle_times, 75)
print(f"SLA Threshold (p75): {sla_threshold:.2f} seconds")

# 3. Initialize Environment
max_cases = 10
env = BusinessProcessEnvironment(simulator, sla_threshold=sla_threshold, max_cases=max_cases)

# 4. Agent Simulation Loop
print(f"Starting simulation for {max_cases} cases...")

obs, info = env.reset()
terminated = False
truncated = False
total_reward = 0

while not (terminated or truncated):
    # --- Agent Decision Logic (Conceptual) ---
    
    # a) Get current case and its last activity
    case_needing_decision = simulator.get_case_needing_decision()
    current_activity = simulator.last_activities.get(case_needing_decision.case_id) if case_needing_decision else None
    
    # b) Get possible next activities from Routing Policy
    probs = setup.routing_policy.probabilities.get(current_activity, {})
    if not probs:
        feasible_activities = [None]
    else:
        feasible_activities = list(probs.keys())

    # c) Agent selects activity (simulated random choice from feasible)
    chosen_activity = random.choice(feasible_activities)
    
    if chosen_activity is None:
        act_idx = simulator.all_activities.index(None)
        res_idx = 0
    else:
        # Find index in all_activities
        try:
            act_idx = simulator.all_activities.index(chosen_activity)
        except ValueError:
            act_idx = 0 
            
        # d) Get resource feasibility mask
        feasible_resources = [
            i for i, r in enumerate(simulator.all_resources)
            if chosen_activity in r.skills
        ]
        
        if not feasible_resources:
            res_idx = random.randint(0, simulator.num_resources - 1)
        else:
            res_idx = random.choice(feasible_resources)

    # 5. Step the environment
    action = np.array([act_idx, res_idx])
    obs, reward, terminated, truncated, info = env.step(action)
    
    total_reward += reward
    if reward > 0:
        print(f"Case completed! Reward: {reward}")

print(f"Simulation finished. Total Reward: {total_reward}")

# 6. Export results
event_log = simulator.event_log
export_event_log_to_csv(event_log, "data/simulated_logs/PurchasingExample/PurchasingExample_RL.csv")
print(f"Event log exported. Total events: {len(event_log)}")
