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

fake_agent = {
    'routing_policy': setup.routing_policy,
    'resources': setup.resource_policy
}

activities_set = simulator.all_activities
resources_set = simulator.all_resources

while not (terminated or truncated):
    # --- Agent Decision Logic (Conceptual) ---
    
    # a) Get current case and its last activity
    case_needing_decision = simulator.get_case_needing_decision()
    if case_needing_decision is None:
        break
    current_activity = simulator.last_activities.get(case_needing_decision.case_id) if case_needing_decision else None
    
    # b) Get possible next activities from Routing Policy
    # c) Agent selects activity (simulated random choice from feasible)
    chosen_activity = fake_agent['routing_policy'].get_next_activity(case_needing_decision, current_activity)
    
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
        chosen_resource = fake_agent['resources'].select_resource(chosen_activity, case_needing_decision)
        res_idx = simulator.all_resources.index(chosen_resource) if chosen_resource in simulator.all_resources else 0

    # 5. Step the environment
    action = np.array([act_idx, res_idx])
    obs, reward, terminated, truncated, info = env.step(action)
    
    total_reward += reward
    
    print(f"Step: Case={case_needing_decision.case_id}, Activity={simulator.all_activities[act_idx]}, Resource={simulator.all_resources[res_idx]}, Reward={reward:.2f}, Total Reward={total_reward:.2f}")
    if reward > 0:
        print(f"Case completed! Reward: {reward}")

print(f"Simulation finished. Total Reward: {total_reward}")

# 6. Export results
event_log = simulator.event_log
export_event_log_to_csv(event_log, "data/simulated_logs/PurchasingExample/PurchasingExample_RL.csv")
print(f"Event log exported. Total events: {len(event_log)}")
