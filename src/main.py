import pandas as pd
import numpy as np
import random

from environment.simulator.adapters.event_log_to_csv import export_event_log_to_csv
from initializer.implementations.DESInitializer import DESInitializer
from environment.simulator.core.setup import SimulationSetup
from environment.environment import BusinessProcessEnvironment
from environment.simulator.core.log_names import LogColumnNames
from environment.simulator.core.engine import SimulatorEngine
from agent.agent import PPOAgent
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

# Initialize actual agent
agent = PPOAgent(
    state_dim=env.observation_space.shape[0], 
    num_activities=simulator.num_activities, 
    num_resources=simulator.num_resources
)

activities_set = simulator.all_activities
resources_set = simulator.all_resources

while not (terminated or truncated):
    # a) Get current case needing decision
    case_needing_decision = simulator.get_case_needing_decision()
    if case_needing_decision is None:
        # If no case needs decision but simulation isn't done, 
        # it might be waiting for arrivals or something.
        # But in this loop, we usually expect one if not terminated.
        break
    
    # b) Get activity mask
    activity_mask = env.get_activity_mask(case_needing_decision, k=None, p=0.9)
    
    # c) Define resource mask callback for the agent
    def res_mask_cb(act_idx):
        act_name = simulator.all_activities[act_idx]
        return env.get_resource_mask(act_name, case_needing_decision)

    # d) Agent selects action (greedy/deterministic)
    act_idx, res_idx = agent.select_action(
        state=obs, 
        activity_mask=activity_mask, 
        resource_mask_callback=res_mask_cb,
        deterministic=True
    )

    # 5. Step the environment
    action = np.array([act_idx, res_idx])
    obs, reward, terminated, truncated, info = env.step(action)
    
    total_reward += reward
    
    chosen_act_name = simulator.all_activities[act_idx]
    chosen_res_id = simulator.all_resources[res_idx].id
    
    print(f"Step: Case={case_needing_decision.case_id}, Activity={chosen_act_name}, Resource={chosen_res_id}, Reward={reward:.2f}, Total Reward={total_reward:.2f}")
    if reward > 0:
        print(f"Case completed! Reward: {reward}")

print(f"Simulation finished. Total Reward: {total_reward}")

# 6. Export results
event_log = simulator.event_log
export_event_log_to_csv(event_log, "data/simulated_logs/PurchasingExample/PurchasingExample_RL.csv")
print(f"Event log exported. Total events: {len(event_log)}")
