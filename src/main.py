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

def run_rl_experiment():
    """
    Runs a Reinforcement Learning experiment using the OPRA framework.
    This script sets up a simulation environment, initializes an RL agent (PPO),
    and runs a simulation loop where the agent learns to make resource allocation
    decisions. The resulting simulated event log is exported to a CSV file.

    To run this script:
    python src/main.py
    """
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
    max_cases = 200
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
        num_resources=simulator.num_resources,
        lr=1e-3, # Higher LR for quick experimentation
    )

    while not (terminated or truncated):
        # a) Get current case needing decision
        case_needing_decision = simulator.get_case_needing_decision()
        if case_needing_decision is None:
            break
        
        # b) Get activity mask
        activity_mask = env.get_activity_mask(case_needing_decision, k=None, p=0.9)
        
        # c) Define resource mask callback for the agent
        def res_mask_cb(act_idx):
            act_name = simulator.all_activities[act_idx]
            return env.get_resource_mask(act_name, case_needing_decision)

        # d) Agent selects action (non-deterministic for exploration during training)
        act_idx, res_idx = agent.select_action(
            state=obs, 
            activity_mask=activity_mask, 
            resource_mask_callback=res_mask_cb,
            deterministic=False
        )

        # 5. Step the environment
        action = np.array([act_idx, res_idx])
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Store reward and terminal flag in agent buffer
        agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(terminated or truncated)
        
        obs = next_obs
        total_reward += reward
        
        chosen_act_name = simulator.all_activities[act_idx]
        chosen_res_id = simulator.all_resources[res_idx].id
        
        print(f"Step: Case={case_needing_decision.case_id}, Activity={chosen_act_name}, Resource={chosen_res_id}, Reward={reward:.2f}, Total Reward={total_reward:.2f}")

    print(f"Simulation finished. Total Reward: {total_reward}")

    # 6. Update Agent (Training)
    print("Updating policy...")
    agent.update()
    print("Policy updated.")

    # 7. Export results
    event_log = simulator.event_log
    export_event_log_to_csv(event_log, "data/simulated_logs/PurchasingExample/PurchasingExample_RL.csv")
    print(f"Event log exported. Total events: {len(event_log)}")

if __name__ == "__main__":
    run_rl_experiment()

