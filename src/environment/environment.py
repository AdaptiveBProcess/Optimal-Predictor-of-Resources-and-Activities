import gymnasium as gym
import numpy as np
import pandas as pd
import simpy

from environment.simulator.core.engine import SimulatorEngine

class BusinessProcessEnvironment(gym.Env):

    def __init__(self, simulator: "SimulatorEngine", sla_threshold, max_cases):
        super().__init__()

        self.simulator = simulator
        self.sla_threshold = sla_threshold
        self.max_cases = max_cases

        self.action_space = gym.spaces.MultiDiscrete(
            [simulator.num_activities, simulator.num_resources]
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.simulator.reset(max_cases=self.max_cases)
        self.completed_cases = 0

        state, _ = self._advance_to_next_decision()
        return state, {}

    def step(self, action):
        act_idx, res_idx = action
        
        # Map indices to actual activity and resource
        # Note: In RL mode, act_idx might represent 'END' if defined in all_activities
        activity_type = self.simulator.all_activities[act_idx]
        resource = self.simulator.all_resources[res_idx]

        # Apply decision to simulator (it will resume the process_case)
        self.simulator.apply_decision(activity_type, resource)

        state, completed = self._advance_to_next_decision()

        reward = 0
        for case in completed:
            if case.cycle_time <= self.sla_threshold:
                reward += 1
            self.completed_cases += 1

        terminated = self.completed_cases >= self.max_cases
        truncated = False

        return state, reward, terminated, truncated, {}


    def _advance_to_next_decision(self):
        completed_cases = self.simulator.run_until_decision()

        state = self._compute_state()

        return state, completed_cases

    def vectorize_state(self):
        """
        Converts the simulator's dictionary state into a numerical vector.
        """
        sim_state = self.simulator.state()
        
        # 1. Resource Occupancy (3 features per resource)
        res_features = []
        for res in self.simulator.all_resources:
            occ = sim_state["resource_occupancy"].get(res.id, {"in_use": 0, "capacity": 0, "waiting": 0})
            res_features.extend([
                float(occ["in_use"]), 
                float(occ["capacity"]), 
                float(occ["waiting"])
            ])
            
        # 2. Activity Waiting Counts (1 feature per activity)
        act_features = []
        for act in self.simulator.all_activities:
            wait_count = sim_state["activities_with_waiting_cases"].get(str(act), 0)
            act_features.append(float(wait_count))
            
        # 3. Global Counts (2 features)
        global_features = [
            float(sim_state["total_cases_processing"]),
            float(sim_state["total_cases_waiting"])
        ]
        
        # 4. Time Features (3 features)
        dt = pd.to_datetime(sim_state["time_info"]["current_absolute_timestamp"])
        day_of_week = float(dt.dayofweek)
        time_of_day = float(dt.hour * 3600 + dt.minute * 60 + dt.second)
        internal_time = float(sim_state["time_info"]["current_time_internal_units"])
        
        time_features = [internal_time, day_of_week, time_of_day]
        
        return np.array(res_features + act_features + global_features + time_features, dtype=np.float32)
    

    @property
    def state_dim(self):
        # 3 features per resource (in_use, capacity, waiting)
        # 1 feature per activity (waiting count)
        # 2 global features (total processing, total waiting)
        # 3 time features (internal time, day of week, time of day)
        return (3 * self.simulator.num_resources) + self.simulator.num_activities + 2 + 3

    def _compute_state(self):
        return self.vectorize_state()
