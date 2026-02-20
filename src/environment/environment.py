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
        Optimized to avoid Pandas overhead in the loop.
        """
        sim_state = self.simulator.state()
        internal_time = sim_state["internal_time"]
        
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
            wait_count = sim_state["activities_waiting"].get(str(act), 0)
            act_features.append(float(wait_count))
            
        # 3. Global Counts (1 feature)
        global_features = [float(sim_state["total_waiting"])]
        
        # 4. Fast Time Features (No Pandas!)
        # Assuming seconds for math. 3600s = 1hr, 86400s = 1day
        # If time_unit is different, these constants should be adjusted
        time_of_day = (internal_time % 86400) / 86400.0  # Normalized 0-1
        day_of_week = (internal_time // 86400) % 7
        
        time_features = [internal_time, float(day_of_week), float(time_of_day)]
        
        return np.array(res_features + act_features + global_features + time_features, dtype=np.float32)
    

    @property
    def state_dim(self):
        # 3 features per resource (in_use, capacity, waiting)
        # 1 feature per activity (waiting count)
        # 1 global feature (total waiting)
        # 3 time features (internal time, day of week, time of day)
        return (3 * self.simulator.num_resources) + self.simulator.num_activities + 1 + 3

    def _compute_state(self):
        return self.vectorize_state()

    def get_activity_mask(self, case, k=None, p=0.9):
        """
        Returns a binary mask for feasible next activities based on the simulator's
        RoutingPolicy, filtered by Top-K and Top-P (Nucleus) sampling.
        """
        current_activity = self.simulator.last_activities.get(case.case_id)
        probs_dict = self.simulator.setup.routing_policy.probabilities.get(current_activity, {})
        
        mask = np.zeros(self.simulator.num_activities, dtype=np.float32)
        
        if not probs_dict:
            # If no transitions are defined, only END (None) is allowed
            # None is at the last index of all_activities
            mask[-1] = 1.0
        else:
            # Map probabilities to indices
            for act_name, prob in probs_dict.items():
                try:
                    idx = self.simulator.all_activities.index(act_name)
                    mask[idx] = prob
                except ValueError:
                    continue # Should not happen if all_activities is correct
            
        # 1. Apply Top-K filtering
        if k is not None and 0 < k < len(mask):
            top_k_indices = np.argsort(mask)[-k:]
            new_mask = np.zeros_like(mask)
            new_mask[top_k_indices] = mask[top_k_indices]
            mask = new_mask
            
        # 2. Apply Top-P (Nucleus) filtering
        if p < 1.0:
            sorted_indices = np.argsort(mask)[::-1]
            sorted_probs = mask[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            
            # Find the cutoff: keep indices until cumulative probability exceeds p
            cutoff_idx = np.where(cumulative_probs >= p)[0]
            if len(cutoff_idx) > 0:
                cutoff = cutoff_idx[0]
                top_p_indices = sorted_indices[:cutoff + 1]
                new_mask = np.zeros_like(mask)
                new_mask[top_p_indices] = mask[top_p_indices]
                mask = new_mask

        # Return binary mask (1 if activity is allowed, 0 otherwise)
        binary_mask = (mask > 0).astype(np.float32)
        
        # Ensure at least one activity is allowed (fallback to END)
        if binary_mask.sum() == 0:
            binary_mask[-1] = 1.0
            
        return binary_mask

    def get_resource_mask(self, activity_name, case=None):
        """
        Returns a binary mask of feasible resources for a given activity.
        A resource is feasible if it has the required skills.
        """
        mask = np.zeros(self.simulator.num_resources, dtype=np.float32)
        
        # If activity is END (None), all resources are technically feasible (no-op)
        if activity_name is None:
            return np.ones(self.simulator.num_resources, dtype=np.float32)
            
        for i, res in enumerate(self.simulator.all_resources):
            # Check if resource has the skill for this activity
            if activity_name in res.skills:
                mask[i] = 1.0
        
        # Fallback: if no resource is skilled (should not happen with good data), allow all
        if mask.sum() == 0:
            return np.ones(self.simulator.num_resources, dtype=np.float32)
            
        return mask
