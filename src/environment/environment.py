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
            low=0.0,
            high=1.0,
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
        Builds a normalized state vector in [0, 1] with three blocks.

        Block A — Global Process Snapshot  (3*R + A features)
        --------------------------------------------------------
        Per resource (ordered by simulator.all_resources):
          · utilization       : count / capacity
          · assignment_enc    : normalized index of activity being executed (0 = idle)
          · queue_pressure    : queue length / 10, clamped to [0, 1]
        Per activity (ordered by simulator.all_activities):
          · pending_count     : cases queued for this activity / 10, clamped to [0, 1]

        Block B — Case-Specific Features  (A + 3 features)
        --------------------------------------------------------
          · branching_probs   : P(next_activity | current_position), one float per activity
          · last_activity_enc : (index_of_last_activity + 1) / num_activities (0 = new case)
          · trace_length_norm : len(history) / 20, clamped to [0, 1]
          · sla_urgency       : elapsed_time / sla_threshold, clamped to [0, 1]

        Block C — Temporal Features  (2 features)
        --------------------------------------------------------
          · hour_of_day  : (now % 86400) / 86400
          · day_of_week  : ((now // 86400) % 7) / 6
        """
        sim_state = self.simulator.state()
        now = sim_state["internal_time"]
        num_act = self.simulator.num_activities

        # ── Block A: Global Process Snapshot ────────────────────────────────────
        res_features = []
        for res in self.simulator.all_resources:
            occ = sim_state["resource_occupancy"].get(
                res.id, {"in_use": 0, "capacity": 1, "waiting": 0}
            )
            capacity = max(occ["capacity"], 1)

            utilization = occ["in_use"] / capacity

            current_act = sim_state["resource_current_activity"].get(res.id)
            if current_act is not None and current_act in self.simulator.all_activities:
                assignment_enc = self.simulator.all_activities.index(current_act) / num_act
            else:
                assignment_enc = 0.0

            queue_pressure = min(occ["waiting"] / 10.0, 1.0)

            res_features.extend([utilization, assignment_enc, queue_pressure])

        # Per-activity: cases in waiting_requests (resource-queue) for each activity
        act_pending = {}
        for _, (_, act) in self.simulator.waiting_requests.items():
            act_pending[act] = act_pending.get(act, 0) + 1

        act_features = [
            min(act_pending.get(act, 0) / 10.0, 1.0)
            for act in self.simulator.all_activities
        ]

        # ── Block B: Case-Specific Features ─────────────────────────────────────
        case = self.simulator.get_case_needing_decision()

        if case is None:
            # Simulation ended or between decisions — safe zero vector
            case_features = [0.0] * (num_act + 3)
        else:
            history = case.activity_history
            current_activity = history[-1] if history else None

            # Last activity encoded as (idx+1)/num_act so that 0 unambiguously means "new case"
            if history and current_activity in self.simulator.all_activities:
                last_act_enc = (self.simulator.all_activities.index(current_activity) + 1) / num_act
            else:
                last_act_enc = 0.0

            trace_length_norm = min(len(history) / 20.0, 1.0)
            sla_urgency = min((now - case.start_time) / max(self.sla_threshold, 1.0), 1.0)

            # Branching probabilities for current routing position
            probs_dict = self.simulator.setup.routing_policy.get_activity_probabilities(case)

            branching_probs = [
                float(probs_dict.get(act, 0.0))
                for act in self.simulator.all_activities
            ]

            case_features = branching_probs + [last_act_enc, trace_length_norm, sla_urgency]

        # ── Block C: Temporal Features ───────────────────────────────────────────
        time_features = [
            (now % 86400) / 86400.0,           # hour_of_day
            ((now // 86400) % 7) / 6.0,        # day_of_week
        ]

        return np.clip(
            np.array(res_features + act_features + case_features + time_features, dtype=np.float32),
            0.0, 1.0,
        )

    @property
    def state_dim(self):
        # Block A: 3 per resource + 1 per activity
        # Block B: num_activities (branching probs) + 3 (last_act, trace_len, sla_urgency)
        # Block C: 2 (hour_of_day, day_of_week)
        return 3 * self.simulator.num_resources + 2 * self.simulator.num_activities + 5

    def _compute_state(self):
        return self.vectorize_state()

    def get_activity_mask(self, case, k=None, p=0.9):
        """
        Returns a binary mask for feasible next activities based on the simulator's
        RoutingPolicy, filtered by Top-K and Top-P (Nucleus) sampling.
        """
        probs_dict = self.simulator.setup.routing_policy.get_activity_probabilities(case)
        
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
