import gymnasium as gym
import numpy as np

class ProcessEnv(gym.Env):

    def __init__(self, simulator, sla_threshold, max_cases):
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
            shape=(simulator.state_dim,),
            dtype=np.float32
        )
 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.simulator.reset()
        self.completed_cases = 0

        state = self._advance_to_next_decision()
        return state, {}

    def step(self, action):
        activity, resource = action

        self.simulator.apply_decision(activity, resource)

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

    def _compute_state(self):
        return self.simulator.get_state_vector()
