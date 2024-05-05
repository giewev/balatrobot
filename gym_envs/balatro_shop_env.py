import gymnasium as gym
from gym_envs.balatro_base_env import BalatroBaseEnv
from balatro_connection import State, Actions
import numpy as np


class BalatroShopEnv(BalatroBaseEnv):
    metadata = {"name": "BalatroShopEnv-v0"}

    def __init__(self, env_config):
        super().__init__(env_config)

        # Placeholders for HRL PoC
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        self.send_action([Actions.END_SHOP])
        self.get_gamestate()
        return self.observation_space.sample(), 0, False, False, {}
