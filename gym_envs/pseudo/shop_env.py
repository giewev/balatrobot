import gymnasium as gym
from gymnasium import spaces as sp
from gym_envs.real_balatro.base_env import BalatroBaseEnv
from balatro_connection import State, Actions
import numpy as np
from gym_envs.joker_observer import JokerObserver
from random import randint
from ray.rllib.utils.spaces.repeated import Repeated


class PseudoShopEnv(gym.Env):
    metadata = {"name": "PseudoShopEnv-v0"}

    def __init__(self, env_config):

        self.dollars = 0
        self.reroll_cost = 5
        self.jokers = []
        self.owned_jokers = []
        self.max_jokers = 5

        self.min_dollars = -25.0
        self.max_dollars = 100.0
        self.joker_observer = JokerObserver()

        self.action_space = PseudoShopEnv.build_action_space()
        self.observation_space = PseudoShopEnv.build_observation_space()

    def get_obs(self):
        return {
            "dollars": np.array(self.dollars, dtype=np.float32),
            "shop_cards": [self.joker_observer.observe(j) for j in self.jokers],
            "shop_card_count": np.array(len(self.jokers), dtype=np.float32),
            "owned_joker_count": np.array(len(self.owned_jokers), dtype=np.float32),
            "max_jokers": np.array(self.max_jokers, dtype=np.float32),
            "owned_jokers": [self.joker_observer.observe(j) for j in self.owned_jokers],
            "action_mask": self.get_action_mask(),
        }

    @staticmethod
    def build_observation_space():
        return sp.Dict(
            {
                "dollars": sp.Box(low=-25.0, high=100.0, shape=(), dtype=np.float32),
                "shop_cards": Repeated(JokerObserver.build_observation_space(), 4),
                "shop_card_count": sp.Box(low=0, high=4, shape=(), dtype=np.float32),
                "owned_joker_count": sp.Box(low=0, high=10, shape=(), dtype=np.float32),
                "max_jokers": sp.Box(low=0, high=10, shape=(), dtype=np.float32),
                "owned_jokers": Repeated(JokerObserver.build_observation_space(), 10),
                "action_mask": sp.Box(low=0, high=1, shape=(9,), dtype=np.float32),
            }
        )

    @staticmethod
    def build_action_space():
        # 0) End shop
        # 1) Purchase left card
        # 2) Purchase right card
        # 3) Re-roll shop
        # 4-8) Sell joker 1-5
        return sp.MultiDiscrete([9])
        # return sp.MultiDiscrete([3])

    def get_action_mask(self):
        mask = np.ones(9, dtype=np.float32)
        for i in range(1, 3):
            if not self.check_card_purchaseable([Actions.BUY_CARD, [i]]):
                mask[i] = 0.0
        for i in range(1, 6):
            if len(self.owned_jokers) < i:
                mask[i + 3] = 0.0
        return mask

    def check_card_purchaseable(self, action):
        i = action[1][0] - 1
        if i >= len(self.jokers):
            return False
        if len(self.owned_jokers) >= self.max_jokers:
            return False
        joker = self.jokers[i]
        return self.joker_observer.is_joker(joker) and joker["value"] <= self.dollars

    def action_vector_to_action(self, action):
        if action == 0:
            return [Actions.END_SHOP]
        elif action in [1, 2]:
            return [Actions.BUY_CARD, [action[0]]]
        elif action == 3:
            return [Actions.REROLL_SHOP]
        else:
            return [Actions.SELL_JOKER, [action[0] - 3]]

    def roll_jokers(self):
        self.jokers = [self.joker_observer.generate_random_joker() for _ in range(2)]

    def roll_shop(self):
        self.roll_jokers()

    def new_shop(self):
        self.roll_shop()
        self.reroll_cost = 5

    def reset(self, seed=None, options=None):
        self.new_shop()
        self.dollars = 4
        self.owned_jokers = []
        self.jokers_purchased = 0
        self.jokers_sold = 0
        self.reroll_count = 0
        return self.get_obs(), {}

    def step(self, action):
        action = self.action_vector_to_action(action)
        action[0] = Actions.END_SHOP
        if action[0] == Actions.BUY_CARD:
            if not self.check_card_purchaseable(action):
                # End the shop if the action is invalid for any reason
                reward = -0.2
                action = [Actions.END_SHOP]
            else:
                reward = 0.2
        elif action[0] == Actions.END_SHOP:
            reward = 0
        elif action[0] == Actions.REROLL_SHOP:
            if self.dollars < self.reroll_cost:
                reward = -0.2
                action = [Actions.END_SHOP]
            elif self.dollars >= self.reroll_cost + 5:
                reward = 0.1
            else:
                reward = -0.05
        elif action[0] == Actions.SELL_JOKER:
            if len(self.owned_jokers) < action[1][0]:
                reward = -0.2
                action = [Actions.END_SHOP]
            else:
                reward = 0.0

        old_obs = self.get_obs()
        self.take_action(action)
        info = {"shop_ended": False}
        if action[0] == Actions.END_SHOP:
            obs = old_obs
            info["shop_ended"] = True
        else:
            obs = self.get_obs()
        return obs, reward, False, False, info

    def take_action(self, action):
        if action[0] == Actions.BUY_CARD:
            purchased_joker = self.jokers.pop(action[1][0] - 1)
            self.dollars -= purchased_joker["value"]
            self.owned_jokers.append(purchased_joker)
            self.jokers_purchased += 1
        elif action[0] == Actions.REROLL_SHOP:
            self.dollars -= self.reroll_cost
            self.reroll_cost += 2
            self.roll_jokers()
            self.reroll_count += 1
        elif action[0] == Actions.SELL_JOKER:
            sold_joker = self.owned_jokers.pop(action[1][0] - 1)
            self.dollars += sold_joker["value"]
            self.jokers_sold += 1
        elif action[0] == Actions.END_SHOP:
            self.roll_shop()
