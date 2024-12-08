import gymnasium as gym
from gymnasium import spaces as sp
from gym_envs.balatro.base_env import BalatroBaseEnv
from balatro_connection import State, Actions
import numpy as np
from gym_envs.joker_observer import JokerObserver
from random import randint
from ray.rllib.utils.spaces.repeated import Repeated


class BalatroShopEnv(BalatroBaseEnv):
    metadata = {"name": "BalatroShopEnv-v0"}

    def __init__(self, env_config):
        super().__init__(env_config)

        self.min_dollars = -25.0
        self.max_dollars = 100.0
        self.joker_observer = JokerObserver()

        # end shop, Purchase left, purchase right
        self.action_space = BalatroShopEnv.build_action_space()

        self.observation_space = BalatroShopEnv.build_observation_space()

    @staticmethod
    def build_observation_space():
        return sp.Dict(
            {
                "dollars": sp.Box(low=-25.0, high=100.0, shape=(), dtype=np.float32),
                # "card_0": JokerObserver.build_observation_space(),
                # "card_1": JokerObserver.build_observation_space(),
                "shop_cards": Repeated(JokerObserver.build_observation_space(), 4),
                "owned_joker_count": sp.Box(low=0, high=10, shape=(), dtype=np.float32),
                "max_jokers": sp.Box(low=0, high=10, shape=(), dtype=np.float32),
                "owned_jokers": Repeated(JokerObserver.build_observation_space(), 10),
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

    def gamestate_to_observation(self, G):
        cards = G["shop"]["cards"]
        joker_0 = cards[0] if len(cards) > 0 else None
        joker_1 = cards[1] if len(cards) > 1 else None
        return {
            "dollars": np.array(
                np.clip(G["dollars"], self.min_dollars, self.max_dollars),
                dtype=np.float32,
            ),
            # "card_0": self.joker_observer.observe(joker_0),
            # "card_1": self.joker_observer.observe(joker_1),
            "shop_cards": [self.joker_observer.observe(joker) for joker in cards],
            # "owned_joker_count": len(G["jokers"]),
            "owned_joker_count": np.array(len(G["jokers"]), dtype=np.float32),
            # "max_jokers": G["max_jokers"],
            "max_jokers": np.array(G["max_jokers"], dtype=np.float32),
            "owned_jokers": [
                self.joker_observer.observe(joker) for joker in G["jokers"]
            ],
        }

    def reset(self, seed=None, options=None):
        # return self.observation_space.sample(), {}
        G = self.get_gamestate()
        return self.gamestate_to_observation(G), {}

    def check_card_purchaseable(self, action, G):
        if action[1][0] > len(G["shop"]["cards"]):
            return False
        if self.joker_observer.is_joker(G["shop"]["cards"][action[1][0] - 1]):
            return True
        return False

    def check_too_expensive(self, action, G):
        if action[0] == Actions.BUY_CARD:
            if action[1][0] == 1:
                return G["dollars"] < G["shop"]["cards"][0]["cost"]
            elif action[1][0] == 2:
                return G["dollars"] < G["shop"]["cards"][1]["cost"]
        return False

    def check_jokers_full(self, G):
        return len(G["jokers"]) >= G["max_jokers"]

    def action_vector_to_action(self, action):
        if action == 0:
            return [Actions.END_SHOP]
        elif action in [1, 2]:
            return [Actions.BUY_CARD, [action[0]]]
        elif action == 3:
            return [Actions.REROLL_SHOP]
        else:
            return [Actions.SELL_JOKER, [action[0] - 3]]

    def step(self, action):
        G = self.get_gamestate()
        action = self.action_vector_to_action(action)
        if action[0] == Actions.BUY_CARD:
            if (
                not self.check_card_purchaseable(action, G)
                or self.check_too_expensive(action, G)
                or self.check_jokers_full(G)
            ):
                # End the shop if the action is invalid for any reason
                reward = -0.2
                action = [Actions.END_SHOP]
            else:
                reward = 0.2
        elif action[0] == Actions.END_SHOP:
            reward = 0
        elif action[0] == Actions.REROLL_SHOP:
            if G["dollars"] < G["reroll_cost"]:
                reward = -0.2
                action = [Actions.END_SHOP]
            elif G["dollars"] >= 5:
                reward = 0.1
            else:
                reward = -0.05
        elif action[0] == Actions.SELL_JOKER:
            if len(G["jokers"]) < action[1][0]:
                reward = -0.2
                action = [Actions.END_SHOP]
            else:
                reward = 0.0

        self.balatro_connection.send_action(action)
        prev_G = G
        G = self.get_gamestate(pending_action=action[0].value)
        info = {"shop_ended": False}
        if action[0] == Actions.END_SHOP:
            obs = self.gamestate_to_observation(prev_G)
            info["shop_ended"] = True
        else:
            obs = self.gamestate_to_observation(G)
        return obs, reward, False, False, info
