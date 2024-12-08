from collections import OrderedDict
import gymnasium as gym
import numpy as np
from balatro_connection import State, Actions
import time
from gamestates import cache_state
from gymnasium import spaces as sp
from balatro_connection import BalatroConnection
from gym_envs.balatro_constants import Suit, rank_lookup
from random import randint, choices, sample, shuffle, random, choice
from ray.rllib.utils.spaces.repeated import Repeated
from gym_envs.joker_effects import calculate_joker_effects, supported_jokers
from gym_envs.pseudo.card import Card
from gym_envs.pseudo.deck import Deck
from gym_envs.pseudo.hand import Hand

from gym_envs.joker_observer import JokerObserver
from gym_envs.pseudo.blind_like_base_env import BlindLikeBaseEnv


class PlayHandTypeEnv(BlindLikeBaseEnv):
    metadata = {"name": "PlayCardEnv-v0"}

    def __init__(self, env_config={}):
        super().__init__(env_config)

        self.action_space = PlayHandTypeEnv.build_action_space(self.max_hand_size)
        self.observation_space = PlayHandTypeEnv.build_observation_space(
            self.max_hand_size
        )

        self.reward_range = (-float("inf"), float("inf"))

    @staticmethod
    def build_observation_space(max_hand_size):
        target_hand_types = sp.Box(low=-10, high=10, shape=(9,), dtype=np.float32)
        space = BlindLikeBaseEnv.build_observation_space(max_hand_size)
        space["target_hand_types"] = target_hand_types
        return sp.Dict(space)

    def get_obs(self):
        obs = super().get_obs()
        obs["target_hand_types"] = self.target_hand_types
        return obs

    def step(self, action):
        action = self.action_vector_to_action(action)

        illegal_reasons = self.check_illegal_actions(action)
        if len(illegal_reasons) > 0:
            print("YOU CAN'T DO THAT")
            return self.get_obs(), -0.1, False, False, {}

        could_have_played_target = (
            self.target_hand_type in self.hand.contained_hand_types()
        )
        played_hand = self.hand.pop_cards(action[1])
        hand_type, scored_cards = played_hand.evaluate()
        self.update_scored_card_stats(scored_cards, hand_type)
        self.draw_cards()
        obs = self.get_obs()
        if action[0] == Actions.PLAY_HAND:
            hand_id = self.hand_to_id[hand_type]
            reward = self.target_hand_types[hand_id]
            self.hand_counts[hand_type] += 1
            if self.target_hand_type == hand_type:
                self.hit_rates[hand_type] += 1
            self.hands_played += 1
            self.confusion_matrix[self.hand_to_id[self.target_hand_type], hand_id] += 1
            done = self.hands_played >= 5
            return obs, reward, done, False, {}
        else:
            self.discards_played += 1
            done = self.discards_played >= 100
            now_can_play_target = (
                self.target_hand_type in self.hand.contained_hand_types()
            )

            if done:
                reward = -2.0
            elif could_have_played_target:
                reward = -self.discard_penalty
            elif now_can_play_target:
                # Reward for discarding when you couldn't have played the target hand
                # And now you can play the target hand
                reward = self.discard_penalty
            else:
                reward = 0.0
            return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        return self.get_obs(), {}
