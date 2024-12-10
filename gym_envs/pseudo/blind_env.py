from collections import OrderedDict
import gymnasium as gym
import numpy as np
from balatro_connection import State, Actions
import time
from gamestates import cache_state
from gymnasium import spaces as sp
from balatro_connection import BalatroConnection
from gym_envs.balatro_constants import Suit, rank_lookup
from random import randint, choices
from ray.rllib.utils.spaces.repeated import Repeated
from gym_envs.joker_effects import calculate_joker_effects, supported_jokers
from gym_envs.pseudo.blind_like_base_env import BlindLikeBaseEnv
from gym_envs.joker_observer import JokerObserver
from gym_envs.pseudo.hand import Hand


class PseudoBlindEnv(BlindLikeBaseEnv):
    metadata = {"name": "BalatroPseudoBlindEnv-v0"}

    def __init__(self, env_config={"max_hand_size": 15}):
        super().__init__(env_config)

        self.hand_scores = {
            "High card": (5, 1),  # 5
            "Pair": (10, 2),  # 20
            "Two pair": (20, 2),  # 40
            "Three of a kind": (30, 3),  # 90
            "Straight": (30, 4),  # 120
            "Flush": (35, 4),  # 140
            "Full house": (40, 4),  # 160
            "Four of a kind": (60, 7),  # 420
            "Straight flush": (100, 8),  # 800
        }

        self.max_discards = {x: 4 for x in self.hand_scores.keys()}
        self.max_plays = {x: 1 for x in self.hand_scores.keys()}
        self.chips_reward_weight = env_config.get("chips_reward_weight", 1.0)
        self.hand_type_reward_weight = env_config.get("hand_type_reward_weight", 0.0)

        self.joker_observer = JokerObserver()

        self.action_space = PseudoBlindEnv.build_action_space(self.max_hand_size)
        self.observation_space = PseudoBlindEnv.build_observation_space(
            self.max_hand_size
        )

    @staticmethod
    def build_observation_space(max_hand_size):
        discards_left = sp.Box(low=0, high=10, shape=(1,), dtype=np.float32)
        hands_left = sp.Box(low=0, high=10, shape=(1,), dtype=np.float32)
        chips = sp.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
        log_chip_goal = sp.Box(low=-3, high=20, shape=(1,), dtype=np.float32)
        target_hand_types = sp.Box(low=-10, high=10, shape=(9,), dtype=np.float32)

        space = BlindLikeBaseEnv.build_observation_space(max_hand_size)
        space["chips"] = chips
        space["log_chip_goal"] = log_chip_goal
        space["discards_left"] = discards_left
        space["hands_left"] = hands_left
        space["target_hand_types"] = target_hand_types
        space["last_hand_played"] = sp.Box(low=0, high=1, shape=(9,), dtype=np.float32)

        return sp.Dict(space)

    def get_obs(self, reset_hand=False, new_hand=None):
        obs = super().get_obs()
        obs["chips"] = np.array(
            [np.clip(self.chips / self.chip_goal, 0, 1)], dtype=np.float32
        )
        obs["log_chip_goal"] = np.array([np.log(self.chip_goal) - 6], dtype=np.float32)
        obs["discards_left"] = np.array([self.discards_left], dtype=np.float32)
        obs["hands_left"] = np.array([self.hands_left], dtype=np.float32)
        # annealed_targets = self.target_hand_types * self.biases[self.target_hand_type]
        # obs["target_hand_types"] = annealed_targets
        obs["target_hand_types"] = self.target_hand_types
        obs["last_hand_played"] = self.last_hand_played
        if reset_hand:
            self.last_hand_played = np.zeros(9, dtype=np.float32)

        return obs

    def check_illegal_actions(self, action):
        fail_reasons = super().check_illegal_actions(action)
        if self.discards_left <= 0 and action[0] == Actions.DISCARD_HAND:
            fail_reasons.append("No discards left")

        return fail_reasons

    def step(self, action):
        action = self.action_vector_to_action(action)
        last_hand_played = np.zeros(9, dtype=np.float32)

        illegal_reasons = self.check_illegal_actions(action)
        if len(illegal_reasons) > 0:
            print("YOU CAN'T DO THAT")
            print(illegal_reasons)
            return (self.get_obs(reset_hand=True), -0.1, False, False, {})

        played_hand = self.hand.pop_cards(action[1])
        if action[0] == Actions.DISCARD_HAND:
            self.discards_played += 1
            self.discards_left -= 1
            self.draw_cards()
            return (
                self.get_obs(reset_hand=True),
                -self.discard_penalty,
                False,
                False,
                # {"hand_played": hand_played_vector},
                {},
            )
        elif action[0] == Actions.PLAY_HAND:
            self.hands_played += 1
            self.hands_left -= 1
            game_over = self.hands_left == 0
            hand_type, scored_cards = played_hand.evaluate()
            self.update_scored_card_stats(scored_cards, played_hand, hand_type)
            self.draw_cards()

            chips, mult = self.hand_scores[hand_type]
            chips += sum([self.chips_from_value(card.value) for card in scored_cards])
            hand_score = chips * mult
            old_chips = self.chips
            self.chips += hand_score

            # reward = 5 * ((hand_score / self.chip_goal) ** 1)
            # reward = 0
            # reward = (hand_score / 50) * (1 - self.biases[self.target_hand_type])
            # if hand_type == self.target_hand_type:
            #     reward += self.biases[self.target_hand_type]
            # capped_score = min(hand_score, 300 - old_chips)
            # reward = capped_score / 300
            reward = hand_score * self.chips_reward_weight
            reward += (
                self.target_hand_types[self.hand_to_id[hand_type]]
                * self.hand_type_reward_weight
            )

            avg_rarity = sum(self.rarities.values()) / len(self.rarities)
            rarity = self.rarities[hand_type]
            if rarity > avg_rarity:
                reward += (rarity - avg_rarity) * self.rarity_bonus

            obs = self.get_obs(reset_hand=True)

            self.last_hand_played[self.hand_to_id[hand_type]] = 1

            return (
                obs,
                reward,
                game_over,
                False,
                {},
            )
        else:
            raise ValueError(f"Invalid action {action[0]}")

    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        self.jokers = []
        self.round = 1
        self.chips = 0
        self.chip_goal = 600
        if type(self.max_plays) == dict:
            self.hands_left = self.max_plays[self.target_hand_type]
            self.discards_left = self.max_discards[self.target_hand_type]
        else:
            self.hands_left = self.max_plays
            self.discards_left = self.max_discards

        self.last_hand_played = np.zeros(9, dtype=np.float32)
        return self.get_obs(), {}
