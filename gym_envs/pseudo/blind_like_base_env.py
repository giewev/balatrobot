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
import random as rand
from ray.rllib.utils.spaces.repeated import Repeated
from gym_envs.joker_effects import calculate_joker_effects, supported_jokers
from gym_envs.pseudo.card import Card
from gym_envs.pseudo.deck import Deck
from gym_envs.pseudo.hand import Hand
import torch

from gym_envs.joker_observer import JokerObserver
from itertools import combinations as combos


class BlindLikeBaseEnv(gym.Env):
    metadata = {"name": "BlindLikeBaseEnv-v0"}

    card_combo_to_index = {}
    index_to_card_combo = {}
    for num_cards in range(0, 6):
        card_combo_to_index[num_cards] = {}
        index_to_card_combo[num_cards] = {}
        c = list(combos(range(8), num_cards))
        for i, c in enumerate(c):
            card_combo_to_index[num_cards][c] = i
            index_to_card_combo[num_cards][i] = c

    def __init__(self, env_config={}):
        self.max_hand_size = env_config.get("max_hand_size", 8)
        self.infinite_deck = env_config.get("infinite_deck", False)
        initial_bias = env_config.get("bias", 0.0)

        self.correct_reward = env_config.get("correct_reward", 1.0)
        self.incorrect_penalty = env_config.get("incorrect_penalty", 1.0)
        self.discard_penalty = env_config.get("discard_penalty", 0.05)
        self.rarity_bonus = env_config.get("rarity_bonus", 0.0)

        self.hands = [
            "High card",  # 0
            "Pair",  # 1
            "Two pair",  # 2
            "Three of a kind",  # 3
            "Straight",  # 4
            "Flush",  # 5
            "Full house",  # 6
            "Four of a kind",  # 7
            "Straight flush",  # 8
        ]

        self.biases = {hand: initial_bias for hand in self.hands}
        self.hit_rates = {hand: 0 for hand in self.hands}
        self.target_counts = {hand: 0 for hand in self.hands}
        self.hand_counts = {k: 0 for k in self.hands}
        self.card_slot_counts = {i: 0 for i in range(1, 9)}
        self.count_counts = {i: 0 for i in range(1, 6)}
        self.hands_played = 0
        self.discards_played = 0
        self.confusion_matrix = np.zeros((9, 9), dtype=np.float32)
        self.rarities = {hand: 0 for hand in self.hands}
        self.scored_ranks = {
            hand: np.zeros(13, dtype=np.float32) for hand in self.hands
        }
        self.scored_suits = {hand: np.zeros(4, dtype=np.float32) for hand in self.hands}

        self.hand_to_id = {x: i for i, x in enumerate(self.hands)}
        self.id_to_hand = {i: x for i, x in enumerate(self.hands)}

        self.action_space = BlindLikeBaseEnv.build_action_space(self.max_hand_size)
        self.observation_space = BlindLikeBaseEnv.build_observation_space(
            self.max_hand_size
        )

        self.reward_range = (-float("inf"), float("inf"))

    @staticmethod
    def build_observation_space(max_hand_size):
        available_hand_types = sp.Box(low=0, high=1, shape=(8,), dtype=np.float32)

        hand_indices = sp.Box(low=0, high=52, shape=(max_hand_size,), dtype=np.int8)
        deck_indices = sp.Box(low=0, high=52, shape=(52 - 8,), dtype=np.int8)

        card_relation_counts = sp.Box(
            low=1, high=max_hand_size, shape=(max_hand_size,), dtype=np.float32
        )

        deck_relation_counts = sp.Box(low=0, high=52, shape=(52 - 8,), dtype=np.float32)

        space = sp.Dict(
            {
                "available_hand_types": available_hand_types,
                "hands_played": sp.Box(low=0, high=10, shape=(), dtype=np.float32),
                "discards_played": sp.Box(low=0, high=100, shape=(), dtype=np.float32),
                "rank_counts": card_relation_counts,
                "suit_counts": card_relation_counts,
                "run_counts": card_relation_counts,
                "suited_run_counts": card_relation_counts,
                "hand_indices": hand_indices,
                "deck_rank_counts": deck_relation_counts,
                "deck_suit_counts": deck_relation_counts,
                "deck_run_counts": deck_relation_counts,
                "deck_suited_run_counts": deck_relation_counts,
                "deck_indices": deck_indices,
            }
        )

        return space

    def get_obs(self):
        rank_counts, suit_counts = self.hand.card_dupe_counts()
        available_hands = np.zeros(8, dtype=np.float32)
        for hand in self.hand.contained_hand_types():
            if hand != "High card":
                available_hands[self.hand_to_id[hand] - 1] = 1.0
        # print(available_hands)
        deck_cards = self.deck.remaining_cards
        deck_cards = deck_cards + [Card(None, None)] * (52 - len(deck_cards) - 8)
        deck_hand = Hand(deck_cards)
        deck_rank_counts, deck_suit_counts = deck_hand.card_dupe_counts()

        return {
            "available_hand_types": available_hands,
            "hands_played": np.array(self.hands_played, dtype=np.float32),
            "discards_played": np.array(self.discards_played, dtype=np.float32),
            "rank_counts": np.array(rank_counts, dtype=np.float32),
            "suit_counts": np.array(suit_counts, dtype=np.float32),
            "run_counts": np.array(
                self.hand.card_run_counts(suited=False), dtype=np.float32
            ),
            "suited_run_counts": np.array(
                self.hand.card_run_counts(suited=True), dtype=np.float32
            ),
            "hand_indices": np.array(
                [card.index() for card in self.hand], dtype=np.int32
            ),
            "deck_rank_counts": np.array(deck_rank_counts, dtype=np.float32),
            "deck_suit_counts": np.array(deck_suit_counts, dtype=np.float32),
            "deck_run_counts": np.array(
                deck_hand.card_run_counts(suited=False), dtype=np.float32
            ),
            "deck_suited_run_counts": np.array(
                deck_hand.card_run_counts(suited=True), dtype=np.float32
            ),
            "deck_indices": np.array(
                [card.index() for card in deck_hand], dtype=np.int32
            ),
        }

    @staticmethod
    def build_action_space(hand_size):
        # return sp.MultiDiscrete([2] + [2] * hand_size)
        return sp.MultiDiscrete([2, 5, 70])

    def check_illegal_actions(self, action):
        fail_reasons = []
        if len(action[1]) < 1 or len(action[1]) > 5:
            fail_reasons.append(f"Invalid number of cards selected: {len(action[1])}")

        # if self.discards_left <= 0 and action[0] == Actions.DISCARD_HAND:
        #     fail_reasons.append("No discards left")

        return fail_reasons

    def update_scored_card_stats(self, scored_cards, played_cards, hand_type):
        if scored_cards is None:
            return
        suit_map = {
            "Clubs": 0,
            "Diamonds": 1,
            "Hearts": 2,
            "Spades": 3,
        }
        for card in scored_cards:
            self.scored_ranks[hand_type][card.value - 2] += 1
            self.scored_suits[hand_type][suit_map[card.suit]] += 1
        self.hand_counts[hand_type] += 1
        if hand_type == self.target_hand_type:
            self.hit_rates[hand_type] += 1

        self.count_counts[len(played_cards)] += 1

    def chips_from_value(self, value):
        if value == 14:
            return 11
        return min(value, 10)

    def action_vector_to_action(self, action_vector):
        play_or_discard = action_vector[0]
        card_count = action_vector[1] + 1
        index = action_vector[2]
        if index >= len(self.index_to_card_combo[card_count]):
            print("INVALID INDEX (may be dummy sample during initialization)")
            index = 0

        # print(cards)
        action = [
            Actions.PLAY_HAND if play_or_discard else Actions.DISCARD_HAND,
            # self.binary_hand_to_card_indices(cards),
            list(self.index_to_card_combo[card_count][index]),
        ]
        # print(action)

        return action

    def binary_hand_to_card_indices(self, binary_hand):
        return (np.where(binary_hand)[0] + 1).tolist()

    def draw_cards(self):
        starting_size = len(self.hand)
        while len(self.hand) < 8:
            if (
                False
                and starting_size > 0
                and random() < self.biases[self.target_hand_type]
            ):
                if self.target_hand_type == "Flush":
                    card = Card.random_flush(self.hand)
                elif self.target_hand_type in [
                    "Pair",
                    "Three of a kind",
                    "Four of a kind",
                ]:
                    card = Card.random_dupe(self.hand)
                elif self.target_hand_type in "Two pair":
                    card = Card.random_two_pair(self.hand)
                elif self.target_hand_type == "Full house":
                    card = Card.random_full_house(self.hand)
                elif self.target_hand_type == "Straight":
                    card = Card.random_straight(self.hand)
                elif self.target_hand_type == "Straight flush":
                    card = Card.random_straight_flush(self.hand)
                elif self.target_hand_type == "High card":
                    card = Card.random()
            elif not self.infinite_deck:
                # card = self.deck.draw()
                # avg_bias = sum(self.biases.values()) / len(self.biases)
                bias = self.biases[self.target_hand_type]
                # bias_calculator = self.hand.general_bias_calculator()
                biasers = [lambda x: 0]
                if self.target_hand_type in [
                    "Pair",
                    "Three of a kind",
                    "Four of a kind",
                    "Two pair",
                    "Full house",
                ]:
                    biasers.append(self.hand.rank_biaser())
                if self.target_hand_type in ["Flush"]:
                    biasers.append(self.hand.suit_biaser())
                if self.target_hand_type in ["Straight"]:
                    biasers.append(self.hand.straight_biaser())
                if self.target_hand_type in ["Straight flush"]:
                    biasers.append(self.hand.straight_flush_biaser())
                biaser = lambda x: sum([b(x) for b in biasers])
                card = self.deck.draw_biased(biaser, bias)
            else:
                card = Card.random()
            self.hand.add_card(card)

        # Sort the hand to make it easier for the model to learn
        # self.hand = sorted(self.hand, key=lambda x: (x.value, x.suit))
        self.hand.sort()

        # Alternatively, shuffle the hand to force the model to generalize
        # self.hand.shuffle()

    def get_bias(self):
        return self.biases

    def set_bias(self, bias):
        self.biases = bias

    def get_and_reset_stats(self):
        stats = {
            "confusion": self.confusion_matrix.copy(),
            "target_counts": self.target_counts.copy(),
            "hit_rates": self.hit_rates.copy(),
            "scored_ranks": self.scored_ranks.copy(),
            "scored_suits": self.scored_suits.copy(),
            "hand_counts": self.hand_counts.copy(),
        }
        # confusion = self.confusion_matrix.copy()
        self.confusion_matrix = np.zeros((9, 9), dtype=np.float32)
        # target_counts = self.target_counts.copy()
        self.target_counts = {hand: 0 for hand in self.hands}
        # hit_rates = self.hit_rates.copy()
        self.hit_rates = {hand: 0 for hand in self.hands}
        self.scored_ranks = {
            hand: np.zeros(13, dtype=np.float32) for hand in self.hands
        }
        self.scored_suits = {hand: np.zeros(4, dtype=np.float32) for hand in self.hands}
        return stats

    def update_target_hand_type(self):
        self.target_hand_types = np.zeros(9, dtype=np.float32)
        self.target_hand_types -= self.incorrect_penalty
        self.target_hand_type = choice(self.hands)
        # self.target_hand_type = "Straight flush"
        self.target_counts[self.target_hand_type] += 1
        chosen_index = self.hand_to_id[self.target_hand_type]
        self.target_hand_types[chosen_index] = self.correct_reward

    def set_rarities(self, rarities):
        self.rarities = rarities

    def reset(self, seed=None, options=None):
        self.hand = Hand()
        self.deck = Deck(infinite=self.infinite_deck)
        self.update_target_hand_type()
        self.draw_cards()

        self.hand_counts = {k: 0 for k in self.hands}
        self.card_slot_counts = {i: 0 for i in range(1, 9)}
        self.count_counts = {i: 0 for i in range(1, 6)}
        self.hands_played = 0
        self.discards_played = 0

        return BlindLikeBaseEnv.get_obs(self), {}
