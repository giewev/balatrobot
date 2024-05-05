import gymnasium as gym
import numpy as np
from balatro_connection import State, Actions
import time
from gamestates import cache_state
from gymnasium import spaces as sp
from balatro_connection import BalatroConnection
from gym_envs.balatro_base_env import BalatroBaseEnv
from gym_envs.balatro_constants import Suit, rank_lookup


class BalatroBlindEnv(BalatroBaseEnv):
    metadata = {"name": "BalatroBlindEnv-v1"}

    def __init__(self, env_config):
        super().__init__(env_config)
        self.last_chip_count = 0
        if self.agency_states is None:
            self.agency_states = ["select_cards_from_hand"]

        # Action space: First 8 binary entries for card selection, last entry for play/discard
        # Discrete version of action space for rllib
        self.action_space = sp.MultiDiscrete([2] * self.hand_size + [2])

        # Observation space: continuous representation limits the dimensionality of the observation space
        # Going back to discrete representation may be better later on as it better represents the nature of the game
        hand_suits = sp.Box(low=0, high=3, shape=(self.hand_size,), dtype=np.float32)
        hand_ranks = sp.Box(low=2, high=14, shape=(self.hand_size,), dtype=np.float32)
        discards_left = sp.Box(low=0, high=3, shape=(1,), dtype=np.float32)
        hands_left = sp.Box(low=1, high=5, shape=(1,), dtype=np.float32)
        chips = sp.Box(low=0, high=600.0, shape=(1,), dtype=np.float32)

        # Flattened observation space since I'm getting issues with Dict spaces on rllib
        self.observation_space = sp.Box(
            low=np.concatenate(
                [
                    chips.low,
                    discards_left.low,
                    hand_ranks.low,
                    hand_suits.low,
                    hands_left.low,
                ]
            ),
            high=np.concatenate(
                [
                    chips.high,
                    discards_left.high,
                    hand_ranks.high,
                    hand_suits.high,
                    hands_left.high,
                ]
            ),
            dtype=np.float32,
        )

        self.reward_range = (-float("inf"), float("inf"))

    def check_illegal_actions(self, action, G):
        fail_reasons = []
        if len(action[1]) < 1 or len(action[1]) > 5:
            fail_reasons.append(f"Invalid number of cards selected: {len(action[1])}")

        if (
            G["current_round"]["discards_left"] == 0
            and action[0] == Actions.DISCARD_HAND
        ):
            fail_reasons.append("No discards left")

        return fail_reasons

    def check_game_over(self, G):
        current_round = G["current_round"]
        if current_round["hands_played"] == 0 and current_round["discards_used"] == 0:
            if G["round"] == 1:
                return "Game over, lost to blind"

    def check_blind_win(self, G):
        # We paused for agency in another state, implying we've exited the blind
        if G["waitingFor"] != "select_cards_from_hand":
            return "Beat previous blind, round won"

        # If there are no other agency states, we may go right into the next round
        current_round = G["current_round"]
        if current_round["hands_played"] == 0 and current_round["discards_used"] == 0:
            if G["round"] != 1:
                return "Beat previous blind, round won"

    def step(self, action):
        action = self.action_vector_to_action(action)

        G = self.get_gamestate()
        if G["current_round"]["discards_left"] == 0:
            action[0] = Actions.PLAY_HAND

        illegal_reasons = self.check_illegal_actions(action, G)
        if len(illegal_reasons) > 0:
            print(f"Illegal action: {illegal_reasons}")
            obs = self.gamestate_to_observation(G)
            print(obs)
            return obs, -0.3, False, False, {}

        prev_G = G
        self.balatro_connection.send_action(action)
        G = self.get_gamestate()
        obs = self.gamestate_to_observation(G)

        game_over = self.check_game_over(G)
        if game_over:
            return obs, 0.0, True, False, {"game_over": 1.0}

        round_won = self.check_blind_win(G)
        if round_won:
            reward = prev_G["current_round"]["hands_left"] * 0.5 + 1.0
            return obs, reward, False, False, {}

        if action[0] == Actions.DISCARD_HAND:
            reward = 0.00
        else:
            new_chips = G["chips"]
            reward = (new_chips - self.last_chip_count) / 100
            self.last_chip_count = new_chips
        return obs, reward, False, False, {}

    def action_vector_to_action(self, action_vector):
        card_selection = action_vector[:-1]
        play_or_discard = action_vector[-1]
        action = [
            Actions.PLAY_HAND if play_or_discard else Actions.DISCARD_HAND,
            self.binary_hand_to_card_indices(card_selection),
        ]

        return action

    def binary_hand_to_card_indices(self, binary_hand):
        return (np.where(binary_hand)[0] + 1).tolist()

    def gamestate_to_observation(self, G):
        hand = G["hand"]
        hand = [self.card_to_vectors(card) for card in hand]
        hand_ranks = np.array([card["ranks"] for card in hand], dtype=np.float32)
        hand_suits = np.array([card["suits"] for card in hand], dtype=np.float32)

        discards_left = np.array(
            [G["current_round"]["discards_left"]], dtype=np.float32
        )
        hands_left = np.array([G["current_round"]["hands_left"]], dtype=np.float32)
        chips = np.array([G["chips"]], dtype=np.float32)

        flattened_obs = np.concatenate(
            [
                chips,
                discards_left,
                hand_ranks,
                hand_suits,
                hands_left,
            ]
        )
        return flattened_obs

    def card_to_vectors(self, card):
        suit = card["suit"]
        rank = card["value"]
        suit = Suit[suit].value
        rank = rank_lookup[rank]
        return {"ranks": rank, "suits": suit}

    def reset(self, seed=None, options=None):
        self.last_chip_count = 0
        obs = self.gamestate_to_observation(self.get_gamestate())
        return obs, {}
