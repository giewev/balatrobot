import gymnasium as gym
import numpy as np
from balatro_connection import State, Actions
import time
from gamestates import cache_state
from gymnasium import spaces as sp
from balatro_connection import BalatroConnection
from gym_envs.real_balatro.base_env import BalatroBaseEnv
from gym_envs.balatro_constants import Suit, rank_lookup
from random import randint
from ray.rllib.utils.spaces.repeated import Repeated
from gym_envs.joker_observer import JokerObserver


class BalatroBlindEnv(BalatroBaseEnv):
    metadata = {"name": "BalatroBlindEnv-v1"}

    def __init__(self, env_config):
        super().__init__(env_config)
        self.last_chip_count = 0
        if self.agency_states is None:
            self.agency_states = ["select_cards_from_hand"]

        self.joker_observer = JokerObserver()

        # Action space: First 8 binary entries for card selection, last entry for play/discard
        # Discrete version of action space for rllib
        self.action_space = BalatroBlindEnv.build_action_space(self.hand_size)

        # Observation space: continuous representation limits the dimensionality of the observation space
        # Going back to discrete representation may be better later on as it better represents the nature of the game

        # Flattened observation space since I'm getting issues with Dict spaces on rllib
        self.observation_space = BalatroBlindEnv.build_observation_space()

        self.reward_range = (-float("inf"), float("inf"))

    @staticmethod
    def build_observation_space():
        # hand_suits = sp.Box(low=0, high=3, shape=(hand_size,), dtype=np.float32)
        # hand_ranks = sp.Box(low=2, high=14, shape=(hand_size,), dtype=np.float32)
        discards_left = sp.Box(low=0, high=10, shape=(1,), dtype=np.float32)
        hands_left = sp.Box(low=1, high=10, shape=(1,), dtype=np.float32)
        chips = sp.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)

        ranks = sp.Discrete(13, start=2)
        ranks_continuous = sp.Box(low=2, high=14, shape=(), dtype=np.float32)
        suits = sp.Discrete(4)
        card = sp.Tuple([ranks, ranks_continuous, suits])
        hand = Repeated(card, 15)
        hand_size = sp.Box(low=0, high=15, shape=(), dtype=np.float32)
        owned_jokers = Repeated(JokerObserver.build_observation_space(), 10)
        num_jokers = sp.Box(low=0, high=10, shape=(), dtype=np.float32)

        return sp.Dict(
            {
                "chips": chips,
                "discards_left": discards_left,
                "hands_left": hands_left,
                "hand": hand,
                "hand_size": hand_size,
                "owned_jokers": owned_jokers,
                "owned_jokers_count": num_jokers,
            }
        )

    @staticmethod
    def build_action_space(hand_size):
        return sp.MultiDiscrete([2] * hand_size + [2])

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
        # action = play_flushes(G)
        action = self.clip_hand_indices(action, G)
        if G["current_round"]["discards_left"] == 0:
            action[0] = Actions.PLAY_HAND

        illegal_reasons = self.check_illegal_actions(action, G)
        if len(illegal_reasons) > 0:
            print(f"Illegal action: {illegal_reasons}")
            obs = self.gamestate_to_observation(G)
            # print(obs)
            return obs, -0.3, False, False, {}

        prev_G = G
        self.balatro_connection.send_action(action)
        G = self.get_gamestate(pending_action=action[0].value)
        obs = self.gamestate_to_observation(G)

        game_over = self.check_game_over(G)
        if game_over:
            return obs, 0.0, True, False, {"game_over": 1.0}

        round_won = self.check_blind_win(G)
        required_chips = prev_G["current_round"]["chips_required"]
        if round_won:
            reward = 2 * (required_chips - prev_G["chips"]) / required_chips
            reward += prev_G["current_round"]["hands_left"] * 0.2
            obs = self.gamestate_to_observation(prev_G)
            return obs, reward, True, False, {"game_over": 0.0}

        if action[0] == Actions.DISCARD_HAND:
            reward = 0.00
        else:
            new_chips = G["chips"]
            reward = 2 * (new_chips - self.last_chip_count) / required_chips
            self.last_chip_count = new_chips
        return obs, reward, False, False, {"game_over": 0.0}

    def clip_hand_indices(self, action, G):
        h = self.real_hand_size(G)
        action[1] = [max(1, min(h, x)) for x in action[1]]
        action[1] = list(set(action[1]))
        return action

    def real_hand_size(self, G):
        return len(G["hand"])

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
        # print(len(hand))
        hand = [self.card_to_vectors(card) for card in hand]
        # You seriously have more than 15 cards in your hand?
        hand = hand[:15]

        discards_left = np.array(
            [G["current_round"]["discards_left"]], dtype=np.float32
        )
        hands_left = np.array([G["current_round"]["hands_left"]], dtype=np.float32)
        chips = np.array([G["chips"]], dtype=np.float32)
        chips /= G["current_round"]["chips_required"]

        owned_jokers = [self.joker_observer.observe(joker) for joker in G["jokers"]]
        num_jokers = np.array(len(G["jokers"]), dtype=np.float32)

        return {
            "chips": chips,
            "discards_left": discards_left,
            "hands_left": hands_left,
            "hand": hand,
            "hand_size": len(hand),
            "owned_jokers": owned_jokers,
            "owned_jokers_count": num_jokers,
        }

    def card_to_vectors(self, card):
        suit = card["suit"]
        rank = card["value"]
        suit = Suit[suit].value
        rank = rank_lookup[rank]
        # return {"ranks": rank, "suits": suit}
        return (rank, rank, suit)

    def reset(self, seed=None, options=None):
        # return self.observation_space.sample(), {}
        self.last_chip_count = 0
        G = self.get_gamestate()
        # print(G["current_round"]["blind_name"])
        obs = self.gamestate_to_observation(G)
        return obs, {}
