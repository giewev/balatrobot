import gymnasium as gym
import numpy as np
from balatro_connection import State, Actions
import time
from gamestates import cache_state
from gymnasium import spaces as sp
import enum
from balatro_connection import BalatroConnection


class Suit(enum.Enum):
    Clubs = 0
    Diamonds = 1
    Hearts = 2
    Spades = 3


rank_lookup = {
    "Ace": 14,
    "King": 13,
    "Queen": 12,
    "Jack": 11,
    "Ten": 10,
    "Nine": 9,
    "Eight": 8,
    "Seven": 7,
    "Six": 6,
    "Five": 5,
    "Four": 4,
    "Three": 3,
    "Two": 2,
    "10": 10,
    "9": 9,
    "8": 8,
    "7": 7,
    "6": 6,
    "5": 5,
    "4": 4,
    "3": 3,
    "2": 2,
}


class BalatroBlindEnv(gym.Env):
    metadata = {"name": "BalatroBlindEnv-v0"}

    def __init__(self, env_config):
        self.balatro_connection = None
        self.port = env_config.worker_index + 12348
        self.hand_size = 8
        self.deck = "Blue Deck"
        self.stake = 1
        self.challenge = None
        self.seed = None
        self.last_chip_count = 0

        # Action space: First 8 binary entries for card selection, last entry for play/discard
        # self.action_space = sp.MultiBinary(self.hand_size + 1)

        # Discrete version of action space for rllib
        self.action_space = sp.MultiDiscrete([2] * self.hand_size + [2])

        # Action space: 5 cards to select, and a binary flag for play/discard
        # 0 : hand_size-1 for card indices
        # Special value of hand_size to indicate no card selected
        # First card cannot be the special value, to ensure at least one card is always played
        # Duplicate values are ignored
        # self.action_space = sp.MultiDiscrete(
        #     [self.hand_size] + [self.hand_size + 1] * 4 + [2]
        # )

        # hand_suits = sp.MultiDiscrete([4] * self.hand_size)

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

        # Using a dictionary for clearer structure
        # self.observation_space = sp.Dict(
        #     {
        #         "hand_suits": hand_suits,
        #         "hand_ranks": hand_ranks,
        #         "discards_left": discards_left,
        #         "hands_left": hands_left,
        #         "chips": chips,
        #     }
        # )

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
            # Testing the results of not truncating the episode when an illegal action is taken
            return (
                self.gamestate_to_observation(G),
                -0.3,
                False,  # Terminated
                False,  # Truncated
                {},
            )

        prev_G = G
        self.balatro_connection.connect()
        self.balatro_connection.send_action(action)
        G = self.get_gamestate()
        obs = self.gamestate_to_observation(G)

        game_over = self.check_game_over(G)
        if game_over:
            return obs, 0.0, True, False, {}

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

    def get_gamestate(self):
        G = self.balatro_connection.poll_state()
        # Wait for the game to be in a state where we can select cards
        while (
            not G.get("waitingForAction", False)
            or G["waitingFor"] != "select_cards_from_hand"
        ):
            if G.get("waitingForAction", False):
                auto_action = self.hardcoded_action(G)
                self.balatro_connection.send_action(auto_action)

            G = self.balatro_connection.poll_state()

        return G

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
        if self.balatro_connection is None:
            self.balatro_connection = BalatroConnection(bot_port=self.port)
            if not self.balatro_connection.poll_state():
                self.balatro_connection.start_balatro_instance()
                time.sleep(10)

        self.last_chip_count = 0
        self.start_new_game()
        obs = self.gamestate_to_observation(self.get_gamestate())
        return obs, {}

    def close(self):
        self.balatro_connection.stop_balatro_instance()

    def start_new_game(self):
        G = self.get_gamestate()
        current_round = G["current_round"]
        if current_round["hands_played"] == 0 and current_round["discards_used"] == 0:
            return
        self.balatro_connection.send_cmd("MENU")

    def seed(self, seed=None):
        pass

    def hardcoded_action(self, game_state):
        match game_state["waitingFor"]:
            case "start_run":
                return [
                    Actions.START_RUN,
                    self.stake,
                    self.deck,
                    self.seed,
                    self.challenge,
                ]
            case "skip_or_select_blind":
                return [Actions.SELECT_BLIND]
            case "select_cards_from_hand":
                return None
            case "select_shop_action":
                return [Actions.END_SHOP]
            case "select_booster_action":
                return [Actions.SKIP_BOOSTER_PACK]
            case "sell_jokers":
                return [Actions.SELL_JOKER, []]
            case "rearrange_jokers":
                return [Actions.REARRANGE_JOKERS, []]
            case "use_or_sell_consumables":
                return [Actions.USE_CONSUMABLE, []]
            case "rearrange_consumables":
                return [Actions.REARRANGE_CONSUMABLES, []]
            case "rearrange_hand":
                return [Actions.REARRANGE_HAND, []]

        return None
