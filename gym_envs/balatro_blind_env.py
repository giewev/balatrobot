import gymnasium as gym
import numpy as np
from balatro_connection import State, Actions
import time
from gamestates import cache_state
from gymnasium import spaces as sp
import enum


class Suit(enum.Enum):
    Clubs = 0
    Diamonds = 1
    Hearts = 2
    Spades = 3


class Rank(enum.Enum):
    Ace = 0
    Two = 1
    Three = 2
    Four = 3
    Five = 4
    Six = 5
    Seven = 6
    Eight = 7
    Nine = 8
    Ten = 9
    Jack = 10
    Queen = 11
    King = 12


class BalatroBlindEnv(gym.Env):
    metadata = {"name": "BalatroBlindEnv-v0"}

    def __init__(self, balatro_connection):
        self.balatro_connection = balatro_connection
        self.cards_per_hand = 8
        self.deck = "Blue Deck"
        self.stake = 1
        self.challenge = None
        self.seed = None

        # Action space: First 8 binary entries for card selection, last entry for play/discard
        self.action_space = sp.MultiBinary(self.cards_per_hand + 1)

        # card = sp.MultiDiscrete([13, 4])
        # hand = sp.MultiDiscrete([13, 4] * self.cards_per_hand)
        # discards_left = sp.Discrete(5)
        # deck = sp.MultiBinary(52)

        # Using a dictionary for clearer structure
        # self.observation_space = sp.Dict({"hand": hand, "discards_left": discards_left})

        # Simplified observation space for PoC
        self.observation_space = sp.MultiDiscrete(([13, 4] * self.cards_per_hand) + [5])

        self.reward_range = (-float("inf"), float("inf"))

    def step(self, action):
        action = self.action_vector_to_action(action)
        # print(action)

        if len(action[1]) < 1 or len(action[1]) > 5:
            print("bad card number, failing")
            reward = -1  # Penalize invalid selections
            info = {"message": f"Invalid number of cards selected: {len(action[1])}"}
            return None, reward, False, True, info  # Truncated episode, not terminated

        G = self.get_gamestate()
        if (
            G["current_round"]["discards_left"] == 0
            and action[0] == Actions.DISCARD_HAND
        ):
            print("No discards left, failing")
            reward = -1  # Penalize invalid discards
            info = {"message": "No discards left"}
            return None, reward, False, True, info  # Truncated episode, not terminated

        self.balatro_connection.connect()
        self.balatro_connection.send_action(action)
        G = self.get_gamestate()
        terminated = False
        truncated = False
        reward = 0.1
        info = {}

        current_round = G["current_round"]
        if current_round["hands_played"] == 0 and current_round["discards_used"] == 0:
            if G["round"] == 1:
                terminated = True
                reward = -0.5
                info = {"message": "Game over"}
                print("game over")
                self.start_new_game()
                G = self.get_gamestate()
            else:
                reward = 2
                # Commented for the moment because these stats would get pulled from the following round
                # reward += current_round["hands_left"] / 5
                # reward += current_round["discards_left"] / 10
                terminated = True
                info = {"message": "Round won"}
                print("round won")
                self.start_new_game()
                G = self.get_gamestate()

        return self.gamestate_to_observation(G), reward, terminated, truncated, info

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
        i = 0
        while (
            not G.get("waitingForAction", False)
            or G["waitingFor"] != "select_cards_from_hand"
        ):
            i += 1
            # if i == 100:
            #     print("Gamestate not ready, waiting for action")
            #     print(G)
            if G.get("waitingForAction", False):
                auto_action = self.hardcoded_action(G)
                self.balatro_connection.send_action(auto_action)

            G = self.balatro_connection.poll_state()
            # time.sleep(0.01)

        return G

    def gamestate_to_observation(self, G):
        hand = G["hand"]
        hand = [self.card_to_vectors(card) for card in hand]
        # flatten hand and append discards_left
        # hand = [item for sublist in hand for item in sublist]

        return np.concatenate(hand + [[G["current_round"]["discards_left"]]])
        # return hand + [G["current_round"]["discards_left"]]
        # return {
        #     "hand": np.concatenate(hand),
        #     "discards_left": G["current_round"]["discards_left"],
        # }

    def card_to_vectors(self, card):
        suit = card["suit"]
        rank = card["value"]
        suit = Suit[suit].value

        if rank[0].isdigit():
            rank = int(rank) - 1
        else:
            rank = Rank[rank].value
        return [rank, suit]

    def reset(self, seed=None):
        print("Resetting environment")
        # self.start_new_game()
        return self.gamestate_to_observation(self.get_gamestate()), {}

    def start_new_game(self):
        G = self.get_gamestate()
        current_round = G["current_round"]
        if current_round["hands_played"] == 0 and current_round["discards_used"] == 0:
            print("prompted to start a new game, but already in a new game")
            return
        print("Starting new game")
        # self.balatro_connection.send_action(
        #     [Actions.START_RUN, self.stake, self.deck, self.seed, self.challenge]
        # )
        self.balatro_connection.send_cmd("MENU")

    def seed(self, seed=None):
        pass

    def hardcoded_action(self, game_state):
        # if self.G["state"] == State.GAME_OVER:
        #     self.running = False
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
