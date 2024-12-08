from ray.rllib.env import MultiAgentEnv
from gymnasium import spaces as sp
from gym_envs.pseudo.blind_env import PseudoBlindEnv
from gym_envs.pseudo.shop_env import PseudoShopEnv
from balatro_connection import BalatroConnection
import time


class CurriculumEnv(MultiAgentEnv):
    def __init__(self, env_config={}):
        super().__init__()
        self.initial_discards = env_config.get("initial_discards", 6)
        self.initial_bias = env_config.get("initial_bias", 0.95)

        self.hands = [
            "Flush",  # 0
            "Four of a kind",  # 1
            "Full house",  # 2
            "High card",  # 3
            "Pair",  # 4
            "Straight",  # 5
            "Straight flush",  # 6
            "Three of a kind",  # 7
            "Two pair",  # 8
        ]

        self.curriculum_steps = [
            "play_target_hand",
            "play_for_chips",
            "chips_varied_deck",
            "chips_varied_joker",
            "reach_max_round",
            "round_purchase_jokers",
        ]

        self.lesson = {
            "step": "play_target_hand",
            "diff": {
                hand: {"bias": self.initial_bias, "discards": self.initial_discards}
                for hand in self.hands
            },
        }

        self.one_more = False
        self.postponed_blind_reward = 0
        self.shop_seen = False
        self._agent_ids = {"blind_player"}  # , "hand_selector" "shopper"}

        self.blind_env = PseudoBlindEnv(env_config)
        # self.shop_env = PseudoShopEnv(env_config)

        self.action_space = sp.Dict(
            {
                "blind_player": self.blind_env.action_space,
                # "shop": self.shop_env.action_space,
            }
        )

        self.observation_space = sp.Dict(
            {
                "blind_player": self.blind_env.observation_space,
                # "shop": self.shop_env.observation_space,
            }
        )

        self._spaces_in_preferred_format = True
        self.set_lesson(self.lesson)

    def get_lesson(self):
        return self.lesson

    def set_lesson(self, lesson):
        self.lesson = lesson
        step, diff = lesson["step"], lesson["diff"]
        if step == "play_target_hand":
            biases = {hand: diff[hand]["bias"] for hand in self.hands}
            discards = {hand: diff[hand]["discards"] for hand in self.hands}
            self.blind_env.biases = biases
            self.blind_env.max_discards = discards
            self.blind_env.chips_reward = 0
            self.blind_env.correct_reward = 2.0
            self.blind_env.incorrect_penalty = 2.0
            self.blind_env.discard_penalty = 0.01
        elif step == "play_for_chips":
            self.blind_env.biases = {hand: 0.0 for hand in self.hands}
            self.blind_env.max_discards = diff["discards"]
            self.blind_env.max_plays = diff["plays"]
            self.blind_env.chips_reward = 0.02
            self.blind_env.correct_reward = 0.0
            self.blind_env.incorrect_penalty = 0.0
            self.blind_env.discard_penalty = 0.01

    def get_and_reset_stats(self):
        return self.blind_env.get_and_reset_stats()

    def reset(self, seed=None, options=None):
        self.postponed_blind_reward = 0
        self.shop_seen = False
        self.set_lesson(self.lesson)
        blind_obs, blind_infos = self.blind_env.reset()
        # shop_obs, shop_infos = self.shop_env.reset()
        return {
            "blind_player": blind_obs,
            # "shop": self.shop_env.observation_space.sample(),
        }, {
            "blind_player": blind_infos,
            # "shop": {},
        }

    def step(self, action):
        results = {}
        # game_over = False
        # print(action)
        if "blind_player" in action:
            blind_action = action["blind_player"]
            obs, reward, term, trunc, info = self.blind_env.step(blind_action)
            results["blind_player"] = (obs, reward, term, trunc, info)

        # Invert the dictionary to get tuple of dictionaries from agent name to tuple of obs, reward, term, trunc, info
        inverted_results = [{}, {}, {}, {}, {}]
        for agent_name, values in results.items():
            for i in range(5):
                inverted_results[i][agent_name] = values[i]

        inverted_results[2]["__all__"] = inverted_results[2]["blind_player"]
        inverted_results[3]["__all__"] = False
        if len(inverted_results[0]) == 0:
            print("No obs returned")
            # exit()
        # print(inverted_results)
        return tuple(inverted_results)

    def close(self):
        pass
