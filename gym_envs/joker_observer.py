import gymnasium.spaces as sp
from gym_envs.balatro_constants import ALL_JOKER_DATA
from ray.rllib.utils.spaces.repeated import Repeated
import numpy as np
import pandas as pd
from gym_envs.joker_effects import supported_jokers
from random import randint


class JokerObserver:
    def __init__(self):
        self.joker_data = ALL_JOKER_DATA
        self.names_to_order = {
            joker["name"]: joker["order"] for joker in self.joker_data
        }

        self.simulated_jokers = supported_jokers()

        # 150 possible jokers, 0 is null
        name_space = sp.Discrete(len(self.joker_data) + 1)

        # Highest base cost is 20 but could be modified outside of this by other cards
        cost_space = sp.Box(low=0, high=20, shape=())

        # Common, Uncommon, Rare, Legendary
        rarity_space = sp.Discrete(4)

        # For jokers that give a bonus to a specific suit
        suit_affiliation_space = sp.MultiBinary(4)

        # For jokers that give a bonus to a specific rank
        # Or e.g. even cards, odd cards, face cards, etc.
        rank_affiliation_space = sp.MultiBinary(14)

        # High card, pair, two pair, three of a kind, straight, flush, full house, four of a kind, straight flush
        hand_affiliation_space = sp.MultiBinary(9)

        # For miscellaneous ability associations
        # joker_ability_flags_space = sp.MultiBinary(len(self.ability_flags))

        self.observation_space = JokerObserver.build_observation_space()

    @staticmethod
    def build_observation_space():
        return sp.Dict(
            {
                "name": sp.Discrete(151),
                "cost": sp.Box(low=0, high=20, shape=(), dtype=np.float32),
                "rarity": sp.Discrete(4),
                "flags": sp.MultiBinary(16),
                # "suit_affiliation": sp.MultiBinary(4),
                # "rank_affiliation": sp.MultiBinary(14),
                # "hand_affiliation": sp.MultiBinary(9),
                # "ability_flags": sp.MultiBinary(16),
            }
        )

    def null_joker_obs(self):
        return {
            "name": 0,
            "cost": np.array(0, dtype=np.float32),
            "rarity": 0,
            "flags": np.zeros(16, dtype=np.float32),
            # "suit_affiliation": [0, 0, 0, 0],
            # "rank_affiliation": [0] * 14,
            # "hand_affiliation": [0] * 9,
            # "ability_flags": [0] * len(self.ability_flags),
        }

    def generate_random_joker(self):
        joker = self.simulated_jokers[randint(0, len(self.simulated_jokers) - 1)]
        return {"label": joker, "value": randint(1, 6)}

    def is_joker(self, joker):
        return joker is not None and joker.get("label", None) in self.names_to_order

    def observe(self, joker):
        # print(joker)
        if self.is_joker(joker):
            order = self.names_to_order[joker["label"]]
            rarity = self.joker_data[order - 1]["rarity"]
            flags = self.joker_data[order - 1]["flags"]
        else:
            print(f'failed to find joker {joker["label"]}')
            return self.null_joker_obs()

        if "cost" in joker:
            cost = joker["cost"]
        else:
            return self.null_joker_obs()

        return {
            "name": order,
            "cost": np.array(cost, dtype=np.float32),
            "rarity": rarity,
            "flags": flags,
        }
