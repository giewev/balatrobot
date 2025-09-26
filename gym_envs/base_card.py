from copy import deepcopy
from gymnasium import spaces as sp
import numpy as np
from random import randint, choice, random


class BaseCard:
    special_cards = 2  # Null/padding and Stone
    num_jokers = 150
    num_suits = 6  # 4 suits + wild + null
    num_ranks = 14  # Ace to King, plus null
    num_tarot = 22  # Tarot cards
    num_planets = 12  # Planets
    num_spectral = 18  # Spectral cards
    num_boosters = 15  # Booster packs, not cards
    num_segments = 11
    num_expert_tokens = 20  # Expert tokens for different learned contexts
    num_special_tokens = (
        11 + num_expert_tokens
    )  # Special tokens like reroll, end shop, etc.

    NULL_INDEX = 0
    STONE_INDEX = 1
    FIRST_JOKER_INDEX = 2
    FIRST_PLAYING_CARD_INDEX = FIRST_JOKER_INDEX + num_jokers
    FIRST_TAROT_INDEX = FIRST_PLAYING_CARD_INDEX + num_suits * num_ranks
    FIRST_PLANET_INDEX = FIRST_TAROT_INDEX + num_tarot
    FIRST_SPECTRAL_INDEX = FIRST_PLANET_INDEX + num_planets
    FIRST_SPECIAL_TOKEN_INDEX = FIRST_SPECTRAL_INDEX + num_spectral
    FIRST_BOOSTER_INDEX = FIRST_SPECIAL_TOKEN_INDEX + num_special_tokens

    class Segments:
        NULL = 0
        HAND = 1
        DECK = 2
        JOKER = 3
        SHOP_BUYABLE = 4
        BOOSTER_CHOICE = 5
        SPECIAL_TOKEN = 6
        PLANET = 7
        TAROT = 8
        SPECTRAL = 9
        SHOP_JOKER = 10

    class SpecialTokens:
        END_SHOP = 0
        REROLL_SHOP = 1
        END_BOOSTER = 2
        JOKER_CONTEXT = 3
        HAND_CONTEXT = 4
        DECK_CONTEXT = 5
        BLIND_CONTEXT = 6
        SHOP_CONTEXT = 7
        PLAY_CONTEXT = 8
        DISCARD_CONTEXT = 9
        STOP_CONTEXT = 10

        @staticmethod
        def expert_context(n):
            if n < 0 or n >= BaseCard.num_expert_tokens:
                raise ValueError(f"Invalid expert context index: {n}")
            return 11 + n

    class Enhancements:
        NORMAL = 0
        STONE = 1
        WILD = 2
        BONUS = 3
        MULT = 4
        GLASS = 5
        STEEL = 6
        GOLD = 7
        LUCKY = 8
        ALL = {
            "NORMAL": NORMAL,
            "STONE": STONE,
            "WILD": WILD,
            "BONUS": BONUS,
            "MULT": MULT,
            "GLASS": GLASS,
            "STEEL": STEEL,
            "GOLD": GOLD,
            "LUCKY": LUCKY,
        }

    class Editions:
        NO_EDITION = 0
        FOIL = 1
        HOLOGRAPHIC = 2
        POLYCHROME = 3
        NEGATIVE = 4
        ALL = {
            "NO_EDITION": NO_EDITION,
            "FOIL": FOIL,
            "HOLOGRAPHIC": HOLOGRAPHIC,
            "POLYCHROME": POLYCHROME,
            "NEGATIVE": NEGATIVE,
        }

    class Seals:
        NO_SEAL = 0
        GOLD = 1
        RED = 2
        BLUE = 3
        PURPLE = 4
        ETERNAL = 5
        PERISHABLE = 6
        RENTAL = 7
        ALL = {
            "NO_SEAL": NO_SEAL,
            "GOLD": GOLD,
            "RED": RED,
            "BLUE": BLUE,
            "PURPLE": PURPLE,
        }

    total_cards = (
        special_cards
        + num_jokers
        + num_suits * num_ranks
        + num_tarot
        + num_planets
        + num_spectral
        + num_special_tokens
        + num_boosters
    )

    num_enhancements = 9
    num_editions = 5  # Including no edition
    num_seals = 5  # including no seal
    num_scalar_properties = 4  # Value, chips, mult, mult_mult

    def __init__(self, segment=0, scalar_properties=None, u_index=None):
        if u_index is None:
            self.universal_index = 0
        else:
            self.universal_index = u_index
        self.segment = segment
        if scalar_properties is not None:
            self.scalar_properties = np.array(scalar_properties, dtype=np.float32)
        else:
            self.scalar_properties = np.zeros(4, dtype=np.float32)

        self.debuffed = False
        self.value = 1
        self.name = ""

    def is_dupe(self, other):
        if type(self) != type(other):
            return False
        if self.get_universal_index() != other.get_universal_index():
            return False
        if self.get_universal_index() == 0 or other.get_universal_index() == 0:
            return False
        return True

    @staticmethod
    def from_gamestate_card(gamestate_card):
        from gym_envs.pseudo.card import Card
        from gym_envs.pseudo.booster import Booster
        from gym_envs.joker import Joker
        from gym_envs.pseudo.planet import PlanetCard
        from gym_envs.pseudo.tarot import TarotCard

        for sub_class in [
            Card,
            Booster,
            Joker,
            PlanetCard,
            TarotCard,
        ]:
            card = sub_class.from_gamestate_card(gamestate_card)
            if card is not None:
                return card
        print("Warning: Card could not be created from gamestate:", gamestate_card)
        return (
            BaseCard()
        )  # Return a null card so we can treat it as a padding card for now

    def get_universal_index(self):
        return self.universal_index

    def get_enhancement(self):
        return 0

    def get_edition(self):
        return 0

    def get_seal(self):
        return 0

    def get_segment(self):
        return self.segment

    def is_debuffed(self):
        return self.debuffed

    def get_scalar_properties(self):
        return np.array(
            [(self.value - 4) / 10.0, 0, 0, 0],
            dtype=np.float32,
        )

    def get_u_suit_index(self):
        return 0

    def get_u_rank_index(self):
        return 0

    @staticmethod
    def observation_space(max_count):
        return sp.Dict(
            {
                "indices": sp.Box(0, BaseCard.total_cards, (max_count,), np.int32),
                "enhancement": sp.Box(
                    0, BaseCard.num_enhancements, (max_count,), np.int32
                ),
                "edition": sp.Box(0, BaseCard.num_editions, (max_count,), np.int32),
                "seal": sp.Box(0, BaseCard.num_seals, (max_count,), np.int32),
                "debuffed": sp.Box(0, 1, (max_count,), np.int32),
                "scalar_properties": sp.Box(
                    -np.inf, np.inf, (max_count, 4), np.float32
                ),
                "segment": sp.Box(0, BaseCard.num_segments, (max_count,), np.int32),
                "suit": sp.Box(0, BaseCard.num_suits, (max_count,), np.int32),
                "rank": sp.Box(0, BaseCard.num_ranks, (max_count,), np.int32),
            }
        )

    @staticmethod
    def observe_list(cards, max_count, override_segment=None):
        obs = {
            "indices": np.zeros(max_count, dtype=np.int32),
            "enhancement": np.zeros(max_count, dtype=np.int32),
            "edition": np.zeros(max_count, dtype=np.int32),
            "seal": np.zeros(max_count, dtype=np.int32),
            "debuffed": np.zeros(max_count, dtype=np.int32),
            "scalar_properties": np.zeros((max_count, 4), dtype=np.float32),
            "segment": np.zeros(max_count, dtype=np.int32),
            "suit": np.zeros(max_count, dtype=np.int32),
            "rank": np.zeros(max_count, dtype=np.int32),
        }
        for i, card in enumerate(cards):
            obs["indices"][i] = card.get_universal_index()
            obs["enhancement"][i] = card.get_enhancement()
            obs["edition"][i] = card.get_edition()
            obs["seal"][i] = card.get_seal()
            obs["debuffed"][i] = int(card.is_debuffed())
            obs["scalar_properties"][i] = card.get_scalar_properties()
            obs["segment"][i] = (
                override_segment if override_segment is not None else card.get_segment()
            )
            obs["suit"][i] = card.get_u_suit_index()
            obs["rank"][i] = card.get_u_rank_index()

        return obs
