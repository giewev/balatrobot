import random
from types import SimpleNamespace

import pytest

from gym_envs.base_card import BaseCard
from gym_envs.pseudo import Card, Hand


def test_card_equality_and_hashing():
    a = Card("Hearts", 10)
    b = Card("Hearts", 10)
    c = Card("Spades", 10)

    assert a == b
    assert a != c
    assert {a, c} == {b, c}


def test_copy_from():
    source = Card("Clubs", 7)
    source.seal = 2
    source.enhancement = 3
    source.edition = 4

    target = Card("Hearts", 10)
    target.copy_from(source)

    assert target.suit == "Clubs"
    assert target.value == 7
    assert target.seal == 2
    assert target.enhancement == 3
    assert target.edition == 4


def test_suit_index_handles_unknown_suits():
    assert Card("Spades", 5).suit_index() == 3
    assert Card("Stars", 5).suit_index() is None


def test_universal_and_embedding_indices():
    normal = Card("Hearts", 9)
    stone = Card("Hearts", 9)
    stone.enhancement = BaseCard.Enhancements.STONE
    wild = Card("Hearts", 2)
    wild.enhancement = BaseCard.Enhancements.WILD

    assert normal.get_universal_index() == 0
    assert stone.get_universal_index() == BaseCard.STONE_INDEX
    assert normal.get_u_rank_index() == 8
    assert stone.get_u_rank_index() == 0
    assert normal.get_u_suit_index() == 1  # hearts => index 0 + 1
    assert stone.get_u_suit_index() == 0
    assert wild.get_u_suit_index() == 5


def test_chip_value_and_scalar_properties():
    low_card = Card("Clubs", 5)
    ace = Card("Spades", 14)
    ace.hiker_chips = 2
    stone = Card("Diamonds", 3)
    stone.enhancement = BaseCard.Enhancements.STONE

    assert low_card.chip_value() == 5
    assert ace.chip_value() == 13
    assert stone.chip_value() == 50

    props = low_card.get_scalar_properties()
    assert props.shape == (4,)
    # Only the first two entries are populated
    assert abs(props[0] - ((low_card.cost - 4) / 10)) < 1e-6
    assert props[1] == low_card.chip_value()


def test_index_round_trip_and_bounds():
    card = Card("Diamonds", 12)
    idx = card.index()
    assert idx == 1 * 13 + (12 - 2)
    assert Card.index_to_card(idx) == card
    assert Card.index_to_card(-1) is None
    assert Card.index_to_card(52) is None


def test_is_face_card_and_pareidolia():
    jack = Card("Clubs", 11)
    nine = Card("Clubs", 9)
    pareidolia = SimpleNamespace(name="Pareidolia")

    assert jack.is_face_card() is True
    assert nine.is_face_card() is False
    assert nine.is_face_card(jokers=[pareidolia]) is True


def test_from_gamestate_card_and_repr():
    gamestate = {"suit": "Hearts", "value": "Queen"}
    card = Card.from_gamestate_card(gamestate)
    assert str(card) == "12 of Hearts"
    assert repr(card) == "12 of Hearts"

    assert Card.from_gamestate_card({"suit": None, "value": 10}) is None


def test_smeared_suit_classification():
    assert Card("Hearts", 5).smeared_suit() == "Red"
    assert Card("Spades", 5).smeared_suit() == "Black"
    assert Card("Stars", 5).smeared_suit() is None


def test_random_flush_prefers_existing_suit(monkeypatch):
    def fixed_choice(options):
        return options[0]

    monkeypatch.setattr("gym_envs.pseudo.card.choice", fixed_choice)

    hand = Hand([Card("Hearts", 2), Card("Hearts", 7), Card("Spades", 9)])
    new_card = Card.random_flush(hand)
    assert new_card.suit == "Hearts"
    assert new_card.value in Card.RANKS


def test_random_straight_and_straight_flush(monkeypatch):
    def fixed_choice(options):
        return options[0]

    monkeypatch.setattr("gym_envs.pseudo.card.choice", fixed_choice)

    hand = Hand([Card("Diamonds", 10)])
    straight_card = Card.random_straight(hand)
    assert straight_card.value in {9, 11}

    expected_flush_suit = Card.random_flush(hand).suit
    straight_flush_card = Card.random_straight_flush(hand)
    assert straight_flush_card.suit == expected_flush_suit
    assert straight_flush_card.value == straight_card.value


def test_random_dupe_and_two_pair(monkeypatch):
    def fixed_choice(options):
        return options[0]

    monkeypatch.setattr("gym_envs.pseudo.card.choice", fixed_choice)

    hand = Hand([Card("Hearts", 4), Card("Clubs", 7), Card("Spades", 7)])
    dupe_card = Card.random_dupe(hand)
    two_pair_card = Card.random_two_pair(hand)

    assert dupe_card.value in {4, 7}
    assert two_pair_card.value in {4}


def test_random_full_house(monkeypatch):
    def fixed_choice(options):
        return options[0]

    monkeypatch.setattr("gym_envs.pseudo.card.choice", fixed_choice)

    hand = Hand(
        [
            Card("Hearts", 5),
            Card("Diamonds", 5),
            Card("Clubs", 5),
            Card("Spades", 7),
            Card("Hearts", 7),
        ]
    )
    full_house_card = Card.random_full_house(hand)
    assert full_house_card.value == 2


def test_random_vanilla_card_has_no_upgrades():
    random.seed(0)
    card = Card.random(vanilla_only=True)
    assert card.enhancement == 0
    assert card.edition == 0
    assert card.seal == 0
