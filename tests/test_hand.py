import random

import pytest

from gym_envs.base_card import BaseCard
from gym_envs.pseudo import Card, Hand


def make_hand(values, suit="Hearts"):
    return Hand([Card(suit, v) for v in values])


def test_add_pop_and_sort(monkeypatch, capsys):
    hand = make_hand([2, 5, 7, 9, 11])
    new_card = Card("Clubs", 13)
    hand.add_card(new_card)

    popped = hand.pop_cards([2, 99])  # 99 should be ignored without crashing
    captured = capsys.readouterr()
    assert "Warning" in captured.out

    assert len(hand) == 5  # only one card removed
    assert popped.cards[0].value == 5

    original = hand.cards[:]

    def reverse(seq):
        seq.reverse()

    monkeypatch.setattr("gym_envs.pseudo.hand.shuffle", reverse)
    hand.shuffle()
    assert hand.cards == list(reversed(original))

    hand.sort()
    assert [card.value for card in hand.cards] == sorted(
        card.value for card in hand.cards
    )


def test_suit_counts_and_homogeneity():
    hand = Hand(
        [
            Card("Hearts", 2),
            Card("Hearts", 4),
            Card("Hearts", 6),
            Card("Spades", 3),
            Card("Spades", 9),
        ]
    )
    counts = hand.suit_counts()
    assert counts["Hearts"] == 3
    assert counts["Spades"] == 2
    assert abs(hand.suit_homogeneity() - ((3 - 2) / 3)) < 1e-6

    short_hand = Hand([Card("Hearts", 2), Card("Spades", 3), Card("Clubs", 4)])
    assert short_hand.suit_homogeneity() == 0.0


def test_mutate_preserves_hand_type():
    hand = Hand(
        [
            Card("Hearts", 10),
            Card("Hearts", 9),
            Card("Hearts", 8),
            Card("Hearts", 7),
            Card("Hearts", 6),
        ]
    )

    random.seed(0)
    mutated = hand.mutate()
    hand_type, _ = mutated.evaluate()
    assert hand_type in {"Flush", "Straight Flush"}
    assert len(mutated.cards) == len(hand.cards)


def test_card_dupe_and_run_counts():
    hand = Hand(
        [
            Card("Hearts", 5),
            Card("Clubs", 5),
            Card("Hearts", 6),
            Card("Hearts", 7),
            Card("Hearts", 8),
        ]
    )
    rank_counts, suit_counts = hand.card_dupe_counts()
    assert rank_counts == [2, 2, 1, 1, 1]
    assert suit_counts == [4, 1, 4, 4, 4]

    unsuited = hand.card_run_counts(suited=False)
    suited = hand.card_run_counts(suited=True)
    assert unsuited == [4, 4, 4, 4, 4]
    assert suited == [4, 1, 4, 4, 4]


@pytest.mark.parametrize(
    "hand_type",
    [
        "Pair",
        "Two Pair",
        "Three of a Kind",
        "Straight",
        "Flush",
        "Full House",
    ],
)
def test_random_prototype_generates_expected_hand(hand_type):
    random.seed(1)
    hand = Hand.random_prototype(hand_type)
    available = hand.contained_hand_types()
    assert hand_type in available


def test_random_hand_size_and_properties():
    random.seed(2)
    hand = Hand.random("Straight", size=7)
    assert len(hand) == 7
    assert "Straight" in hand.contained_hand_types()


def test_multiples_ignore_stones():
    stone = Card("Hearts", 5)
    stone.enhancement = BaseCard.Enhancements.STONE
    hand = Hand([Card("Clubs", 5), stone, Card("Spades", 5)])
    assert hand.multiples() == {5: 2}


def test_contained_hand_types_with_options():
    straight_hand = Hand(
        [
            Card("Hearts", 2),
            Card("Clubs", 3),
            Card("Diamonds", 4),
            Card("Spades", 5),
        ]
    )
    types = straight_hand.contained_hand_types(allow_4_straight=True)
    assert "Straight" in types

    smeared_hand = Hand(
        [
            Card("Hearts", 2),
            Card("Diamonds", 5),
            Card("Hearts", 8),
            Card("Diamonds", 9),
            Card("Spades", 12),
        ]
    )
    types = smeared_hand.contained_hand_types(allow_4_flush=True, smeared=True)
    assert "Flush" in types


def test_evaluate_edge_cases():
    empty_hand = Hand([])
    hand_type, best = empty_hand.evaluate()
    assert hand_type == "High Card"
    assert len(best.cards) == 0

    high_card_hand = Hand([Card("Spades", 12)])
    hand_type, best = high_card_hand.evaluate()
    assert hand_type == "High Card"
    assert best.cards == [Card("Spades", 12)]

    full_house_hand = Hand(
        [
            Card("Hearts", 10),
            Card("Clubs", 10),
            Card("Spades", 10),
            Card("Hearts", 7),
            Card("Clubs", 7),
        ]
    )
    hand_type, best = full_house_hand.evaluate()
    assert hand_type == "Full House"
    assert len(best.cards) == 5


def test_longest_run_handles_low_ace():
    hand = Hand(
        [
            Card("Hearts", 14),
            Card("Clubs", 2),
            Card("Spades", 3),
            Card("Diamonds", 4),
        ]
    )
    assert hand.longest_run() == 4


def test_biasers_return_expected_scores():
    hand = Hand(
        [
            Card("Hearts", 5),
            Card("Hearts", 7),
            Card("Clubs", 5),
        ]
    )

    general = hand.general_biaser()
    rank_bias = hand.rank_biaser()
    suit_bias = hand.suit_biaser()
    straight_bias = hand.straight_biaser()
    straight_flush_bias = hand.straight_flush_biaser()

    assert general(Card("Hearts", 5)) == 6
    assert rank_bias(Card("Hearts", 5)) == 30
    assert suit_bias(Card("Hearts", 9)) == 20
    assert straight_bias(Card("Diamonds", 4)) == 30
    assert straight_flush_bias(Card("Hearts", 6)) == 20
    assert straight_flush_bias(Card("Clubs", 6)) == 10


def test_iteration_helpers():
    hand = make_hand([3, 6, 9])
    assert len(hand) == 3
    assert [card.value for card in hand] == [3, 6, 9]
    assert hand[1].value == 6
