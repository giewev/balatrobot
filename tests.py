import unittest
from gym_envs.pseudo.hand import Hand
from gym_envs.pseudo.card import Card
from gym_envs.pseudo.deck import Deck
from copy import deepcopy


class TestHand(unittest.TestCase):
    def setUp(self):
        self.straight_hand = Hand(
            [
                Card("Hearts", 10),
                Card("Spades", 9),
                Card("Diamonds", 8),
                Card("Clubs", 7),
                Card("Hearts", 6),
            ]
        )

        self.flush_hands = []
        for suit in Card.SUITS:
            cards = [Card(suit, i) for i in range(2, 14, 2)]
            self.flush_hands.append(Hand(cards))

        self.high_bias = 0.95
        self.medium_bias = 0.5
        self.low_bias = 0.2

    def test_add_card(self):
        self.straight_hand.add_card(Card("hearts", 5))
        self.assertEqual(len(self.straight_hand.cards), 6)

    def test_pop_cards(self):
        popped_hand = self.straight_hand.pop_cards([1, 2])
        self.assertEqual(len(popped_hand.cards), 2)
        self.assertEqual(len(self.straight_hand.cards), 3)

    def test_sort(self):
        self.straight_hand.sort()
        self.assertEqual(
            self.straight_hand.cards,
            sorted(self.straight_hand.cards, key=lambda x: x.value),
        )

    def test_run_lengths(self):
        mixed_suited_runs_hand = Hand(
            [
                Card("Hearts", 12),
                Card("Hearts", 10),
                Card("Hearts", 9),
                Card("Hearts", 8),
                Card("Hearts", 7),
                Card("Hearts", 6),
                Card("Spades", 5),
                Card("Spades", 4),
                Card("Clubs", 3),
                Card("Spades", 2),
                Card("Spades", 14),
            ]
        )

        unsuited_run_counts = mixed_suited_runs_hand.card_run_counts(suited=False)
        self.assertEqual(
            unsuited_run_counts, [1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        )
        suited_run_counts = mixed_suited_runs_hand.card_run_counts(suited=True)
        self.assertEqual(suited_run_counts, [1, 5, 5, 5, 5, 5, 2, 2, 1, 2, 2])

    def test_shuffle(self):
        original_order = self.straight_hand.cards[:]
        self.straight_hand.shuffle()
        self.assertNotEqual(self.straight_hand.cards, original_order)

    def test_card_dupe_counts(self):
        self.straight_hand.add_card(Card("Hearts", 10))
        rank_counts, suit_counts = self.straight_hand.card_dupe_counts()
        self.assertEqual(rank_counts.count(2), 2)
        self.assertEqual(suit_counts.count(3), 3)

    def test_card_run_counts(self):
        run_counts = self.straight_hand.card_run_counts()
        self.assertEqual(run_counts, [5, 5, 5, 5, 5])

    def test_multiples(self):
        self.straight_hand.add_card(Card("hearts", 10))
        multiples = self.straight_hand.multiples()
        self.assertEqual(multiples, {10: 2})

    def test_contained_hand_types(self):
        hand_types = self.straight_hand.contained_hand_types()
        self.assertEqual(hand_types, {"High card", "Straight"})

    def test_evaluate_straight(self):
        hand_type, best_hand = self.straight_hand.evaluate()
        self.assertEqual(hand_type, "Straight")
        self.assertEqual(best_hand.cards, self.straight_hand.cards)

    def test_evaluate_flush(self):
        for flush_hand in self.flush_hands:
            hand_type, best_hand = flush_hand.evaluate()
            self.assertEqual(hand_type, "Flush")
            self.assertEqual(best_hand.cards, flush_hand.cards)

    def test_evaluate_full_house(self):
        full_house_hand = Hand(
            [
                Card("Hearts", 10),
                Card("Spades", 10),
                Card("Diamonds", 10),
                Card("Clubs", 7),
                Card("Hearts", 7),
            ]
        )
        hand_type, best_hand = full_house_hand.evaluate()
        self.assertEqual(hand_type, "Full house")
        self.assertEqual(best_hand.cards, full_house_hand.cards)

    def test_evaluate_four_of_a_kind(self):
        four_of_a_kind_hand = Hand(
            [
                Card("Hearts", 10),
                Card("Spades", 10),
                Card("Diamonds", 10),
                Card("Clubs", 10),
                Card("Hearts", 7),
            ]
        )
        should_score = [x for x in four_of_a_kind_hand.cards if x.value == 10]

        hand_type, best_hand = four_of_a_kind_hand.evaluate()
        self.assertEqual(hand_type, "Four of a kind")
        self.assertEqual(best_hand.cards, should_score)

    def test_evaluate_straight_flush(self):
        straight_flush_hand = Hand(
            [
                Card("Hearts", 10),
                Card("Hearts", 9),
                Card("Hearts", 8),
                Card("Hearts", 7),
                Card("Hearts", 6),
            ]
        )
        hand_type, best_hand = straight_flush_hand.evaluate()
        self.assertEqual(hand_type, "Straight flush")
        self.assertEqual(best_hand.cards, straight_flush_hand.cards)

    def test_evaluate_high_card(self):
        high_card_hand = Hand(
            [
                Card("Hearts", 10),
                Card("Spades", 4),
                Card("Diamonds", 8),
                Card("Clubs", 7),
                Card("Hearts", 6),
            ]
        )
        hand_type, best_hand = high_card_hand.evaluate()
        self.assertEqual(hand_type, "High card")
        self.assertEqual(best_hand.cards, [Card("Hearts", 10)])

    def test_evaluate_three_of_a_kind(self):
        three_of_a_kind_hand = Hand(
            [
                Card("Hearts", 10),
                Card("Spades", 10),
                Card("Diamonds", 10),
                Card("Clubs", 7),
                Card("Hearts", 6),
            ]
        )
        should_score = [x for x in three_of_a_kind_hand.cards if x.value == 10]

        hand_type, best_hand = three_of_a_kind_hand.evaluate()
        self.assertEqual(hand_type, "Three of a kind")
        self.assertEqual(best_hand.cards, should_score)

    def test_evaluate_two_pair(self):
        two_pair_hand = Hand(
            [
                Card("Hearts", 10),
                Card("Spades", 10),
                Card("Diamonds", 7),
                Card("Clubs", 7),
                Card("Hearts", 6),
            ]
        )
        should_score = [x for x in two_pair_hand.cards if x.value in [10, 7]]

        hand_type, best_hand = two_pair_hand.evaluate()
        self.assertEqual(hand_type, "Two pair")
        self.assertEqual(best_hand.cards, should_score)

    def test_evaluate_pair(self):
        pair_hand = Hand(
            [
                Card("Hearts", 10),
                Card("Spades", 10),
                Card("Diamonds", 8),
                Card("Clubs", 7),
                Card("Hearts", 6),
            ]
        )
        should_score = [x for x in pair_hand.cards if x.value == 10]

        hand_type, best_hand = pair_hand.evaluate()
        self.assertEqual(hand_type, "Pair")
        self.assertEqual(best_hand.cards, should_score)

    def test_ace_low_straight(self):
        ace_low_straight = Hand(
            [
                Card("Hearts", 2),
                Card("Spades", 3),
                Card("Diamonds", 4),
                Card("Clubs", 5),
                Card("Hearts", 14),
            ]
        )
        hand_type, best_hand = ace_low_straight.evaluate()
        self.assertEqual(hand_type, "Straight")
        self.assertEqual(best_hand.cards, ace_low_straight.cards)

    def test_straight_flush_biaser(self):
        straight_flush_hand = Hand(
            [
                Card("Hearts", 10),
                Card("Hearts", 9),
                Card("Hearts", 8),
                Card("Hearts", 7),
                Card("Hearts", 6),
            ]
        )
        biaser = straight_flush_hand.straight_flush_biaser()
        high_cards = [Card("Hearts", 11), Card("Hearts", 5)]
        low_cards = [
            Card("Hearts", 4),
            Card("Hearts", 2),
            Card("Hearts", 6),
            Card("Hearts", 10),
            Card("Spades", 5),
            Card("Spades", 11),
        ]

        for h in high_cards:
            for l in low_cards:
                self.assertGreater(biaser(h), biaser(l))

    def test_biased_draw_straight(self):
        # Since the bias is probabilistic, we make sure it works at least 9/10 times
        bias = self.high_bias
        successes = 0
        for _ in range(100):
            hand = Hand([])
            deck = Deck()
            for _ in range(5):
                biaser = hand.straight_biaser()
                hand.add_card(deck.draw_biased(biaser, bias))
            if hand.evaluate()[0] == "Straight":
                successes += 1
        self.assertGreaterEqual(successes, 90)

    def test_biased_draw_straight_flush(self):
        # Since the bias is probabilistic, we make sure it works at least 9/10 times
        bias = self.high_bias
        successes = 0
        for _ in range(100):
            hand = Hand([])
            deck = Deck()
            for _ in range(5):
                biaser = hand.straight_flush_biaser()
                hand.add_card(deck.draw_biased(biaser, bias))
            if hand.evaluate()[0] == "Straight flush":
                successes += 1
            else:
                print(hand.cards)
        self.assertGreaterEqual(successes, 90)

    def test_med_biased_draw_straight_flush(self):
        # Since the bias is probabilistic, we make sure it works at least 9/10 times
        bias = self.medium_bias
        successes = 0
        for _ in range(100):
            hand = Hand([])
            deck = Deck()
            for _ in range(5):
                biaser = hand.straight_flush_biaser()
                hand.add_card(deck.draw_biased(biaser, bias))
            if hand.evaluate()[0] == "Straight flush":
                successes += 1
            else:
                print(hand.cards)
        self.assertGreaterEqual(successes, 30)
        self.assertLessEqual(successes, 70)

    def test_low_biased_draw_straight_flush(self):
        # Since the bias is probabilistic, we make sure it works at least 9/10 times
        bias = self.low_bias
        successes = 0
        for _ in range(100):
            hand = Hand([])
            deck = Deck()
            for _ in range(5):
                biaser = hand.straight_flush_biaser()
                hand.add_card(deck.draw_biased(biaser, bias))
            if hand.evaluate()[0] == "Straight flush":
                successes += 1
            else:
                print(hand.cards)
        self.assertGreaterEqual(successes, 1)
        self.assertLessEqual(successes, 20)

    def test_biased_draw_four_of_kind(self):
        # Since the bias is probabilistic, we make sure it works at least 9/10 times
        bias = self.high_bias
        successes = 0
        for _ in range(100):
            hand = Hand([])
            deck = Deck()
            for _ in range(5):
                biaser = hand.rank_biaser()
                hand.add_card(deck.draw_biased(biaser, bias))
            if hand.evaluate()[0] == "Four of a kind":
                successes += 1
        self.assertGreaterEqual(successes, 90)

    def test_biased_draw_full_house(self):
        # Since the bias is probabilistic, we make sure it works at least 9/10 times
        bias = self.high_bias
        successes = 0
        for _ in range(100):
            hand = Hand([])
            deck = Deck()
            for _ in range(6):
                biaser = hand.rank_biaser()
                hand.add_card(deck.draw_biased(biaser, bias))
            if "Full house" in hand.contained_hand_types():
                successes += 1
        self.assertGreaterEqual(successes, 90)


if __name__ == "__main__":

    unittest.main()
