from gym_envs.pseudo.card import Card
from random import shuffle


class Hand:
    def __init__(self, cards=[]):
        self.cards = cards[:]

    def add_card(self, card):
        self.cards.append(card)

    def pop_cards(self, indices):
        popped = []
        for i in sorted(indices, reverse=True):
            popped.append(self.cards.pop(i - 1))
        hand = Hand()
        hand.cards = popped
        return hand

    def sort(self):
        self.cards = list(sorted(self.cards, key=lambda x: x.value))

    def shuffle(self):
        shuffle(self.cards)

    def card_dupe_counts(self):
        rank_counts = []
        suit_counts = []
        for card in self.cards:
            if card.value is None:
                rank_counts.append(0)
                suit_counts.append(0)
                continue
            rank_counts.append(len([x for x in self.cards if x.value == card.value]))
            suit_counts.append(len([x for x in self.cards if x.suit == card.suit]))
        return rank_counts, suit_counts

    def card_run_counts(self, suited=False):
        ranks = {card.value for card in self.cards}
        run_counts = []
        for card in self.cards:
            if card.value is None:
                run_counts.append(0)
                continue
            if suited:
                ranks = {c.value for c in self.cards if c.suit == card.suit}
            run_up = 0
            run_down = 0
            v = card.value
            low_v = v
            if v == 14:
                low_v = 1
            for i in range(low_v + 1, 15):
                if i in ranks:
                    run_up += 1
                else:
                    break

            for i in range(v - 1, 0, -1):
                if i in ranks:
                    run_down += 1
                elif i == 1 and 14 in ranks:
                    run_down += 1
                else:
                    break
            run_counts.append(run_up + run_down + 1)
        return run_counts

    def multiples(self):
        multiples = {}
        for card in self.cards:
            if card.value in multiples:
                multiples[card.value] += 1
            else:
                multiples[card.value] = 1

        multiples = {k: v for k, v in multiples.items() if v > 1}
        return multiples

    def contained_hand_types(self):
        available = set()
        multiples = self.multiples()
        multiple_counts = list(multiples.values())
        if len(self.cards) > 0:
            available.add("High card")
        if len(multiples) > 0:
            available.add("Pair")
            if len(multiples) > 1:
                available.add("Two pair")
                if any([x >= 3 for x in multiple_counts]):
                    available.add("Full house")
            if any([x >= 3 for x in multiple_counts]):
                available.add("Three of a kind")
            if any([x >= 4 for x in multiple_counts]):
                available.add("Four of a kind")

        if self.longest_run() >= 5:
            available.add("Straight")
        for suit in Card.SUITS:
            suit_hand = Hand([card for card in self.cards if card.suit == suit])
            if len(suit_hand) >= 5:
                available.add("Flush")
                if suit_hand.longest_run() >= 5:
                    available.add("Straight flush")

        return available

    def evaluate(self):
        contained_hands = self.contained_hand_types()
        for hand_type in [
            "Straight flush",
            "Four of a kind",
            "Full house",
            "Flush",
            "Straight",
            "Three of a kind",
            "Two pair",
            "Pair",
            "High card",
        ]:
            if hand_type in contained_hands:
                hand = hand_type
                break
        if hand in ["Straight flush", "Flush", "Straight", "Full house"]:
            return hand, Hand(self.cards)
        multiples = self.multiples()
        if hand == "High card":
            return hand, Hand([max(self.cards, key=lambda x: x.value)])
        return hand, Hand([x for x in self.cards if x.value in multiples])

    def longest_run(self):
        ranks = {card.value for card in self.cards}
        if 14 in ranks:
            ranks.add(1)
        rank_hand = sorted(list(ranks))
        highest_run = 1
        run = 1
        for i in range(1, len(rank_hand)):
            if rank_hand[i] - rank_hand[i - 1] == 1:
                run += 1
            else:
                highest_run = max(highest_run, run)
                run = 1
        highest_run = max(highest_run, run)

        return highest_run

    def general_biaser(self):
        rank_counts = {value: 0 for value in Card.RANKS}
        suit_counts = {suit: 0 for suit in Card.SUITS}
        for card in self.cards:
            rank_counts[card.value] += 1
            suit_counts[card.suit] += 1

        def calculate_bias(card):
            return rank_counts[card.value] + suit_counts[card.suit] * 2

        return calculate_bias

    def rank_biaser(self):
        rank_counts = {value: 0 for value in Card.RANKS}
        for card in self.cards:
            rank_counts[card.value] += 1

        def calculate_bias(card):
            return rank_counts[card.value] * 15

        return calculate_bias

    def suit_biaser(self):
        suit_counts = {suit: 0 for suit in Card.SUITS}
        for card in self.cards:
            suit_counts[card.suit] += 1

        def calculate_bias(card):
            return suit_counts[card.suit] * 10

        return calculate_bias

    def straight_biaser(self):
        adjacency_counts = {value: 0 for value in Card.RANKS}
        for card in self.cards:
            if card.value == 14:
                # adjacency_counts[2] += 1
                pass
            else:
                adjacency_counts[card.value + 1] += 1

            if card.value == 2:
                # adjacency_counts[14] += 1
                pass
            else:
                adjacency_counts[card.value - 1] += 1

        # We don't want to draw duplicates when we have a partial straight
        for card in self.cards:
            adjacency_counts[card.value] = 0

        def calculate_bias(card):
            return adjacency_counts[card.value] * 15

        return calculate_bias

    def straight_flush_biaser(self):
        adjacency_counts = {
            (suit, value): 0 for value in Card.RANKS for suit in Card.SUITS
        }
        for card in self.cards:
            if card.value == 14:
                # adjacency_counts[(card.suit, 2)] += 1
                pass
            else:
                adjacency_counts[(card.suit, card.value + 1)] += 1

            if card.value == 2:
                # adjacency_counts[(card.suit, 14)] += 1
                pass
            else:
                adjacency_counts[(card.suit, card.value - 1)] += 1

        # We don't want to draw duplicates when we have a partial straight
        for card in self.cards:
            adjacency_counts[(card.suit, card.value)] = 0

        def calculate_bias(card):
            return adjacency_counts[(card.suit, card.value)] * 10

        return calculate_bias

    def __str__(self):
        return f"Hand: {self.cards}"

    def __len__(self):
        return len(self.cards)

    def __iter__(self):
        return iter(self.cards)
