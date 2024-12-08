from random import choice


class Card:
    SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
    RANKS = list(range(2, 15))

    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

    def index(self):
        if self.suit is None or self.value is None:
            return 52

        suit_map = {
            "Clubs": 0,
            "Diamonds": 1,
            "Hearts": 2,
            "Spades": 3,
        }
        return suit_map[self.suit] * 13 + self.value - 2

    def __str__(self):
        return f"{self.value} of {self.suit}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.suit == other.suit and self.value == other.value

    def __hash__(self):
        return hash((self.suit, self.value))

    @staticmethod
    def random():
        return Card(choice(Card.SUITS), choice(Card.RANKS))

    @staticmethod
    def random_flush(hand):
        flush_suit = choice(Card.SUITS)
        if len(hand) > 0:
            suit_counts = {suit: 0 for suit in Card.SUITS}
            for card in hand:
                suit_counts[card.suit] += 1
            flush_suit = max(suit_counts, key=suit_counts.get)
        return Card(flush_suit, choice(Card.RANKS))

    @staticmethod
    def random_straight(hand):
        valid_ranks = []
        for card in hand:
            if card.value == 2:
                valid_ranks.append(14)
            else:
                valid_ranks.append(card.value - 1)

            if card.value == 14:
                valid_ranks.append(2)
            else:
                valid_ranks.append(card.value + 1)
        if len(hand) == 0:
            return Card.random()

        for card in hand:
            valid_ranks = [x for x in valid_ranks if x != card.value]
        return Card(choice(Card.SUITS), choice(valid_ranks))

    @staticmethod
    def random_straight_flush(hand):
        flush = Card.random_flush(hand)
        straight = Card.random_straight(hand)

        return Card(flush.suit, straight.value)

    # For pair, three of a kind, and four of a kind
    @staticmethod
    def random_dupe(hand):
        if len(hand) == 0:
            return Card.random()
        dupe_rank = choice([x.value for x in hand if x.value in Card.RANKS])

        return Card(choice(Card.SUITS), dupe_rank)

    @staticmethod
    def random_two_pair(hand):
        if len(hand) == 0:
            return Card.random()
        multiples = hand.multiples()

        # We want to return a dupe of any card that is not in a pair
        non_pair_hand = [x.value for x in hand if x.value not in multiples]
        if len(non_pair_hand) == 0:
            rank = choice([x for x in Card.RANKS if x not in multiples])
            return Card(choice(Card.SUITS), rank)
        return Card(choice(Card.SUITS), choice(non_pair_hand))

    @staticmethod
    def random_full_house(hand):
        if len(hand) == 0:
            return Card.random()
        multiples = hand.multiples()
        triple_ranks = [k for k, v in multiples.items() if v > 2]
        non_triple_ranks = set(Card.RANKS) - set(triple_ranks)
        if len(non_triple_ranks) == 0:
            return Card.random()
        double_ranks = [k for k, v in multiples.items() if v == 2]
        if len(triple_ranks) == 0:
            if len(double_ranks) != 0:
                return Card(choice(Card.SUITS), choice(double_ranks))
            else:
                return Card.random_dupe(hand)
        return Card(choice(Card.SUITS), choice(list(non_triple_ranks)))
