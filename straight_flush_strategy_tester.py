# Updated intelligent discarding strategy simulation
import random
from collections import Counter
from gym_envs.pseudo.deck import Deck
from gym_envs.pseudo.hand import Hand
from gym_envs.pseudo.card import Card


# Define a function to check for a straight flush
def is_straight_flush(hand):
    values = [card[0] for card in hand]
    suits = [card[1] for card in hand]
    value_counts = Counter(values)

    # Check each suit
    for suit in set(suits):
        suit_values = [value for value, s in zip(values, suits) if s == suit]
        suit_values.sort()

        # Check for low-Ace straight flush (A-2-3-4-5)
        if suit_values[:5] == [2, 3, 4, 5, 14]:
            return True

        # # Check for high-Ace straight flush (10-J-Q-K-A)
        # if suit_values[-4:] == [10, 11, 12, 13] and suit_values[0] == 1:
        #     return True

        # Check for other straight flushes
        for i in range(len(suit_values) - 4):
            if suit_values[i : i + 5] == list(
                range(suit_values[i], suit_values[i] + 5)
            ):
                return True

    return False


def choose_discards(hand, deck):
    combined = hand[:] + deck[:]
    combined = [Card(suit, value) for value, suit in combined]
    combined_hand = Hand(combined)
    suited_run_lengths = combined_hand.card_run_counts(suited=True)
    should_discard = []
    for i, card in enumerate(hand):
        if suited_run_lengths[i] < 5:
            should_discard.append(i)
    should_discard = should_discard[:5]

    # Keep the cards with the highest run lengths
    while len(should_discard) < 5:
        l, h = min(suited_run_lengths[: len(hand)]), max(
            suited_run_lengths[: len(hand)]
        )

        if l == h:
            break

        for i, run_length in enumerate(suited_run_lengths[: len(hand)]):
            if run_length < h and i not in should_discard:
                should_discard.append(i)
                break
        else:
            break

    return should_discard


target_range = list(range(4, 11))


def choose_discard_dumb(hand, deck, suit="H"):
    # Always try to keep 2,3,4,5,6,7,8,9 of hearts
    # Discard the rest

    should_discard = []
    for i, card in enumerate(hand):
        if card[0] not in target_range:
            should_discard.append(i)
        elif card[1] != suit:
            should_discard.append(i)
    should_discard = should_discard[:5]

    return should_discard


def choose_discard_smart(hand, deck):
    suit = choose_suit(hand)
    return choose_discard_dumb(hand, deck, suit)


def choose_suit(hand):
    scores = {suit: 0 for suit in "CDHS"}
    for i, card in enumerate(hand):
        if card[0] in target_range:
            scores[card[1]] += 1

    if max(scores.values()) == 0:
        return None
    return max(scores, key=scores.get)


# Function to simulate a single game
def simulate_game():
    deck = [(value, suit) for value in range(2, 15) for suit in "CDHS"]
    random.shuffle(deck)

    hand = [deck.pop() for _ in range(8)]
    discards_left = 7
    suit = None

    while discards_left > 0:
        if is_straight_flush(hand):
            return True

        # Evaluate potential straight flushes
        # potential = evaluate_potential(hand)
        # best_suit = max(potential, key=potential.get)

        # Keep cards that have potential to form a straight flush
        # hand = [card for card in hand if card[1] == best_suit]
        if suit is None:
            suit = choose_suit(hand)
        discard_indices = choose_discard_dumb(hand, deck, suit=suit)
        # discard_indices = choose_discard_smart(hand, deck)
        # discard_indices = choose_discards(hand, deck)
        hand = [card for i, card in enumerate(hand) if i not in discard_indices]

        # Discard the rest and draw new cards
        discard_count = min(5, 8 - len(hand))
        hand += [deck.pop() for _ in range(discard_count)]

        discards_left -= 1

    return is_straight_flush(hand)


# Run the simulation
def run_simulation(n_games):
    wins = sum(simulate_game() for _ in range(n_games))
    return wins / n_games


# Run the simulation for 10,000 games
win_probability = run_simulation(10000)
print(win_probability)
