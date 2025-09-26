from gym_envs.pseudo import Card, Deck, ErraticDeck


def test_deck_reset_and_draw(monkeypatch):
    monkeypatch.setattr("gym_envs.pseudo.deck.shuffle", lambda seq: None)
    deck = Deck()
    initial_len = len(deck)

    card = deck.draw()
    assert isinstance(card, Card)
    assert len(deck) == initial_len - 1

    deck.reset()
    assert len(deck) == len(deck.all_cards)


def test_deck_add_card_and_empty_draw(monkeypatch):
    monkeypatch.setattr("gym_envs.pseudo.deck.shuffle", lambda seq: None)
    deck = Deck()
    extra = Card("Hearts", 2)
    deck.add_card(extra)
    assert extra in deck.all_cards

    deck.remaining_cards = []
    assert deck.draw() is None


def test_draw_biased_uses_weights(monkeypatch):
    deck = Deck()
    deck.remaining_cards = [Card("Clubs", 3), Card("Hearts", 14)]

    captured_weights = {}

    def fake_choices(indices, weights):
        captured_weights["weights"] = list(weights)
        return [indices[1]]

    monkeypatch.setattr("gym_envs.pseudo.deck.choices", fake_choices)

    def bias(card):
        return 10 if card.value == 14 else 0

    drawn = deck.draw_biased(bias, bias=1.0)
    assert drawn.value == 14
    assert captured_weights["weights"][1] > captured_weights["weights"][0]


def test_infinite_deck_draw(monkeypatch):
    deck = Deck(infinite=True)
    deck.all_cards = [Card("Clubs", 2), Card("Spades", 5)]

    def fake_choice(cards):
        return cards[1]

    monkeypatch.setattr("gym_envs.pseudo.deck.choice", fake_choice)

    before = len(deck)
    card = deck.draw()
    assert card.value == 5
    assert len(deck) == before  # infinite decks leave the pool unchanged


def test_from_gamestate_deck():
    gamestate = [
        {"suit": "Hearts", "value": 10},
        {"suit": "Spades", "value": "Ace"},
    ]
    deck = Deck.from_gamestate_deck(gamestate)
    assert len(deck.remaining_cards) == 2
    assert deck.remaining_cards[0] == Card("Hearts", 10)
    assert deck.remaining_cards[1] == Card("Spades", 14)


def test_erratic_deck_initialises_full(monkeypatch):
    monkeypatch.setattr("gym_envs.pseudo.deck.shuffle", lambda seq: None)
    deck = ErraticDeck()
    assert len(deck.all_cards) == 52
    assert len(deck.remaining_cards) == 52
