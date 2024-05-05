import enum


class Suit(enum.Enum):
    Clubs = 0
    Diamonds = 1
    Hearts = 2
    Spades = 3


rank_lookup = {
    "Ace": 14,
    "King": 13,
    "Queen": 12,
    "Jack": 11,
    "Ten": 10,
    "Nine": 9,
    "Eight": 8,
    "Seven": 7,
    "Six": 6,
    "Five": 5,
    "Four": 4,
    "Three": 3,
    "Two": 2,
    "10": 10,
    "9": 9,
    "8": 8,
    "7": 7,
    "6": 6,
    "5": 5,
    "4": 4,
    "3": 3,
    "2": 2,
}
