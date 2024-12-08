import enum
import json
import pandas as pd


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

with open("./gym_envs/joker_data.json") as joker_data_file:
    ALL_JOKER_DATA = json.load(joker_data_file)


# effects = set()
# for x in ALL_JOKER_DATA:
#     print(x["config"])
#     print(x.get("effect", None))
#     effects.add(x.get("effect", None))
# print(effects)


# df = pd.DataFrame(ALL_JOKER_DATA)


# def unpack_dictionary_column(df, column):
#     return pd.concat([df.drop([column], axis=1), df[column].apply(pd.Series)], axis=1)


# for col in df.columns:
#     if df[col].dtype == "object":
#         df = unpack_dictionary_column(df, col)
# print(df.columns)
# print(df)

# for col in df.columns:
#     print(df[col].unique())

joker_flags_df = pd.read_csv("./gym_envs/joker_flags.csv")

for joker in ALL_JOKER_DATA:
    flags = joker_flags_df[joker_flags_df["name"] == joker["name"]].iloc[0]
    joker["flags"] = []
    for flag in joker_flags_df.columns:
        if flag not in ["name", "description"]:
            joker["flags"].append(flags[flag])
    # print(joker)
    # print(len(joker["flags"]))
