import json
from gym_envs.pseudo.play_hand_type_env import PlayHandTypeEnv
from random import choice, randint, sample
from copy import deepcopy
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.policy.sample_batch import SampleBatch


def generate_data():
    env = PlayHandTypeEnv(
        env_config={
            "max_hand_size": 8,
            "correct_reward": 1.0,
            "incorrect_penalty": 0.3,
            "discard_penalty": 0.05,
        }
    )

    num_steps = 100000
    trajectories = []
    obs, info = env.reset()
    for i in range(num_steps):
        # Play flushes or discard otherwise
        hand = env.hand
        suit_counts = {suit: 0 for suit in env.suit_pool}
        for card in hand:
            suit_counts[card["suit"]] += 1

        flush_suit = max(suit_counts, key=suit_counts.get)
        flush_indices = [i for i, card in enumerate(hand) if card["suit"] == flush_suit]
        # Check if we can play a flush
        if len(flush_indices) >= 5:
            # Select 5 random indices of the flush suit
            flush_indices = sample(flush_indices, 5)
            cards = [0] * 8
            for index in flush_indices:
                cards[index] = 1

            # Play the flush play, 1, 5 (zero index), cards
            action = [1, 4] + cards
        else:
            # Discard up to 5 cards that are not part of the flush
            discard_indices = [
                i for i, card in enumerate(hand) if card["suit"] != flush_suit
            ]
            discard_indices = sample(discard_indices, min(5, len(discard_indices)))
            cards = [0] * 8
            for index in discard_indices:
                cards[index] = 1

            # Discard the cards, 0, card count (zero index), cards
            action = [0, len(discard_indices) - 1] + cards
        new_obs, reward, done, truncated, info = env.step(action)
        trajectories.append(
            {
                "obs": obs,
                "action": action,
                "reward": reward,
                "new_ob": new_obs,
                "done": done,
            }
        )
        if done or truncated:
            obs, info = env.reset()
        else:
            obs = new_obs

    # Invert trajectories into a dict of lists instead of a list of dicts
    trajectories = {k: [d[k] for d in trajectories] for k in trajectories[0]}

    batch = SampleBatch(trajectories)
    writer = JsonWriter("./offline_data")
    writer.write(batch)
    # with open("./offline_data.json", "w") as f:
    #     json.dump(trajectories, f)


if __name__ == "__main__":
    generate_data()
