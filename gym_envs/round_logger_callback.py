from typing import Dict, Tuple
import argparse
import gymnasium as gym
import numpy as np
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.algorithm import Algorithm
from functools import partial
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    # EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from copy import deepcopy


def _update_bias(worker, bias):
    worker.foreach_env(lambda env: env.set_bias(bias))


def _update_rarities(worker, rarities):
    worker.foreach_env(lambda env: env.set_rarities(rarities))


def get_biases(worker):
    return worker.foreach_env(lambda env: env.get_bias())


def get_and_reset_stats(worker):
    return worker.foreach_env(lambda env: env.get_and_reset_stats())


class RoundLoggerCallback(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        # assert episode.length == 0, (
        #     "ERROR: `on_episode_start()` callback should be called right "
        #     "after env reset!"
        # )
        # # Create lists to store angles in
        # episode.user_data["pole_angles"] = []
        # episode.hist_data["pole_angles"] = []
        pass

    # def on_episode_step(
    #     self,
    #     *,
    #     worker: RolloutWorker,
    #     base_env: BaseEnv,
    #     policies: Dict[str, Policy],
    #     episode: Episode,
    #     env_index: int,
    #     **kwargs,
    # ):
    #     # Make sure this episode is ongoing.
    #     assert episode.length > 0, (
    #         "ERROR: `on_episode_step()` callback should not be called right "
    #         "after env reset!"
    #     )
    #     pole_angle = abs(episode.last_observation_for()[2])
    #     raw_angle = abs(episode.last_raw_obs_for()[2])
    #     assert pole_angle == raw_angle
    #     episode.user_data["pole_angles"].append(pole_angle)

    #     # Sometimes our pole is moving fast. We can look at the latest velocity
    #     # estimate from our environment and log high velocities.
    #     if np.abs(episode.last_info_for()["pole_angle_vel"]) > 0.25:
    #         print("This is a fast pole!")

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        # blind_env = base_env.get_sub_environments()[env_index].blind_env
        # shop_env = base_env.get_sub_environments()[env_index].shop_env
        # episode.custom_metrics["round"] = blind_env.round
        blind_env = base_env.get_sub_environments()[env_index]

        hand_counts = blind_env.hand_counts
        hand_counts_flat = []
        for i, hand_type in enumerate(sorted(hand_counts.keys())):
            if hand_counts[hand_type] > 0:
                hand_counts_flat += [i] * hand_counts[hand_type]

        episode.hist_data["hands_counts"] = hand_counts_flat

        count_counts = blind_env.count_counts
        count_counts_flat = []
        for i, count_type in enumerate(sorted(count_counts.keys())):
            if count_counts[count_type] > 0:
                count_counts_flat += [i] * count_counts[count_type]

        episode.hist_data["count_counts"] = count_counts_flat

        card_slot_counts = blind_env.card_slot_counts
        card_slot_counts_flat = []
        for i, card_slot_type in enumerate(sorted(card_slot_counts.keys())):
            if card_slot_counts[card_slot_type] > 0:
                card_slot_counts_flat += [i] * card_slot_counts[card_slot_type]

        episode.hist_data["card_slot_counts"] = card_slot_counts_flat

        for hand_type in hand_counts:
            episode.custom_metrics[f"hand_{hand_type}"] = hand_counts[hand_type]

        # episode.custom_metrics["final_joker_count"] = len(shop_env.owned_jokers)
        # episode.custom_metrics["reroll_count"] = shop_env.reroll_count
        # episode.custom_metrics["jokers_purchased"] = shop_env.jokers_purchased
        # episode.custom_metrics["jokers_sold"] = shop_env.jokers_sold
        episode.custom_metrics["hands_played"] = blind_env.hands_played
        episode.custom_metrics["discards_played"] = blind_env.discards_played

        episode.custom_metrics["chips"] = blind_env.chips

    # def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
    #     # We can also do our own sanity checks here.
    #     assert (
    #         samples.count == 2000
    #     ), f"I was expecting 2000 here, but got {samples.count}!"

    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger=None,
        result: dict,
        **kwargs,
    ) -> None:
        hands = [
            "High card",
            "Pair",
            "Two pair",
            "Three of a kind",
            "Straight",
            "Flush",
            "Full house",
            "Four of a kind",
            "Straight flush",
        ]
        # 0    "Flush",
        # 1    "Four of a kind",
        # 2    "Full house",
        # 3    "High card",
        # 4    "Pair",
        # 5    "Straight",
        # 6    "Straight flush",
        # 7    "Three of a kind",
        # 8    "Two pair",

        # confusion, target_counts, hit_rates, scored_ranks, scored_suits
        stats = algorithm.workers.foreach_worker(func=get_and_reset_stats)
        stats = [item for sublist in stats for item in sublist]
        # confusion = np.sum([stat[0] for stat in stats], axis=0)
        target_counts = {hand: 0 for hand in hands}
        hit_rates = {hand: 0 for hand in hands}
        scored_ranks = {hand: np.zeros(13, dtype=np.float32) for hand in hands}
        scored_suits = {hand: np.zeros(4, dtype=np.float32) for hand in hands}
        for stat in stats:
            for hand in hands:
                target_counts[hand] += stat["target_counts"][hand]
                hit_rates[hand] += stat["hit_rates"][hand]
                scored_ranks[hand] += stat["scored_ranks"][hand]
                scored_suits[hand] += stat["scored_suits"][hand]
        hands_on_target = {
            hand: (
                hit_rates[hand] / target_counts[hand]
                if target_counts[hand] > 0
                else 0.0
            )
            for hand in hands
        }

        # rank_hist = []

        biases = algorithm.workers.foreach_worker(func=get_biases)
        # Flatten the list of lists of biases
        biases = [item for sublist in biases for item in sublist]
        biases = biases[0]
        new_biases = deepcopy(biases)
        # new_biases = {
        #     hand: (bias * 0.99 if bias > 0.01 else 0.0)
        #     for hand, bias in new_biases.items()
        # }

        # flushes = result[ENV_RUNNER_RESULTS]["custom_metrics"]["hand_Flush_mean"]
        # mean_return = result[ENV_RUNNER_RESULTS]["episode_return_mean"]

        for hand in hands:
            target_hit_rate = 2.0
            actual_hit_rate = hands_on_target[hand]
            new_biases[hand] -= 0.005
            # if actual_hit_rate < target_hit_rate:
            #     new_biases[hand] *= 0.99
            # elif actual_hit_rate > target_hit_rate:
            #     new_biases[hand] -= 0.05
            if new_biases[hand] < 0.01:
                new_biases[hand] = 0.00
            new_biases[hand] = np.clip(new_biases[hand], 0.0, 0.99)
            result[ENV_RUNNER_RESULTS]["custom_metrics"][f"bias_{hand}"] = new_biases[
                hand
            ]
            result[ENV_RUNNER_RESULTS]["custom_metrics"][f"hit_rate_{hand}"] = (
                hands_on_target[hand]
            )
        average_bias = np.mean(list(new_biases.values()))
        result[ENV_RUNNER_RESULTS]["custom_metrics"]["average_bias"] = average_bias

        hand_means = {}
        for hand_type in hands:
            hand_means[hand_type] = result[ENV_RUNNER_RESULTS]["custom_metrics"][
                f"hand_{hand_type}_mean"
            ]
        denom = max(hand_means.values())
        hand_rarities = {k: (denom - v) / denom for k, v in hand_means.items()}

        # new_bias = average_bias
        # if mean_return < target_return:
        #     new_bias = average_bias + 0.01
        # elif mean_return > target_return:
        #     new_bias = average_bias - 0.05

        # new_bias = np.clip(new_bias, 0.0, 0.9)

        algorithm.workers.foreach_worker(func=partial(_update_bias, bias=new_biases))
        algorithm.workers.foreach_worker(
            func=partial(_update_rarities, rarities=hand_rarities)
        )

        # result[ENV_RUNNER_RESULTS]["custom_metrics"]["bias"] = new_bias

        # for i in range(confusion.shape[0]):
        #     confusion_inverted = []
        #     for j in range(confusion.shape[1]):
        #         confusion_inverted.append([j] * int(confusion[i][j]))
        #     result[ENV_RUNNER_RESULTS]["custom_metrics"][
        #         f"confusion {hands[i]}"
        #     ] = confusion_inverted

    # def on_learn_on_batch(
    #     self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    # ) -> None:
    #     result["sum_actions_in_train_batch"] = train_batch["actions"].sum()
    #     # Log the sum of actions in the train batch.
    #     print(
    #         "policy.learn_on_batch() result: {} -> sum actions: {}".format(
    #             policy, result["sum_actions_in_train_batch"]
    #         )
    #     )

    # def on_postprocess_trajectory(
    #     self,
    #     *,
    #     worker: RolloutWorker,
    #     episode: Episode,
    #     agent_id: str,
    #     policy_id: str,
    #     policies: Dict[str, Policy],
    #     postprocessed_batch: SampleBatch,
    #     original_batches: Dict[str, Tuple[Policy, SampleBatch]],
    #     **kwargs,
    # ):
    #     if "num_batches" not in episode.custom_metrics:
    #         episode.custom_metrics["num_batches"] = 0
    #     episode.custom_metrics["num_batches"] += 1
