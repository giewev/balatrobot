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
from gym_envs.joker_effects import supported_jokers


def _update_bias(worker, bias):
    worker.foreach_env(lambda env: env.set_bias(bias))


def _update_rarities(worker, rarities):
    worker.foreach_env(lambda env: env.set_rarities(rarities))


def get_biases(worker):
    return worker.foreach_env(lambda env: env.get_bias())


def get_and_reset_stats(worker):
    return worker.foreach_env(lambda env: env.get_and_reset_stats())


def get_lesson(worker):
    return worker.foreach_env(lambda env: env.get_lesson())


def set_lesson(worker, lesson):
    worker.foreach_env(lambda env: env.set_lesson(lesson))


class CurriculumCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.hands = [
            "Flush",  # 0
            "Four of a kind",  # 1
            "Full house",  # 2
            "High card",  # 3
            "Pair",  # 4
            "Straight",  # 5
            "Straight flush",  # 6
            "Three of a kind",  # 7
            "Two pair",  # 8
        ]

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
        curr_env = base_env.get_sub_environments()[env_index]
        blind_env = curr_env.blind_env

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

        episode.custom_metrics["hands_played"] = blind_env.hands_played
        episode.custom_metrics["discards_played"] = blind_env.discards_played

        episode.custom_metrics["chips"] = blind_env.chips

    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger=None,
        result: dict,
        **kwargs,
    ) -> None:
        stats = algorithm.workers.foreach_worker(func=get_and_reset_stats)
        stats = [item for sublist in stats for item in sublist]
        # confusion = np.sum([stat[0] for stat in stats], axis=0)
        target_counts = {hand: 0 for hand in self.hands}
        hit_rates = {hand: 0 for hand in self.hands}
        scored_ranks = {hand: np.zeros(13, dtype=np.float32) for hand in self.hands}
        scored_suits = {hand: np.zeros(4, dtype=np.float32) for hand in self.hands}
        for stat in stats:
            for hand in self.hands:
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
            for hand in self.hands
        }

        custom_metrics = result[ENV_RUNNER_RESULTS]["custom_metrics"]

        chips = custom_metrics["chips"]
        custom_metrics["chips_99th"] = np.percentile(chips, 99)
        custom_metrics["chips_mean"] = np.mean(chips)
        custom_metrics["chips_1st"] = np.percentile(chips, 1)

        train_stats = {
            "target_counts": target_counts,
            "hit_rates": hit_rates,
            "scored_ranks": scored_ranks,
            "scored_suits": scored_suits,
            "hands_on_target": hands_on_target,
            "chips": chips,
            "chips_99th": custom_metrics["chips_99th"],
            "chips_1st": custom_metrics["chips_1st"],
            "chips_mean": custom_metrics["chips_mean"],
        }

        hand_means = {}
        for hand_type in self.hands:
            custom_metrics[f"hit_rate_{hand_type}"] = hands_on_target[hand_type]
            hand_means[hand_type] = custom_metrics[f"hand_{hand_type}_mean"]
        lesson = algorithm.workers.foreach_worker(func=get_lesson)
        lesson = [item for sublist in lesson for item in sublist]
        lesson = lesson[0]
        step = lesson["step"]
        diff = lesson["diff"]

        step, diff = self.adjust_lesson(step, diff, train_stats)

        algorithm.workers.foreach_worker(
            func=partial(set_lesson, lesson={"step": step, "diff": diff})
        )

        for hand in diff:
            custom_metrics[f"bias_{hand}"] = diff[hand]["bias"]
            custom_metrics[f"max_discards_{hand}"] = diff[hand]["discards"]

    def adjust_lesson(self, step, diff, stats):
        if step == "play_target_hand":
            if all([diff[hand]["bias"] == 0.0 for hand in self.hands]):
                step = "play_for_chips"
                diff = {
                    "discards": 3,
                    "plays": 4,
                }
            else:
                hands_on_targets = stats["hands_on_target"]
                new_diff = deepcopy(diff)
                for hand in self.hands:
                    target_hit_rate = 0.25
                    actual_hit_rate = hands_on_targets[hand]
                    if actual_hit_rate > target_hit_rate:
                        new_diff[hand]["bias"] *= 0.75
                        new_diff[hand]["bias"] *= 1.01
                    if new_diff[hand]["bias"] < 0.9:
                        new_diff[hand]["bias"] -= 0.01
                    new_diff[hand]["discards"] = np.clip(
                        new_diff[hand]["discards"], 1, 7
                    )
                    new_diff[hand]["bias"] = np.clip(new_diff[hand]["bias"], 0.0, 0.99)

                return step, new_diff
        if step == "play_for_chips":
            chips_1st = stats["chips_1st"]
            if chips_1st > 300:
                step = "chips_varied_deck"
                diff = {"discards": 3, "plays": 4, "deck_var": 0.1}
            else:
                return step, diff
        if step == "chips_varied_deck":
            chips_mean = stats["chips_mean"]
            if chips_mean > 600:
                step = "chips_varied_joker"
                all_jokers = supported_jokers()
                diff = {
                    "discards": 3,
                    "plays": 4,
                    "joker_count_weights": [4, 1, 0.0, 0.0, 0.0, 0.0],
                }
            else:
                return step, diff
        if step == "chips_varied_joker":
            