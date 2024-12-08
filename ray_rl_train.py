import os

os.environ["RAY_DEDUP_LOGS"] = "0"
import ray
import ray.rllib
from ray.rllib.algorithms.ppo import PPOConfig
from gym_envs.balatro.hierarchical_env import BalatroHierarchicalEnv
from gym_envs.balatro_blind_env import BalatroBlindEnv
from gym_envs.balatro_shop_env import BalatroShopEnv
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
from ray.rllib.connectors.env_to_module import (
    AddObservationsFromEpisodesToBatch,
    FlattenObservations,
    WriteObservationsToEpisodes,
    AgentToModuleMapping,
)
from balatro_blind_model import BalatroBlindModel
from ray.rllib.models import ModelCatalog

from gymnasium import spaces as sp
from ray.rllib.algorithms.algorithm import Algorithm
import torch.nn as nn


def policy_mapping(agent_id, episode, **kwargs):
    if agent_id == "blind":
        return "blind_policy"
    else:
        return "shop_policy"


def _env_to_module_pipeline(env):
    return [
        AddObservationsFromEpisodesToBatch(),
        FlattenObservations(multi_agent=True, agent_ids={"shop"}),
        WriteObservationsToEpisodes(),
    ]


global shared_joker_layer
shared_joker_layer = nn.LSTM(
    input_size=18,
    hidden_size=256,
    num_layers=1,
    batch_first=True,
)

if __name__ == "__main__":
    model_name = "ppo_blind_shop_experimental"

    ray.init()

    ModelCatalog.register_custom_model("my_model", BalatroBlindModel)

    # algo = Algorithm.from_checkpoint(f"model_snapshots/{model_name}/latest/")
    # # algo.evaluation_config["create_env_on_driver"] = True

    # while True:
    #     algo.train()
    #     # algo.evaluate()
    # exit()

    algo = (
        PPOConfig()
        .environment(
            env=BalatroHierarchicalEnv,
            env_config={},
        )
        .framework("torch")
        # .experimental(_enable_new_api_stack=True)
        .env_runners(
            # env_runner_cls=MultiAgentEnvRunner,
            # env_to_module_connector=_env_to_module_pipeline,
            num_env_runners=0,
            num_envs_per_env_runner=1,
            sample_timeout_s=60,
            rollout_fragment_length=64,
            # observation_filter="MeanStdFilter",
            # batch_mode="complete_episodes",
        )
        # .resources(num_gpus=1)
        .training(
            # model={"uses_new_env_runners": True},
            train_batch_size=256,
            sgd_minibatch_size=64,
        )
        .evaluation(
            evaluation_num_env_runners=1,
            # evaluation_interval=1,
            evaluation_force_reset_envs_before_iteration=True,
            # evaluation_parallel_to_training=True,
        )
        .multi_agent(
            policies={
                "blind_policy": (
                    None,
                    BalatroBlindEnv.build_observation_space(),
                    BalatroBlindEnv.build_action_space(8),
                    {
                        "model": {
                            "custom_model": "my_model",
                        }
                    },
                ),
                "shop_policy": (
                    None,
                    BalatroShopEnv.build_observation_space(),
                    BalatroShopEnv.build_action_space(),
                    {},
                ),
            },
            policy_mapping_fn=policy_mapping,
        )
        # .resources(num_gpus=1)
    ).build()

    i = 0
    snapshot_interval = 50
    while True:
        algo.train()
        algo.save(f"model_snapshots/{model_name}/latest")
        if i % snapshot_interval == 0:
            algo.save(f"model_snapshots/{model_name}/snapshot_{i}")
        i += 1
    # algo.evaluate()
