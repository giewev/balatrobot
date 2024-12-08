import os

os.environ["RAY_DEDUP_LOGS"] = "0"
import ray
from ray import tune, train
import ray.rllib
from ray.rllib.algorithms.ppo import PPOConfig
from gym_envs.pseudo.hierarchical_env import PseudoHierarchicalEnv
from gym_envs.pseudo.blind_env import PseudoBlindEnv
from gym_envs.pseudo.shop_env import PseudoShopEnv
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
from ray.rllib.connectors.env_to_module import (
    AddObservationsFromEpisodesToBatch,
    FlattenObservations,
    WriteObservationsToEpisodes,
    AgentToModuleMapping,
)
from ray.tune.search.bayesopt import BayesOptSearch


# from balatro_blind_model import BalatroBlindModel
# from modeling.autoregressive_blind_model import BalatroBlindModel
# from modeling.parametric_blind_model import BalatroBlindModel
from modeling.parametric_sequential_blind_model import (
    ParametricSequentialBalatroBlindModel,
)
from modeling.parametric_sequential_play_hand_model import (
    ParametricSequentialBalatroPlayHandModel,
)
from modeling.balatro_shop_model import BalatroShopModel
from modeling.n_choose_k_sequential_dist import NChooseKDistribution
from modeling.sequential_choice_distribution import SequentialChoiceDistribution
from modeling.n_choose_k_simultaneous_dist import NChooseKSimultaneousDistribution
from modeling.parametric_play_hand_model import ParametricBalatroPlayHandModel
from modeling.play_discard_choose_dist import PlayDiscardChooseDist
from modeling.modal_multinomial_dist import ModalMultinomialDist
from modeling.para_attention_play_hand_model import ParametricAttentionPlayHandModel
from modeling.attention_blind_model import AttentionBlindModel
from modeling.attention_blind_deck_model import AttentionBlindDeckModel
from modeling.new_api.attention_blind_module import AttentionBlindModule
from modeling.new_api.ppo_torch_auxilliary_learner import PPOTorchAuxilliaryLearner
from modeling.modal_multibinary_dist import ModalMultibinaryDist
from modeling.all_combos_dist import AllCombosDist
from modeling.combo_index_dist import ComboIndexDist
from ray.rllib.models import ModelCatalog

from gymnasium import spaces as sp
from ray.rllib.algorithms.algorithm import Algorithm
from gym_envs.pseudo.blind_env import PseudoBlindEnv
from gym_envs.pseudo.curriculum_env import CurriculumEnv
from gym_envs.pseudo.play_hand_type_env import PlayHandTypeEnv
import numpy as np
import torch
from gym_envs.round_logger_callback import RoundLoggerCallback
from gym_envs.curriculum_callback import CurriculumCallback
from numpy import array, float32
from ray.tune.search.optuna import OptunaSearch
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
import os

# os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
# torch.distributed.init_process_group(backend="gloo")


def policy_mapping(agent_id, episode, **kwargs):
    if agent_id == "blind":
        return "blind_policy"
    else:
        return "shop_policy"


def play_hand_config():
    return (
        PPOConfig()
        .environment(
            env=PlayHandTypeEnv,
            env_config={
                "max_hand_size": 8,
                # "correct_reward": tune.loguniform(0.1, 3.0),
                "correct_reward": 1.0,
                # "incorrect_penalty": tune.loguniform(0.1, 3.0),
                "incorrect_penalty": 0.3,
                # "discard_penalty": tune.loguniform(0.1, 3.0),
                "discard_penalty": 0.05,
                "rank_count": 13,
                "fixed_ranks": True,
                "suit_count": 4,
                "fixed_suits": True,
                "infinite_deck": True,
                "bias": 1.0,
            },
            observation_space=PlayHandTypeEnv.build_observation_space(8),
            action_space=PlayHandTypeEnv.build_action_space(8),
        )
        # .offline_data(
        #     input_config={
        #         "paths": "./offline_data/output-2024-05-25_11-34-42_worker-0_0.json",
        #         "format": "json",
        #     }
        # )
        .callbacks(RoundLoggerCallback)
        .framework("torch")
        .resources(num_gpus=1, num_cpus_per_worker=1)
        # .resources(num_gpus=1)
        .env_runners(
            num_env_runners=12,
            num_envs_per_env_runner=1,
            sample_timeout_s=60,
            rollout_fragment_length="auto",
            # observation_filter="MeanStdFilter",
            # batch_mode="complete_episodes",
        )
        .training(
            # model={"uses_new_env_runners": True},
            # train_batch_size=tune.choice([256, 512, 1024, 2048, 4096]),
            train_batch_size=int(2**13),
            sgd_minibatch_size=int(2**13),
            num_sgd_iter=10,
            # grad_clip=10,
            lr=1e-3,
            gamma=0.99,
            kl_coeff=0.5,
            # clip_param=0.1,
            entropy_coeff=0.002,
            # vf_clip_param=10.0,
            kl_target=0.01,
            lambda_=0.95,
            vf_loss_coeff=0.5,
            model={
                "custom_model": "attn_play_hand_model",
                "custom_action_dist": "play_discard_dist",
                "vf_share_layers": True,
                "custom_model_config": {
                    "learn_embeddings": True,
                    "card_embedding_size": 64,
                    "hidden_size": 512,
                    "context_size": 512,
                    "num_heads": 4,
                    "num_attention_layers": 4,
                    "num_hidden_layers": 1,
                    "hand_size": 8,
                    # "action_module_size": 256,
                    # "context_i": tune.randint(5, 10),
                    # "context_size": tune.sample_from(
                    #     lambda spec: 2
                    #     ** (spec.config.model.custom_model_config.context_i)
                    # ),
                    # "hidden_i": tune.randint(5, 10),
                    # "hidden_size": tune.sample_from(
                    #     lambda spec: 2
                    #     ** (spec.config.model.custom_model_config.hidden_i)
                    # ),
                    # "action_module_i": tune.randint(5, 10),
                    # "action_module_size": tune.sample_from(
                    #     lambda spec: 2
                    #     ** (spec.config.model.custom_model_config.action_module_i)
                    # ),
                },
            },
        )
    )


def blind_config():
    hand_size = 8
    return (
        PPOConfig()
        .environment(
            env=PseudoBlindEnv,
            env_config={
                "max_hand_size": hand_size,
                "correct_reward": 1.0,
                "incorrect_penalty": 0.3,
                "discard_penalty": 0.05,
                "infinite_deck": False,
                "bias": 1.0,
                "rarity_bonus": 0.0,
            },
            observation_space=PseudoBlindEnv.build_observation_space(hand_size),
            action_space=PseudoBlindEnv.build_action_space(hand_size),
        )
        .callbacks(RoundLoggerCallback)
        .framework("torch")
        .resources(num_gpus=1, num_cpus_per_worker=1)
        # .resources(num_gpus=1)
        .env_runners(
            num_env_runners=15,
            num_envs_per_env_runner=10,
            sample_timeout_s=60,
            rollout_fragment_length="auto",
            # observation_filter="MeanStdFilter",
            batch_mode="complete_episodes",
        )
        .training(
            # model={"uses_new_env_runners": True},
            # train_batch_size=tune.choice([256, 512, 1024, 2048, 4096]),
            train_batch_size=int(2**10),
            sgd_minibatch_size=int(2**10),
            # num_sgd_iter=tune.randint(1, 21),
            num_sgd_iter=1,
            grad_clip=10,
            lr=8e-4,
            # lr=tune.loguniform(1e-6, 1e-3),
            gamma=0.99,
            kl_coeff=0.0,
            # clip_param=0.1,
            entropy_coeff=0.001,
            # entropy_coeff_schedule=[
            #     (0, 0.5),
            #     (int(1e4), 0.1),
            #     (int(3e4), 0.01),
            #     (int(1e5), 0.01),
            #     (int(3e5), 0.005),
            #     (int(5e5), 0.000),
            # ],
            # entropy_coeff=tune.loguniform(1e-3, 5e-2),
            # vf_clip_param=10.0,
            kl_target=0.002,
            # kl_target=tune.loguniform(1e-4, 1e-2),
            # lambda_=tune.uniform(0.8, 1.0),
            lambda_=0.99,
            # vf_loss_coeff=tune.uniform(0.2, 1.0),
            vf_loss_coeff=0.5,
            model={
                "custom_model": "attn_blind_deck_model",
                "custom_action_dist": "combo_index_dist",
                "vf_share_layers": True,
                "custom_model_config": {
                    "learn_embeddings": True,
                    "card_embedding_size": 64,
                    "hidden_size": 256,
                    "context_size": 64,
                    "num_heads": 8,
                    "num_attention_layers": 2,
                    "num_hidden_layers": 1,
                    "hand_size": hand_size,
                    "max_embedding_norm": None,
                },
            },
        )
    )


def curriculum_config():
    hand_size = 8
    return (
        PPOConfig()
        .experimental(_disable_preprocessor_api=True, _enable_new_api_stack=True)
        .environment(
            env=CurriculumEnv,
            env_config={
                "max_hand_size": hand_size,
                "correct_reward": 1.0,
                "incorrect_penalty": 0.3,
                "discard_penalty": 0.05,
                "infinite_deck": False,
                "bias": 0.0,
                "rarity_bonus": 0.0,
            },
            observation_space=CurriculumEnv().observation_space,
            action_space=CurriculumEnv().action_space,
        )
        .callbacks(CurriculumCallback)
        .framework("torch", torch_compile_learner_dynamo_backend="gloo")
        .resources(
            # num_gpus=1,
            num_cpus_per_worker=1,
            num_learner_workers=0,  # <- in most cases, set this value to the number of GPUs
            num_gpus_per_learner_worker=1,  # <- set this to 1, if you have at least 1 GPU
            # num_cpus_for_local_worker=2,
        )
        # .resources(num_gpus=1)
        .env_runners(
            num_env_runners=15,
            num_envs_per_env_runner=20,
            sample_timeout_s=60,
            rollout_fragment_length="auto",
            # observation_filter="MeanStdFilter",
            batch_mode="complete_episodes",
        )
        .rl_module(
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs=SingleAgentRLModuleSpec(
                    module_class=AttentionBlindModule,
                    observation_space=PseudoBlindEnv.build_observation_space(hand_size),
                    action_space=PseudoBlindEnv.build_action_space(hand_size),
                    model_config_dict={
                        "learn_embeddings": True,
                        "card_embedding_size": 32,
                        "hidden_size": 64,
                        "context_size": 64,
                        "num_heads": 4,
                        "num_attention_layers": 1,
                        "num_hidden_layers": 2,
                        "hand_size": hand_size,
                        "max_embedding_norm": None,
                    },
                ),
            )
        )
        .training(
            learner_class=PPOTorchAuxilliaryLearner,
            # model={"uses_new_env_runners": True},
            # train_batch_size=tune.choice([256, 512, 1024, 2048, 4096]),
            train_batch_size=int(2**12),
            sgd_minibatch_size=int(2**12),
            # num_sgd_iter=tune.randint(1, 21),
            num_sgd_iter=3,
            grad_clip=10,
            lr=3e-3,
            # lr=tune.loguniform(1e-6, 1e-3),
            gamma=0.99,
            kl_coeff=0.0,
            # clip_param=0.1,
            # entropy_coeff=0.02,
            # entropy_coeff=[
            #     (0, 0.10),
            #     # (int(2e5), 0.10),
            #     (int(1e5), 0.05),
            #     (int(3e5), 0.01),
            #     (int(4e5), 0.005),
            #     (int(5e5), 0.000),
            # ],
            entropy_coeff=[
                (0, 0.03),
                (int(5e5), 0.00),
            ],
            # entropy_coeff=tune.loguniform(1e-3, 5e-2),
            # vf_clip_param=10.0,
            kl_target=0.01,
            # kl_target=tune.loguniform(1e-4, 1e-2),
            # lambda_=tune.uniform(0.8, 1.0),
            lambda_=0.99,
            # vf_loss_coeff=tune.uniform(0.2, 1.0),
            vf_loss_coeff=0.1,
            model={
                "custom_action_dist": "combo_index_dist",
                "vf_share_layers": True,
            },
        )
        .multi_agent(
            policies={
                "blind_player_policy": (
                    None,
                    PseudoBlindEnv.build_observation_space(hand_size),
                    PseudoBlindEnv.build_action_space(hand_size),
                    {},
                )
            },
            # policy_mapping_fn=lambda agent_id, episode, **kwargs: "blind_player_policy",
            policy_mapping_fn=policy_mapper,
        )
    )


def policy_mapper(agent_id, episode, **kwargs):
    # print(agent_id)
    return "blind_player_policy"


if __name__ == "__main__":

    model_name = "ppo_play_hand_type"
    # torch.autograd.set_detect_anomaly(True)
    ray.init()

    ModelCatalog.register_custom_model(
        "sequential_blind_model", ParametricSequentialBalatroBlindModel
    )
    ModelCatalog.register_custom_model(
        "play_hand_model", ParametricSequentialBalatroPlayHandModel
    )
    ModelCatalog.register_custom_action_dist("auto_reg_dist", NChooseKDistribution)
    ModelCatalog.register_custom_action_dist(
        "sequential_dist", SequentialChoiceDistribution
    )
    ModelCatalog.register_custom_model("custom_shop_model", BalatroShopModel)
    ModelCatalog.register_custom_action_dist(
        "n_choose_k_simul_dist", NChooseKSimultaneousDistribution
    )
    ModelCatalog.register_custom_model(
        "param_play_hand_model", ParametricBalatroPlayHandModel
    )
    ModelCatalog.register_custom_action_dist("play_discard_dist", PlayDiscardChooseDist)
    ModelCatalog.register_custom_model(
        "attn_play_hand_model", ParametricAttentionPlayHandModel
    )
    ModelCatalog.register_custom_model("attn_blind_model", AttentionBlindModel)
    ModelCatalog.register_custom_action_dist(
        "modal_multinomial_dist", ModalMultinomialDist
    )
    ModelCatalog.register_custom_action_dist(
        "modal_multibinary_dist", ModalMultibinaryDist
    )
    ModelCatalog.register_custom_model("attn_blind_deck_model", AttentionBlindDeckModel)
    ModelCatalog.register_custom_action_dist("all_combos_dist", AllCombosDist)
    ModelCatalog.register_custom_action_dist("combo_index_dist", ComboIndexDist)

    # config = hier_config()
    # config = play_hand_config()
    # config = blind_config()
    config = curriculum_config()

    # config = play_hand_config()
    # config = tune.with_resources(config, {"gpu": 0.25, "cpu": 2})

    algo = config.build()

    # results = tune.run(
    #     "PPO",
    #     config=config,
    #     stop={"time_total_s": 600},
    #     # resources_per_trial={"gpu": 0.25, "cpu": 2},
    #     # checkpoint_at_end=True,
    #     # checkpoint_freq=50,
    #     num_samples=-1,
    #     # search_alg=BayesOptSearch(metric="episode_reward_mean", mode="max"),
    #     search_alg=OptunaSearch(
    #         metric="env_runner_results/custom_metrics/chips_mean", mode="max"
    #     ),
    #     # local_dir=f"model_snapshots/{model_name}",
    # )

    # print("Best hyperparameters found were: ", results.get_best_result().config)

    i = 0
    snapshot_interval = 50
    while True:
        algo.train()
        algo.save(f"model_snapshots/{model_name}/latest")
        if i % snapshot_interval == 0:
            algo.save(f"model_snapshots/{model_name}/snapshot_{i}")
        i += 1
