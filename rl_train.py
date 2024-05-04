from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
import numpy as np
from gym_envs.balatro_blind_env import BalatroBlindEnv
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)
from demo import get_latest_vecnormalize, get_latest_model_path


class dummyconfig:
    def __init__(self, rank):
        self.worker_index = rank


def make_env(rank):
    def _init():
        env = BalatroBlindEnv(dummyconfig(rank))
        return env

    return _init


if __name__ == "__main__":
    model_name = "ppo_balatro_blind_env"
    algo = PPO

    load = True

    num_envs = 2
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    env = VecMonitor(env)
    if not load:
        env = VecNormalize(
            env,
            norm_reward=True,
            clip_obs=10.0,
            norm_obs_keys=["chips", "hand_ranks", "hands_left", "discards_left"],
        )
    else:
        env = VecNormalize.load(
            get_latest_vecnormalize("./model_snapshots", model_name), env
        )

    callbacks = [
        CheckpointCallback(
            save_freq=(512),
            save_path="./model_snapshots/",
            name_prefix=model_name,
            save_replay_buffer=True,
            save_vecnormalize=True,
        ),
        # Add back later once we have custom metrics for this environment
        # TensorboardCallback()
    ]

    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=np.array([128, 128, 128]))

    if load:
        model = PPO.load(get_latest_model_path("./model_snapshots", model_name), env)
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            n_steps=128 // num_envs,
            batch_size=16,
        )

    model.learn(
        total_timesteps=20_000,
        tb_log_name=model_name,
        reset_num_timesteps=not load,
        # reset_num_timesteps=True,
        callback=callbacks,
    )
    model.save(f"./saved_models/{model_name}")
    env.close()
