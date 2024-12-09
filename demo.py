from stable_baselines3 import PPO
import os
import glob
from gym_envs.real_balatro.blind_env import BalatroBlindEnv
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
)


def get_latest_vecnormalize(directory, pattern):
    return get_latest_with_file_extension(directory, pattern, "pkl")


def get_latest_model_path(directory, pattern):
    return get_latest_with_file_extension(directory, pattern, "zip")


def get_latest_with_file_extension(directory, pattern, extension):
    list_of_files = glob.glob(f"{directory}/*{pattern}*.{extension}")
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file


def load_latest_model(directory, pattern, env, previous_model_path=None):
    latest_model_path = get_latest_model_path(directory, pattern)
    if latest_model_path and latest_model_path != previous_model_path:
        model = PPO.load(latest_model_path, env)
        return model, latest_model_path
    return None, None


if __name__ == "__main__":
    model_name = "ppo_balatro_blind_env"
    env = BalatroBlindEnv(12375)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(
        get_latest_vecnormalize("./model_snapshots", model_name), env
    )
    env.training = False
    env.norm_reward = False

    model, latest_model_path = load_latest_model("./model_snapshots", model_name, env)
    if model is not None:
        print(f"Loading with {latest_model_path}")
    else:
        print("No model snapshots found.")

    obs = env.reset()
    n_steps = 10000
    for _ in range(n_steps):
        act = model.predict(obs, deterministic=True)[0]
        obs, reward, done, info = env.step(act)

        new_model, new_model_path = load_latest_model(
            "./model_snapshots", model_name, env, latest_model_path
        )
        if new_model is not None:
            obs = env.reset()
            model = PPO.load(new_model_path, env)
            latest_model_path = new_model_path
            print(f"Model updated. Reloading with {latest_model_path}.")
        elif done:
            obs = env.reset()
