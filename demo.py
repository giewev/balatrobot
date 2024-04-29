from stable_baselines3 import A2C, PPO
import pygame
import os
import glob
from gym_envs.balatro_blind_env import BalatroBlindEnv
from balatro_connection import BalatroConnection
import time


def get_latest_model_path(directory, pattern):
    list_of_files = glob.glob(f"{directory}/*{pattern}*.zip")
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
    connection = BalatroConnection(bot_port=12375)
    connection.start_balatro_instance()
    time.sleep(10)
    env = BalatroBlindEnv(connection)
    model, latest_model_path = load_latest_model("./model_snapshots", model_name, env)
    if model is not None:
        print(f"Loading with {latest_model_path}")
    else:
        print("No model snapshots found.")

    # clock = pygame.time.Clock()

    obs, info = env.reset()
    n_steps = 10000
    # steps_to_catch = 0
    while True:
        act = model.predict(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, info = env.step(act)

        new_model, new_model_path = load_latest_model(
            "./model_snapshots", model_name, env, latest_model_path
        )
        if new_model is not None:
            obs, info = env.reset()
            model = PPO.load(new_model_path, env)
            latest_model_path = new_model_path
            print(f"Model updated. Reloading with {latest_model_path}.")
        elif terminated or truncated:
            obs, info = env.reset()
        # env.render()
        # clock.tick(15)
