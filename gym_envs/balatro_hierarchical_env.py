from ray.rllib.env import MultiAgentEnv
from gymnasium import spaces as sp
from gym_envs.balatro_blind_env import BalatroBlindEnv
from gym_envs.balatro_shop_env import BalatroShopEnv
from balatro_connection import BalatroConnection
import time


class BalatroHierarchicalEnv(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        self.balatro_connection = None
        self.port = env_config.worker_index + 12348
        self.hand_size = 8
        self.deck = "Blue Deck"
        self.stake = 1
        self.challenge = None
        self.seed = None
        self.last_chip_count = 0
        self._agent_ids = {"blind", "shop"}

        env_config["agency_states"] = ["select_cards_from_hand", "select_shop_action"]

        self.blind_env = BalatroBlindEnv(env_config)
        self.shop_env = BalatroShopEnv(env_config)

        self.action_space = sp.Dict(
            {
                "blind": self.blind_env.action_space,
                "shop": self.shop_env.action_space,
            }
        )

        self.observation_space = sp.Dict(
            {
                "blind": self.blind_env.observation_space,
                "shop": self.shop_env.observation_space,
            }
        )

    def reset(self, seed=None, options=None):
        if self.balatro_connection is None:
            self.balatro_connection = BalatroConnection(bot_port=self.port)
            if not self.balatro_connection.poll_state():
                print(self.port)
                print("Starting Balatro instance")
                self.balatro_connection.start_balatro_instance()
                time.sleep(10)
            self.blind_env.balatro_connection = self.balatro_connection
            self.shop_env.balatro_connection = self.balatro_connection
        self.blind_env.start_new_game()
        blind_obs, blind_infos = self.blind_env.reset()
        return {"blind": blind_obs}, {"blind": blind_infos}

    def step(self, action):
        print("HRLEnv step")
        print(action)
        results = {}
        game_over = False
        if "blind" in action:
            print("Blind action")
            blind_action = action["blind"]
            obs, reward, term, trunc, info = self.blind_env.step(blind_action)
            if term or trunc:
                obs = self.blind_env.observation_space.sample()
            results["blind"] = (obs, reward, term, trunc, info)
            if term or trunc:
                if "game_over" in info:
                    game_over = True
                else:
                    shop_obs, shop_infos = self.shop_env.reset()
                    results["shop"] = (shop_obs, None, False, False, shop_infos)
        elif "shop" in action:
            print("Shop action")
            shop_action = action["shop"]
            obs, reward, term, trunc, info = self.shop_env.step(shop_action)
            if term or trunc:
                obs = self.shop_env.observation_space.sample()
            results["shop"] = (obs, reward, term, trunc, info)
            if term or trunc:
                blind_obs, blind_infos = self.blind_env.reset()
                results["blind"] = (blind_obs, None, False, False, blind_infos)

        # Invert the dictionary to get tuple of dictionaries from agent name to tuple of obs, reward, term, trunc, info
        inverted_results = [{}, {}, {}, {}, {}]
        for agent_name, values in results.items():
            for i in range(5):
                inverted_results[i][agent_name] = values[i]

        inverted_results[2]["__all__"] = game_over
        inverted_results[3]["__all__"] = any(inverted_results[3].values())
        print(inverted_results)
        return tuple(inverted_results)

    def close(self):
        self.balatro_connection.stop_balatro_instance()
