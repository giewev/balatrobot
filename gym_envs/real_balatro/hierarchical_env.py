from ray.rllib.env import MultiAgentEnv
from gymnasium import spaces as sp
from gym_envs.real_balatro.blind_env import BalatroBlindEnv
from gym_envs.real_balatro.shop_env import BalatroShopEnv
from balatro_connection import BalatroConnection
import time


class BalatroHierarchicalEnv(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        print(env_config)
        self.balatro_connection = None
        self.port = env_config.worker_index + 12348
        self.hand_size = 8
        self.deck = "Blue Deck"
        self.stake = 1
        self.challenge = None
        self.seed = None
        self.postponed_blind_reward = 0
        self.last_chip_count = 0
        self.shop_seen = False
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

        self._spaces_in_preferred_format = True

    def reset(self, seed=None, options=None):
        print("Being reset")
        self.postponed_blind_reward = 0
        self.shop_seen = False
        if (
            self.balatro_connection is None
            or time.time() - self.balatro_connection.start_time > 60 * 60
        ):
            if self.balatro_connection is not None:
                self.balatro_connection.stop_balatro_instance()
            self.balatro_connection = BalatroConnection(bot_port=self.port)
            if not self.balatro_connection.poll_state():
                print(self.port)
                # print("failed to connect")
                # exit()
                print("Starting Balatro instance")
                self.balatro_connection.start_balatro_instance()
                time.sleep(10)
            self.blind_env.balatro_connection = self.balatro_connection
            self.shop_env.balatro_connection = self.balatro_connection
        self.blind_env.start_new_game()
        blind_obs, blind_infos = self.blind_env.reset()
        # print("Blind reset")
        # print(blind_obs)
        return {
            "blind": blind_obs,
            # "shop": self.shop_env.observation_space.sample(),
        }, {
            "blind": blind_infos,
            # "shop": {},
        }

    def step(self, action):
        results = {}
        game_over = False
        if "blind" in action:
            blind_action = action["blind"]
            obs, reward, term, trunc, info = self.blind_env.step(blind_action)
            # obs = self.blind_env.observation_space.sample()
            # if term or trunc:
            #     obs = self.blind_env.observation_space.sample()
            # results["blind"] = (obs, reward, term, trunc, info)
            if term or trunc:
                if info["game_over"] == 1:
                    obs = self.blind_env.observation_space.sample()
                    results["blind"] = (obs, reward, term, trunc, info)
                    print("Game over")
                    game_over = True
                else:
                    self.postponed_blind_reward = reward
                    shop_obs, shop_infos = self.shop_env.reset()
                    # shop_obs = self.shop_env.observation_space.sample()
                    if self.shop_seen:
                        reward = 1.0
                    else:
                        reward = 0.0
                    results["shop"] = (shop_obs, reward, False, False, shop_infos)
                    self.shop_seen = True
            else:
                results["blind"] = (obs, reward, term, trunc, info)
        elif "shop" in action:
            shop_action = action["shop"]
            obs, reward, term, trunc, info = self.shop_env.step(shop_action)
            # obs = self.shop_env.observation_space.sample()
            if info["shop_ended"]:
                blind_obs, blind_infos = self.blind_env.reset()
                # blind_obs = self.blind_env.observation_space.sample()
                results["blind"] = (
                    blind_obs,
                    self.postponed_blind_reward,
                    False,
                    False,
                    {"game_over": 0.0},
                )
            else:
                results["shop"] = (obs, reward, term, trunc, info)

        # if game_over and not self.shop_seen:
        #     results["shop"] = (
        #         self.shop_env.observation_space.sample(),
        #         0.0,
        #         True,
        #         False,
        #         {},
        #     )

        # Invert the dictionary to get tuple of dictionaries from agent name to tuple of obs, reward, term, trunc, info
        inverted_results = [{}, {}, {}, {}, {}]
        for agent_name, values in results.items():
            for i in range(5):
                inverted_results[i][agent_name] = values[i]

        inverted_results[2]["__all__"] = game_over
        inverted_results[3]["__all__"] = False
        if len(inverted_results[0]) == 0:
            print("No obs returned")
        # print(inverted_results[0])
        # print(inverted_results[1])

        # if "blind" in inverted_results[0]:
        #     if inverted_results[0]["blind"] not in self.blind_env.observation_space:
        #         print(f'Blind obs not in space {inverted_results[0]["blind"]}')
        # if "shop" in inverted_results[0]:
        #     if inverted_results[0]["shop"] not in self.shop_env.observation_space:
        #         print(f'Shop obs not in space {inverted_results[0]["shop"]}')

        return tuple(inverted_results)

    def close(self):
        # self.balatro_connection.stop_balatro_instance()
        pass
