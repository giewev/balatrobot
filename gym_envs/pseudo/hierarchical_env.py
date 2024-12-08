from ray.rllib.env import MultiAgentEnv
from gymnasium import spaces as sp
from gym_envs.pseudo.blind_env import PseudoBlindEnv
from gym_envs.pseudo.shop_env import PseudoShopEnv
from balatro_connection import BalatroConnection
import time


class PseudoHierarchicalEnv(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        self.postponed_blind_reward = 0
        self.shop_seen = False
        self._agent_ids = {"blind", "shop"}
        self._spaces_in_preferred_format = True

        self.blind_env = PseudoBlindEnv(env_config)
        self.shop_env = PseudoShopEnv(env_config)

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
        # print("Being reset")
        self.postponed_blind_reward = 0
        self.shop_seen = False
        blind_obs, blind_infos = self.blind_env.reset()
        shop_obs, shop_infos = self.shop_env.reset()
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
            if term or trunc:
                if info["game_over"] == 1:
                    results["blind"] = (obs, reward, term, trunc, info)
                    game_over = True
                else:
                    self.postponed_blind_reward = reward
                    self.shop_env.dollars += 5
                    self.shop_env.new_shop()
                    shop_obs = self.shop_env.get_obs()
                    if self.shop_seen:
                        reward = 1.0
                    else:
                        reward = 0.0
                    results["shop"] = (shop_obs, reward, False, False, {})
                    self.shop_seen = True
            else:
                results["blind"] = (obs, reward, term, trunc, info)
        elif "shop" in action:
            shop_action = action["shop"]
            obs, reward, term, trunc, info = self.shop_env.step(shop_action)
            if info["shop_ended"]:
                self.blind_env.jokers = self.shop_env.owned_jokers
                blind_obs = self.blind_env.get_obs()
                results["blind"] = (
                    blind_obs,
                    self.postponed_blind_reward,
                    False,
                    False,
                    {"game_over": 0.0},
                )
            else:
                results["shop"] = (obs, reward, term, trunc, info)

        # Invert the dictionary to get tuple of dictionaries from agent name to tuple of obs, reward, term, trunc, info
        inverted_results = [{}, {}, {}, {}, {}]
        for agent_name, values in results.items():
            for i in range(5):
                inverted_results[i][agent_name] = values[i]

        inverted_results[2]["__all__"] = game_over
        inverted_results[3]["__all__"] = False
        if len(inverted_results[0]) == 0:
            print("No obs returned")
        return tuple(inverted_results)

    def close(self):
        pass
