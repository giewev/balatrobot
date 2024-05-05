import ray
from ray.rllib.algorithms.ppo import PPOConfig
from gym_envs.balatro_hierarchical_env import BalatroHierarchicalEnv
from gym_envs.balatro_blind_env import BalatroBlindEnv
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner

if __name__ == "__main__":
    model_name = "ppo_balatro_blind_env"

    load = True

    ray.init()

    def policy_mapping(agent_id, episode, **kwargs):
        if agent_id == "blind":
            return "blind_policy"
        else:
            return "shop_policy"

    algo = (
        PPOConfig()
        .environment(
            env=BalatroHierarchicalEnv,
            env_config={},
        )
        .experimental(_enable_new_api_stack=True)
        .env_runners(
            env_runner_cls=MultiAgentEnvRunner,
            num_env_runners=1,
            num_envs_per_env_runner=1,
            sample_timeout_s=60,
            rollout_fragment_length=32,
        )
        .training(
            model={"uses_new_env_runners": True},
            train_batch_size=128,
            sgd_minibatch_size=32,
        )
        .multi_agent(
            policies={"blind_policy", "shop_policy"},
            policy_mapping_fn=policy_mapping,
        )
    ).build()

    print("Training for 5 iterations")

    for _ in range(5):
        print(algo.train())
    # algo.evaluate()
