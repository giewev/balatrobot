import ray
from ray.rllib.algorithms.ppo import PPOConfig
from gym_envs.balatro_blind_env import BalatroBlindEnv
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner

if __name__ == "__main__":
    model_name = "ppo_balatro_blind_env"

    load = True

    ray.init()
    algo = (
        PPOConfig()
        .environment(
            env=BalatroBlindEnv,
            env_config={},
        )
        .experimental(_enable_new_api_stack=True)
        .env_runners(
            env_runner_cls=SingleAgentEnvRunner,
            sample_timeout_s=60,
            rollout_fragment_length=32,
        )
        .training(
            model={"uses_new_env_runners": True},
            train_batch_size=128,
            sgd_minibatch_size=32,
        )
    ).build()

    for _ in range(5):
        print(algo.train())
    algo.evaluate()
