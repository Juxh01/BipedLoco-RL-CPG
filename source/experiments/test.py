from pathlib import Path

import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from source.environments.EnvironmentHandler import EnvironmentHandler

CONFIG_DIR = Path(__file__).parents[2] / "configs"


@hydra.main(config_path=str(CONFIG_DIR), config_name="example.yaml", version_base="1.1")
def main(cfg: DictConfig) -> float:
    env = EnvironmentHandler.create_environment(cfg)
    obs = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Environment Obs: {env.observation_space}")
    print(f"Environment Act: {env.action_space}")

    model = PPO("MlpPolicy", env, device="cpu", verbose=1)

    model.learn(total_timesteps=int(1000000), progress_bar=True)

    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10
    )
    print(f"mean reward: {mean_reward}, std reward: {std_reward}")


"""     while True:
        # deterministic=True für stabilere Demo
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)

        # Render ruft der Env-Treiber selbst auf, aber explizit geht auch:
        env.unwrapped.mj_render()

        if terminated or truncated:
            obs, _ = env.reset()
 """
# return 0.0


if __name__ == "__main__":
    main()
