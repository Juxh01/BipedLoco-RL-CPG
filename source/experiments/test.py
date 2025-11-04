from typing import Any, Dict, cast

from pathlib import Path

import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from source.environments.EnvironmentHandler import EnvironmentHandler

CONFIG_DIR = Path(__file__).parents[2] / "configs"


@hydra.main(config_path=str(CONFIG_DIR), config_name="example.yaml", version_base="1.1")
def main(cfg: DictConfig) -> float:
    # Cast Hydra DictConfig to a plain dict for type checkers; at runtime it's mapping-compatible
    env = EnvironmentHandler.create_environment(cast(Dict[str, Any], cfg))

    # Gymnasium/Gym compatibility: reset may return (obs, info) or obs
    _reset_out = env.reset()
    if isinstance(_reset_out, tuple) and len(_reset_out) == 2:
        obs, _info = _reset_out
    else:
        obs = _reset_out
    print(f"Initial observation: {obs}")
    print(f"Environment Obs: {env.observation_space}")
    print(f"Environment Act: {env.action_space}")

    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
    )

    model.learn(total_timesteps=int(1000000), progress_bar=True)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"mean reward: {mean_reward}, std reward: {std_reward}")

    if isinstance(mean_reward, list):
        ret = (
            float(sum(mean_reward) / len(mean_reward)) if len(mean_reward) > 0 else 0.0
        )
    else:
        ret = float(mean_reward)
    return ret


"""
    while True:
        # deterministic=True für stabilere Demo
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)

        # Render ruft der Env-Treiber selbst auf, aber explizit geht auch:
        env.unwrapped.mj_render()
"""


if __name__ == "__main__":
    main()
