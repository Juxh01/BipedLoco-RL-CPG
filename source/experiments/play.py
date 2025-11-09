from typing import Any, Dict, cast

import os
from pathlib import Path

# ---- Thread-/BLAS-Config muss VOR dem Import schwerer Libs passieren ----


os.environ["MUJOCO_GL"] = "glfw"

import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from source.environments.EnvironmentHandler import EnvironmentHandler

CONFIG_DIR = Path(__file__).parents[2] / "configs"

timestamp = "20251104_120301"

# __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python source/experiments/play.py


@hydra.main(config_path=str(CONFIG_DIR), config_name="example.yaml", version_base="1.1")
def main(cfg: DictConfig) -> float:
    # Cast Hydra DictConfig to a plain dict for type checkers; at runtime it's mapping-compatible
    cfg.env.num_envs = 1

    # Create raw environment first so we can wrap it with video recorder BEFORE vectorizing
    raw_env = EnvironmentHandler.create_environment(cast(Dict[str, Any], cfg))

    # Now wrap into vector envs / monitors / normalization as before
    env = DummyVecEnv([lambda: raw_env])
    env = VecMonitor(env)
    stats_env = VecNormalize.load(
        os.path.join(
            hydra.utils.get_original_cwd(),
            "Training",
            "PPO_1",
            "best_model_vecnormalize.pkl",
        ),
        env,
    )
    stats_env.training = False
    stats_env.norm_reward = False
    # Gymnasium/Gym compatibility: reset may return (obs, info) or obs
    _reset_out = env.reset()
    if isinstance(_reset_out, tuple) and len(_reset_out) == 2:
        obs, _info = _reset_out
    else:
        obs = _reset_out
    print(f"Initial observation: {obs}")
    print(f"Environment Obs: {env.observation_space}")
    print(f"Environment Act: {env.action_space}")

    dir = os.path.join(
        hydra.utils.get_original_cwd(), "Training", "PPO_1", "best_model.zip"
    )

    model = PPO.load(dir, env=stats_env, device="cpu", eval_mode=True)

    mean_reward, std_reward = evaluate_policy(model, stats_env, n_eval_episodes=20)
    print(f"mean reward: {mean_reward}, std reward: {std_reward}")

    while True:
        # deterministic=True für stabilere Demo
        obs = stats_env.normalize_obs(obs)
        action, _ = model.predict(obs, deterministic=True)

        (
            obs,
            _,
            terminated,
            truncated,
            _,
        ) = raw_env.step(action)
        # Render ruft der Env-Treiber selbst auf, aber explizit geht auch:
        raw_env.unwrapped.mj_render()
        if terminated or truncated:
            obs, _ = raw_env.reset()

    return 0


if __name__ == "__main__":
    main()
