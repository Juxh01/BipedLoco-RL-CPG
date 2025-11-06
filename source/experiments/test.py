from typing import Any, Dict, cast

from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

from source.environments.EnvironmentHandler import EnvironmentHandler

CONFIG_DIR = Path(__file__).parents[2] / "configs"


@hydra.main(config_path=str(CONFIG_DIR), config_name="example.yaml", version_base="1.1")
def main(cfg: DictConfig) -> float:
    # Cast Hydra DictConfig to a plain dict for type checkers; at runtime it's mapping-compatible
    env = EnvironmentHandler.create_environment(cast(Dict[str, Any], cfg))
    env = VecMonitor(env)

    # === Separates Eval-Env (NICHT vektorisiert!) ===
    # Falls dein Factory immer VecEnv baut: erzwinge num_envs=1 / eval_mode=True
    eval_cfg = cast(Dict[str, Any], cfg).copy()
    eval_cfg["env"]["num_envs"] = 1
    eval_env = EnvironmentHandler.create_environment(eval_cfg)
    eval_env = Monitor(eval_env)  # wichtig fürs Evaluations-Logging

    # eval_freq: bei VecEnv ~timesteps/num_envs
    num_envs = cfg.env.num_envs
    print(f"num_envs: {num_envs}")
    eval_freq = max(1, 100_000 // num_envs)

    # Loggt eval/mean_reward u. eval/mean_ep_length nach tb_logs/PPO_1/
    eval_cb = EvalCallback(
        eval_env,
        log_path="tb_logs/eval",
        eval_freq=eval_freq,  # nach Bedarf
        deterministic=True,
        n_eval_episodes=20,
    )

    # Gymnasium/Gym compatibility: reset may return (obs, info) or obs
    _reset_out = env.reset()
    if isinstance(_reset_out, tuple) and len(_reset_out) == 2:
        obs, _info = _reset_out
    else:
        obs = _reset_out
    print(f"Initial observation: {obs}")
    print(f"Environment Obs: {env.observation_space}")
    print(f"Environment Act: {env.action_space}")

    total_batch = 65536 // 2
    n_steps = int(total_batch / cfg.env.num_envs)

    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        n_steps=n_steps,
        tensorboard_log="./tensorboard",
    )

    model.learn(total_timesteps=int(10000000), progress_bar=True, callback=eval_cb)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"agent-{timestamp}")

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
