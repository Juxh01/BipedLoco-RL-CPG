from typing import Any, Dict, cast

import os
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from source.environments.EnvironmentHandler import EnvironmentHandler

CONFIG_DIR = Path(__file__).parents[2] / "configs"


class SaveVecNormalizeStatsCallback(BaseCallback):
    """
    Ein Callback, der die Statistiken des VecNormalize-Environments speichert.
    Dieser Callback wird üblicherweise vom EvalCallback ausgelöst.

    :param save_path: Der Pfad, unter dem die Statistiken gespeichert werden sollen.
    """

    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

    def _init_callback(self) -> None:
        # Sicherstellen, dass der Speicherpfad existiert
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Dieser Callback wird vom EvalCallback aufgerufen.
        # self.training_env (das VecNormalize-Env) wird automatisch
        # vom EvalCallback bereitgestellt.

        # Wir speichern die Statistiken neben dem besten Modell
        stats_path = os.path.join(self.save_path, "best_model_vecnormalize.pkl")

        self.training_env.save(stats_path)

        if self.verbose > 0:
            print(f"Saving VecNormalize stats for best model to {stats_path}")

        # Wichtig: True zurückgeben, um das Training nicht abzubrechen
        return True


@hydra.main(config_path=str(CONFIG_DIR), config_name="example.yaml", version_base="1.1")
def main(cfg: DictConfig) -> float:
    # Cast Hydra DictConfig to a plain dict for type checkers; at runtime it's mapping-compatible
    env = EnvironmentHandler.create_environment(cast(Dict[str, Any], cfg))

    # === Separates Eval-Env (NICHT vektorisiert!) ===
    # Falls dein Factory immer VecEnv baut: erzwinge num_envs=1 / eval_mode=True
    eval_cfg = cast(Dict[str, Any], cfg).copy()
    eval_cfg["env"]["num_envs"] = 1
    eval_cfg["env"]["use_observation_normalization"] = False
    eval_cfg["env"]["use_reward_normalization"] = False
    # Verhindere, dass _make_single_env einen Monitor hinzufügt (macht VecMonitor)
    eval_cfg["env"]["log_dir"] = None
    eval_env = EnvironmentHandler.create_environment(eval_cfg)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecMonitor(eval_env)

    norm_obs = cfg.env.get("use_observation_normalization", True)

    if norm_obs:
        eval_env = VecNormalize(eval_env, norm_obs=norm_obs, norm_reward=False)

    # eval_freq: bei VecEnv ~timesteps/num_envs
    num_envs = cfg.env.num_envs
    print(f"num_envs: {num_envs}")
    eval_freq = max(1, 250000 // num_envs)

    # Loggt eval/mean_reward u. eval/mean_ep_length nach tb_logs/PPO_1/
    eval_cb = EvalCallback(
        eval_env,
        log_path="tb_logs/eval",
        best_model_save_path="checkpoints/best",
        eval_freq=eval_freq,  # nach Bedarf
        deterministic=True,
        n_eval_episodes=20,
        callback_after_eval=SaveVecNormalizeStatsCallback(save_path="checkpoints/best"),
    )

    # Regelmäßige Schnappschüsse (z. B. alle 500k/num_envs Schritte)
    ckpt_cb = CheckpointCallback(
        save_freq=4 * eval_freq,  # gleiche Frequenz wie Eval oder was dir passt
        save_path="checkpoints/regular",  # Ordner für laufende Snapshots
        name_prefix="ppo",
    )
    callbacks = CallbackList([eval_cb, ckpt_cb])

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

    model.learn(
        total_timesteps=int(cfg.total_timesteps), progress_bar=True, callback=callbacks
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"checkpoints/final/ppo_final/agent-{timestamp}")

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
