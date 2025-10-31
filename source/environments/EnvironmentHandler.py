from __future__ import annotations

from typing import Any, Callable, Dict, get_args, get_origin

import copy
import os
import random
import subprocess
from dataclasses import fields

import numpy as np
import torch

try:
    import source.environments.myoassist.rl_train.envs as _ensure_env_registration  # noqa: F401
except Exception as e:
    print(f"[EnvRegistry] Import of myoassist-Registry failed: {e} ")

from myosuite.utils import gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from source.environments.myoassist.rl_train.train.train_configs.config import (
    TrainSessionConfigBase,
)
from source.environments.myoassist.rl_train.train.train_configs.config_imiatation_exo import (
    ExoImitationTrainSessionConfig,
)
from source.environments.myoassist.rl_train.train.train_configs.config_imitation import (
    ImitationTrainSessionConfig,
)


def _require(d: Dict[str, Any], key: str, ctx: str = ""):
    if key not in d:
        raise KeyError(f"Missing required config key '{key}' in {ctx or 'config'}")


def _dataclass_from_dict(dc_type, data):
    """Recursive mapping: dict -> dataclass instance (supports Optional/Union/List)."""
    if data is None:
        return None
    # Fast path: if target is not a dataclass type, handle generics or return raw data
    if not hasattr(dc_type, "__dataclass_fields__"):
        origin = get_origin(dc_type)
        # Handle List[T]
        if origin is list:
            (inner,) = get_args(dc_type) or (Any,)
            return [_dataclass_from_dict(inner, v) for v in (data or [])]
        # For Dict[K, V] or other non-dataclass generics, return as-is
        return data

    kwargs = {}
    for f in fields(dc_type):
        name = f.name
        typ = f.type
        if name not in data:
            continue
        val = data[name]

        origin = get_origin(typ)
        args = get_args(typ)
        target_typ = typ

        # If type is Optional[T] or Union[..., None], pick the most appropriate inner type
        if origin is not None and args:
            # Prefer the first dataclass type inside the Union, otherwise the first non-None type
            dc_arg = next((a for a in args if hasattr(a, "__dataclass_fields__")), None)
            if dc_arg is not None:
                target_typ = dc_arg
            else:
                non_none = [a for a in args if a is not type(None)]  # noqa: E721
                target_typ = non_none[0] if non_none else typ

        # Recurse if the target type is a dataclass
        if hasattr(target_typ, "__dataclass_fields__"):
            kwargs[name] = _dataclass_from_dict(target_typ, val)
        else:
            # If it's a List[...] handle elements recursively
            if get_origin(target_typ) is list:
                (inner,) = get_args(target_typ) or (Any,)
                kwargs[name] = [_dataclass_from_dict(inner, v) for v in (val or [])]
            else:
                # Primitive or unsupported generic: assign directly
                kwargs[name] = val

    return dc_type(**kwargs)


def _select_env_params_type(env_id: str):
    """Choose the appropriate EnvParams dataclass based on env_id."""
    if env_id.startswith("myoAssistLegImitationExo"):
        if ExoImitationTrainSessionConfig is None:
            raise ImportError("ExoImitationTrainSessionConfig not available")
        return ExoImitationTrainSessionConfig.EnvParams
    if env_id.startswith("myoAssistLegImitation"):
        if ImitationTrainSessionConfig is None:
            raise ImportError("ImitationTrainSessionConfig not available")
        return ImitationTrainSessionConfig.EnvParams
    return TrainSessionConfigBase.EnvParams


def _make_single_env(env_cfg: Dict[str, Any]):
    """Creates a single Gym environment based on the provided config."""
    _require(env_cfg, "env_id", "env")
    _require(env_cfg, "seed", "env")
    _require(env_cfg, "model_path", "env")

    env_id = str(env_cfg["env_id"])
    _seed_random_generators(int(env_cfg["seed"]))

    # Dict -> Dataclass (EnvParams)
    env_params_dict = env_cfg.get("env_params", {}) or {}
    EnvParamsType = _select_env_params_type(env_id)
    env_params_obj = _dataclass_from_dict(EnvParamsType, env_params_dict)

    gym_make_args = {
        "seed": int(env_cfg["seed"]),
        "model_path": env_cfg["model_path"],
        "env_params": env_params_obj,
        "is_evaluate_mode": bool(env_cfg.get("is_evaluate_mode", False)),
    }

    env = gym.make(env_id, **gym_make_args).unwrapped

    if bool(env_cfg.get("render", False)):
        setattr(env, "mujoco_render_frames", True)

    log_dir = env_cfg.get("log_dir")
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))

    return env


def _seed_random_generators(seed: int):
    """Seed random, numpy and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_vec_env(env_cfg: Dict[str, Any], num_envs: int):
    """SubprocVecEnv with individual seeds per env and improved reproducibility."""
    base_seed = int(env_cfg.get("seed", 0))

    def make_thunk(rank: int):
        def _thunk():
            seed = base_seed + rank + 1000  # Offset to avoid overlap

            # Prepare env cfg and create env
            cfg = copy.deepcopy(env_cfg)
            cfg["seed"] = seed
            if cfg.get("render"):
                cfg["render"] = False
            if cfg.get("log_dir"):
                cfg["log_dir"] = False
            env = _make_single_env(cfg)

            if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
                env.action_space.seed(seed)
            if hasattr(env, "observation_space") and hasattr(
                env.observation_space, "seed"
            ):
                env.observation_space.seed(seed)
            return env

        return _thunk

    # Use spawn to avoid inheriting parent RNG state (more reproducible across runs)
    return SubprocVecEnv([make_thunk(i) for i in range(num_envs)], start_method="spawn")


# Einfache Algo-Registry. Du kannst hier weitere Ansätze hinterlegen.
RL_ALG_REGISTRY: Dict[str, Callable[[Dict[str, Any], Any], Any]] = {
    # Stable-Baselines3 PPO
    "ppo": lambda algo_cfg, env: PPO(
        algo_cfg.get("policy", "MlpPolicy"),
        env,
        verbose=algo_cfg.get("verbose", 1),
        tensorboard_log=algo_cfg.get("tensorboard_log"),
        **algo_cfg.get("kwargs", {}),
    ),
    # Stable-Baselines3 SAC (Beispiel)
    "sac": lambda algo_cfg, env: SAC(
        algo_cfg.get("policy", "MlpPolicy"),
        env,
        verbose=algo_cfg.get("verbose", 1),
        tensorboard_log=algo_cfg.get("tensorboard_log"),
        **algo_cfg.get("kwargs", {}),
    ),
    # Platzhalter für eigene Implementierungen
    "none": lambda algo_cfg, env: None,
}


class EnvironmentHandler:
    @staticmethod
    def create_environment(config: Dict[str, Any]):
        """
        Erzeugt ein Gym Env (single oder vectorized) anhand eines Config-Dicts.
        Erwartete Struktur:
        config = {
          "env": {
            "env_id": "myoAssistLeg-v0",
            "model_path": "...xml",
            "seed": 0,
            "num_envs": 1,
            "render": False,
            "is_evaluate_mode": False,
            "log_dir": "runs/exp1",
            "env_params": {
              # Beispiele (MyoAssistLegBase._setup erwartet diese Felder):
              "physics_sim_framerate": 1000,
              "control_framerate": 50,
              "safe_height": 0.6,
              "min_target_velocity": 0.0,
              "max_target_velocity": 1.5,
              "min_target_velocity_period": 1.0,
              "max_target_velocity_period": 3.0,
              "custom_max_episode_steps": 2000,
              "enable_lumbar_joint": False,
              "lumbar_joint_fixed_angle": 0.0,
              "lumbar_joint_damping_value": 5.0,
              "observation_joint_pos_keys": [...],
              "observation_joint_vel_keys": [...],
              "observation_sensor_keys": [...],
              "joint_limit_sensor_keys": [...],
              "terrain_type": "flat",        # oder: "random", "harmonic_sinusoidal", "slope"
              "terrain_params": "",          # string-encoded Parameter (siehe HfieldManager)
              "reward_keys_and_weights": {...},
            },
          }
        }
        """
        _require(config, "env")
        env_cfg = config["env"]
        num_envs = int(env_cfg.get("num_envs", 1))

        git_root = (
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        model_path = os.path.join(git_root, "source", "environments", "myoassist")
        model_path = os.path.join(model_path, env_cfg.get("model_path", ""))
        env_cfg["model_path"] = model_path

        if num_envs <= 1 or bool(env_cfg.get("render", False)):
            return _make_single_env(env_cfg)
        return _make_vec_env(env_cfg, num_envs=num_envs)

    @staticmethod
    def create_rl_model(config: Dict[str, Any], env):
        """
        Wählt und instanziiert den RL-Algorithmus anhand eines Algo-Blocks.
        Erwartete Struktur:
        config = {
          "algo": {
            "name": "ppo",                  # "ppo" | "sac" | "none" | eigener key
            "policy": "MlpPolicy",          # SB3-Policy-String
            "verbose": 1,
            "tensorboard_log": "runs/tb",
            "kwargs": {
              "learning_rate": 3e-4,
              "n_steps": 2048,
              "batch_size": 64,
              "gamma": 0.99,
              # weitere SB3-Parameter ...
            }
          }
        }
        """
        _require(config, "algo")
        algo_cfg: Dict[str, Any] = config["algo"]
        _require(algo_cfg, "name", "algo")

        name = str(algo_cfg["name"]).lower()
        if name not in RL_ALG_REGISTRY:
            raise ValueError(
                f"Unknown algorithm '{name}'. Available: {list(RL_ALG_REGISTRY.keys())}"
            )
        return RL_ALG_REGISTRY[name](algo_cfg, env)
