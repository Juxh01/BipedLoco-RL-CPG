from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import logging

import mujoco
import numpy as np
from myosuite.utils import gym

from source.agents.CPG import EF, JOINTS, MUSCLE_ORDER, SIDES, TwoLayerCPG

logger = logging.getLogger(__name__)

# Resolve a base wrapper class for runtime while keeping type-checkers calm
try:
    BaseWrapper = gym.Wrapper  # type: ignore[attr-defined]
except Exception:  # pragma: no cover

    class BaseWrapper:  # type: ignore
        def __init__(self, env: Any) -> None:
            self.env = env


# TODO: Check wether using tendon velocity as a proxy for Ia is valid. Actuator velocity refers to kartesian velocity
# TODO: Solving MTU Equilibrium to get muscle length and velocity for Ia calculation could be more accurate.
class CPGWrapper(BaseWrapper):
    """
    Gymnasium-compatible wrapper that:
    - Exposes a RL action space for CPG modulation (D, Gw, Gc, K_Ia, K_Ib).
    - Internally integrates a TwoLayerCPG and sends 26 muscle activations to the env.
    - Uses Ia = rectified tendon velocity (as proxy for the muscles velocity), Ib = actuator_force as feedback each step.
    - Sub-steps the CPG ODE to ensure stable integration.

    Observations: unchanged (pass-through).
    Actions: vector of size 42 by default (see _build_action_space()).

    Notes
    - Assumes underlying env follows Gymnasium API (reset() -> (obs, info); step() -> (obs, reward, terminated, truncated, info)).
    - Automatically maps CPG MUSCLE_ORDER -> env actuator order and vice versa using exact name matching.
    """

    def __init__(
        self,
        env: Any,
        *,
        cpg: Optional[TwoLayerCPG] = None,
        cpg_substeps: int = 10,
        dt_env: Optional[float] = None,
        ia_rectify: bool = True,
        ia_scale: float = 1.0,
        ib_scale: float = 1.0,
    ):
        super().__init__(env)

        # Disallow wrapping a vectorized environment; wrap single envs before vectorization
        if self._is_vectorized_env(env):
            raise TypeError(
                "CPGWrapper must wrap a single environment, not a vectorized VecEnv. Wrap each env inside the env_fn before creating SubprocVecEnv."
            )

        # Cache MuJoCo sim handle robustly
        self.sim = self._find_sim(env)

        # CPG state and timing
        self.cpg = cpg or TwoLayerCPG()
        self.cpg_substeps = int(max(1, cpg_substeps))
        self.dt_env = (
            float(dt_env)
            if dt_env is not None
            else float(getattr(env, "dt", 1.0 / 30.0))
        )
        self.ia_rectify = bool(ia_rectify)
        self.ia_scale = float(ia_scale)
        self.ib_scale = float(ib_scale)

        # Resolve actuator ordering and build reindex maps between env and CPG MUSCLE_ORDER (exact names only)
        self._act_names_env: List[str] = self._get_actuator_names()

        # Static mappings
        self._name_to_env_idx: Dict[str, int] = {
            n: i for i, n in enumerate(self._act_names_env)
        }
        self._name_to_cpg_idx: Dict[str, int] = {
            n: i for i, n in enumerate(MUSCLE_ORDER)
        }

        # env_idx for each CPG muscle (to reorder Ia/Ib into CPG order)
        self._env_idx_for_cpg: List[Optional[int]] = [
            self._name_to_env_idx.get(name, None) for name in MUSCLE_ORDER
        ]
        # cpg_idx for each env actuator (to reorder CPG output into env order)
        self._cpg_idx_for_env: List[Optional[int]] = [
            self._name_to_cpg_idx.get(name, None) for name in self._act_names_env
        ]

        # TODO: Refactor mapping check in a separate class helper function
        # Log mismatches if any (kept simple; no canonicalization)
        missing_in_env = [
            name
            for name, idx in zip(MUSCLE_ORDER, self._env_idx_for_cpg)
            if idx is None
        ]
        if missing_in_env:
            logger.warning(
                f"[CPGWrapper] CPG MUSCLE_ORDER not fully present in env actuators. Missing: {missing_in_env}"
            )
        missing_in_cpg = [
            name
            for name, idx in zip(self._act_names_env, self._cpg_idx_for_env)
            if idx is None
        ]
        if missing_in_cpg:
            logger.warning(
                f"[CPGWrapper] Env actuators not fully present in CPG MUSCLE_ORDER. Missing: {missing_in_cpg}"
            )

        # TODO: Wrap tendon velocity mapping in a separate function
        # Build indices into env's sensor observation for tendon velocities (strict)
        if not hasattr(self.env, "observation_sensor_keys"):
            raise AttributeError(
                "CPGWrapper: env.observation_sensor_keys not available."
            )
        sensor_keys: List[str] = list(getattr(self.env, "observation_sensor_keys"))
        sensor_key_to_idx: Dict[str, int] = {k: i for i, k in enumerate(sensor_keys)}
        self._vel_idx_for_cpg: List[int] = []
        missing_vel: List[str] = []
        for m in MUSCLE_ORDER:
            k = f"{m}_vel"
            if k in sensor_key_to_idx:
                self._vel_idx_for_cpg.append(sensor_key_to_idx[k])
            else:
                missing_vel.append(k)
        if missing_vel:
            raise KeyError(
                f"CPGWrapper: Missing required tendon-velocity sensor keys in env.observation_sensor_keys: {missing_vel}"
            )

        # TODO: Continue Validation from here
        # Precompute Fmax normalization (Ib) from actuator_gainprm[:, 2]
        # Guard for shapes and non-muscle actuators
        gainprm = np.asarray(self.sim.model.actuator_gainprm, dtype=np.float32)
        if (
            gainprm.ndim == 2
            and gainprm.shape[0] >= self.sim.model.nu
            and gainprm.shape[1] >= 3
        ):
            self._fmax_env = gainprm[: self.sim.model.nu, 2].copy()
        else:
            self._fmax_env = np.ones(self.sim.model.nu, dtype=np.float32)
        # Map to CPG muscle order (fallback epsilon to avoid divide-by-zero)
        eps = np.finfo(np.float32).eps
        self._fmax_cpg = np.ones(len(MUSCLE_ORDER), dtype=np.float32)
        for i_cpg, env_idx in enumerate(self._env_idx_for_cpg):
            if env_idx is not None and 0 <= env_idx < self._fmax_env.shape[0]:
                val = float(self._fmax_env[env_idx])
                self._fmax_cpg[i_cpg] = val if val > eps else 1.0
            else:
                self._fmax_cpg[i_cpg] = 1.0
        # Velocity normalization scale for Ia (m/s). If you have per-muscle vmax, replace this scalar.
        self._ia_vnorm = 1.0

        # Build RL action space for CPG modulation
        self._act_names_rl, low, high = self._build_action_space()
        self.action_space = gym.spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
            shape=(len(low),),
        )
        # Observations unchanged
        self.observation_space = env.observation_space

    # --------- Gym API ---------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None and hasattr(self.env, "reset"):
            try:
                obs, info = self.env.reset(seed=seed, options=options)
            except TypeError:
                # Back-compat if env.reset signature differs
                obs, info = self.env.reset()
        else:
            out = self.env.reset()
            # Normalize to gymnasium (obs, info)
            if isinstance(out, tuple) and len(out) == 2:
                obs, info = out
            else:
                obs, info = out, {}
        self.cpg.reset()
        return obs, info

    def step(self, action: np.ndarray):
        # 1) Update CPG gains from RL action vector
        action = np.asarray(action, dtype=np.float32).ravel()
        self._apply_rl_action_to_cpg(action)

        # 2) Read Ia/Ib from MuJoCo aligned to CPG MUSCLE_ORDER
        Ia_cpg, Ib_cpg = self._get_Ia_Ib_cpg_order()

        # 3) Sub-step the CPG ODE
        sub_dt = self.dt_env / float(self.cpg_substeps)
        for _ in range(self.cpg_substeps):
            self.cpg.step(
                dt_env=sub_dt,
                n_substeps=1,
                Ia_muscle=Ia_cpg,
                Ib_muscle=Ib_cpg,
            )

        # 4) Map CPG output activations to env actuator order and step env
        act_cpg = self.cpg.cpg_to_muscle_activation()  # shape (26,) in MUSCLE_ORDER
        act_env = self._map_cpg_to_env_action(act_cpg)
        step_out = self.env.step(act_env)
        # Normalize to Gymnasium 5-tuple
        if len(step_out) == 4:
            obs, reward, done, info = step_out
            terminated, truncated = (done, False) if isinstance(done, bool) else done
            return obs, reward, terminated, truncated, info
        else:
            obs, reward, terminated, truncated, info = step_out
            return obs, reward, terminated, truncated, info

    # --------- Internal helpers ---------
    def _get_actuator_names(self) -> List[str]:
        # Read actuator names from the Gym env's sim wrapper
        if not hasattr(self, "sim") or not hasattr(self.sim, "model"):
            raise AttributeError(
                "CPGWrapper: env.sim.model not available to fetch actuator names."
            )
        model = self.sim.model
        if not hasattr(model, "actuator") or not hasattr(model, "nu"):
            raise AttributeError(
                "CPGWrapper: sim.model does not expose actuator() or nu to fetch actuator names."
            )
        names: List[str] = [model.actuator(i).name for i in range(int(model.nu))]
        if any((n is None) or (str(n).strip() == "") for n in names):
            raise ValueError(
                f"CPGWrapper: Encountered empty actuator names from env (names={names})."
            )
        return [str(n) for n in names]

    def _get_tendon_name_to_id(self) -> Dict[str, int]:
        model = self.sim.model
        out: Dict[str, int] = {}
        try:
            obj_type = int(mujoco.mjtObj.mjOBJ_TENDON)  # type: ignore[attr-defined]
        except Exception:
            obj_type = 21  # best-effort fallback id for tendons
        for i in range(getattr(model, "ntendon", 0)):
            try:
                if hasattr(model, "id2name"):
                    n = model.id2name(obj_type, i)
                else:
                    n = mujoco.mj_id2name(model, obj_type, i)  # type: ignore[arg-type]
                if n:
                    out[n] = int(i)
            except Exception:
                continue
        return out

    def _build_action_space(self) -> Tuple[List[str], List[float], List[float]]:
        """
        RL action vector layout (size = 42):
        - D (descending drive) per side, joint, EF: 2*3*2 = 12 in [0, 5]
        - Gw couplings: Gw_lr_hip, Gw_H_to_K, Gw_K_to_A: 3 in [0, 3]
        - Gc couplings per joint: Gc_hip, Gc_knee, Gc_ankle: 3 in [0, 3]
        - K_Ia per side, joint, EF: 12 in [-3, 3]
        - K_Ib per side, joint, EF: 12 in [-3, 3]
        """
        names: List[str] = []
        low: List[float] = []
        high: List[float] = []

        # D
        for side in SIDES:  # r, l
            for joint in JOINTS:  # hip, knee, ankle
                for ef in EF:  # E, F
                    names.append(f"D_{joint}_{ef}_{side}")
                    low.append(0.0)
                    high.append(5.0)

        # Gw
        for k in ("Gw_lr_hip", "Gw_H_to_K", "Gw_K_to_A"):
            names.append(k)
            low.append(0.0)
            high.append(3.0)

        # Gc
        for k in ("Gc_hip", "Gc_knee", "Gc_ankle"):
            names.append(k)
            low.append(0.0)
            high.append(3.0)

        # K_Ia
        for side in SIDES:
            for joint in JOINTS:
                for ef in EF:
                    names.append(f"K_Ia_{joint}_{ef}_{side}")
                    low.append(-3.0)
                    high.append(3.0)

        # K_Ib
        for side in SIDES:
            for joint in JOINTS:
                for ef in EF:
                    names.append(f"K_Ib_{joint}_{ef}_{side}")
                    low.append(-3.0)
                    high.append(3.0)

        return names, low, high

    def _apply_rl_action_to_cpg(self, a: np.ndarray):
        """
        Decode RL action vector (as defined in _build_action_space) and update CPG gains.
        """
        assert a.shape[0] == len(self._act_names_rl), (
            f"Expected action length {len(self._act_names_rl)}, got {a.shape[0]}"
        )
        payload: Dict[str, float] = {k: float(v) for k, v in zip(self._act_names_rl, a)}
        self.cpg.apply_action(payload)

    def _get_Ia_Ib_cpg_order(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read env signals and reorder to MUSCLE_ORDER for the CPG.
        - Ia = rectified tendon velocity (m/s) from env observations, scaled by ia_scale. STRICT: all keys must exist.
        - Ib = actuator_force (N), scaled by ib_scale.
        Returns arrays of shape (26,) matching MUSCLE_ORDER. Missing entries -> 0.
        """
        # Tendon velocities for Ia from environment observation dict
        if not hasattr(self.env, "get_obs_dict"):
            raise AttributeError(
                "CPGWrapper: env.get_obs_dict(sim) not available to fetch sensor observations."
            )
        obs_dict = self.env.get_obs_dict(self.sim)
        if "sensor" not in obs_dict:
            raise KeyError("CPGWrapper: 'sensor' key missing in env observation dict.")
        sensor_arr = np.asarray(obs_dict["sensor"], dtype=np.float32)
        f_env = np.asarray(self.sim.data.actuator_force, dtype=np.float32)  # (nu,)

        Ia = np.zeros(len(MUSCLE_ORDER), dtype=np.float32)
        Ib = np.zeros(len(MUSCLE_ORDER), dtype=np.float32)
        vnorm = (
            float(self._ia_vnorm)
            if getattr(self, "_ia_vnorm", None) is not None
            else 1.0
        )
        vnorm = vnorm if vnorm > 0 else 1.0
        for i_cpg, env_idx in enumerate(self._env_idx_for_cpg):
            # Ia from sensor observations using precomputed indices
            idx = self._vel_idx_for_cpg[i_cpg]
            if idx >= sensor_arr.shape[0]:
                raise IndexError(
                    f"CPGWrapper: Sensor index {idx} out of range for muscle '{MUSCLE_ORDER[i_cpg]}'."
                )
            v = float(sensor_arr[idx])
            if self.ia_rectify:
                v = max(0.0, v)
            Ia[i_cpg] = self.ia_scale * (v / vnorm)
            # Ib from actuator force if available; if actuator mapping missing, treat as 0
            if env_idx is None or env_idx >= f_env.shape[0]:
                Ib[i_cpg] = 0.0
            else:
                denom = (
                    self._fmax_cpg[i_cpg] if i_cpg < self._fmax_cpg.shape[0] else 1.0
                )
                Ib[i_cpg] = self.ib_scale * (
                    f_env[env_idx] / (denom if denom > 0 else 1.0)
                )
        # Bound signals to [0, 1] for stability
        Ia = np.clip(Ia, 0.0, 1.0)
        Ib = np.clip(Ib, 0.0, 1.0)
        return Ia, Ib

    def _map_cpg_to_env_action(self, act_cpg: np.ndarray) -> np.ndarray:
        """
        Reorder CPG activation vector (MUSCLE_ORDER) to env actuator order.
        Missing mappings -> 0.0.
        """
        nu = len(self._act_names_env)
        out = np.zeros(nu, dtype=np.float32)
        for env_idx, cpg_idx in enumerate(self._cpg_idx_for_env):
            if cpg_idx is None or cpg_idx >= act_cpg.shape[0]:
                out[env_idx] = 0.0
            else:
                out[env_idx] = float(act_cpg[cpg_idx])
        return out

    # --------- Convenience ---------
    @property
    def rl_action_names(self) -> List[str]:
        return list(self._act_names_rl)

    # --------- Introspection utilities ---------
    def _is_vectorized_env(self, env) -> bool:
        # SB3 VecEnv check
        try:
            from stable_baselines3.common.vec_env.base_vec_env import (
                VecEnv,  # type: ignore
            )

            if isinstance(env, VecEnv):
                return True
        except Exception:
            pass
        # Heuristics
        if hasattr(env, "envs") and isinstance(getattr(env, "envs"), (list, tuple)):
            return True
        if hasattr(env, "num_envs"):
            try:
                if int(getattr(env, "num_envs")) > 1:
                    return True
            except Exception:
                pass
        return False

    # Locate mujoco.MjSim.sim handle from env by walking wrappers (for multiple layers)
    def _find_sim(self, env):
        # direct access
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "sim"):
            return env.unwrapped.sim
        # walk wrappers
        cur = env
        for _ in range(32):
            if hasattr(cur, "unwrapped") and hasattr(cur.unwrapped, "sim"):
                return cur.unwrapped.sim
            if hasattr(cur, "env"):
                cur = cur.env
                continue
            break
        # single-env vector containers
        if (
            hasattr(env, "envs")
            and isinstance(env.envs, (list, tuple))
            and len(env.envs) == 1
        ):
            inner = env.envs[0]
            if hasattr(inner, "unwrapped") and hasattr(inner.unwrapped, "sim"):
                return inner.unwrapped.sim
        raise AttributeError("Could not locate MuJoCo sim on the provided env.")
