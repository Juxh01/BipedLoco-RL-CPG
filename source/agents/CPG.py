# This implements a two-layer CPG (Rhythm Generator + Pattern Formation)
# The implementation is based on Deng et al. 2022, but uses a half-center oscillator model for the rhythm generator.
# The code was created with the help of ChatGPT-5, but has been thoroughly reviewed and modified.
# The planmaking was done by Gemini Deep Research based on Cheng et al. 2022 and our own ideas


from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    return expit(z)


def _sigmoid_affine(x: np.ndarray, x0: float, k: float) -> np.ndarray:
    eps = np.finfo(float).eps
    # Avoid division by zero
    denom = k if k != 0 else eps
    z = (x - x0) / denom
    return _stable_sigmoid(z)


# ============ Muscle ordering (matches MyoLeg26 action order) ============
MUSCLE_ORDER: List[str] = [
    "abd_r",
    "add_r",
    "hamstrings_r",
    "bifemsh_r",
    "edl_r",
    "fdl_r",
    "glutmax_r",
    "iliopsoas_r",
    "rectfem_r",
    "vasti_r",
    "gastroc_r",
    "soleus_r",
    "tibant_r",
    "abd_l",
    "add_l",
    "hamstrings_l",
    "bifemsh_l",
    "edl_l",
    "fdl_l",
    "glutmax_l",
    "iliopsoas_l",
    "rectfem_l",
    "vasti_l",
    "gastroc_l",
    "soleus_l",
    "tibant_l",
]

# Joint/module sets and sides
JOINTS = ("hip", "knee", "ankle")
SIDES = ("r", "l")
EF = ("E", "F")  # Extensor, Flexor

# ----------- Anatomical mapping from muscles to joint modules ---------------
# Map monoarticular muscles to PF module (joint, Extensor/Flexor)
MONO_MAP: Dict[str, Tuple[str, str]] = {
    # Hip mono
    "glutmax": ("hip", "E"),
    "iliopsoas": ("hip", "F"),
    "abd": (
        "hip",
        "F",
    ),  # abductors -> support flexor synergy
    "add": ("hip", "E"),  # adductors -> support extensor synergy
    # Knee mono
    "vasti": ("knee", "E"),
    "bifemsh": ("knee", "F"),
    # Ankle mono
    "soleus": ("ankle", "E"),
    "fdl": ("ankle", "E"),
    "tibant": ("ankle", "F"),
    "edl": ("ankle", "F"),
}

# Biarticular mapping (weights for PF sources per muscle)
# rectfem: Hip-F + Knee-E; hamstrings: Hip-E + Knee-F; gastroc: Knee-F + Ankle-E
# TODO: Offline Tuning of Biarticular Weights
BIART_WEIGHTS = {
    "rectfem": {("hip", "F"): 0.5, ("knee", "E"): 0.5},
    "hamstrings": {("hip", "E"): 0.5, ("knee", "F"): 0.5},
    "gastroc": {("knee", "F"): 0.5, ("ankle", "E"): 0.5},
}


# ============ Neuron model (non-spiking -> continuous; parametric) ============


@dataclass
class NeuronParams:
    Cm: float = 1.0  # membrane capacitance
    Gm: float = 1.0  # membrane conductance
    Er: float = -60.0  # resting potential
    ENa: float = 50.0  # sodium reversal potential
    GNa: float = 1.5  # sodium conductance

    # steady-state activation/inactivation (sigmoids)
    V_half_m: float = -40.0  # activation
    k_m: float = 5.0  # activation slope
    V_half_h: float = -45.0  # inactivation
    k_h: float = -5.0  # inactivation slope

    # tau_h shaping (sigmoid blend) for inactivation
    tau_h_min: float = 5.0  # minimum inactivation time constant
    tau_h_max: float = 80.0  # maximum inactivation time constant
    V_tau: float = -50.0  # tau_h sigmoid inflection point
    k_tau: float = 5.0  # tau_h sigmoid slope

    # synaptic output shaping
    V_th_syn: float = -40.0  # synaptic output threshold
    k_syn: float = 4.0  # synaptic output slope

    def m_inf(self, V: np.ndarray) -> np.ndarray:
        return _sigmoid_affine(V, self.V_half_m, self.k_m)

    def h_inf(self, V: np.ndarray) -> np.ndarray:
        return _sigmoid_affine(V, self.V_half_h, self.k_h)

    def tau_h(self, V: np.ndarray) -> np.ndarray:
        s = _sigmoid_affine(V, self.V_tau, self.k_tau)
        return self.tau_h_min + (self.tau_h_max - self.tau_h_min) * s

    def syn_out(self, V: np.ndarray) -> np.ndarray:
        return _sigmoid_affine(V, self.V_th_syn, self.k_syn)


# ---- These Parameters are dynamically modulated by the CPG agent's actions ----------------
@dataclass
class CPGGains:
    # Descending drive to RG E/F per joint and side: shape (2 sides, 3 joints, 2 EF)
    D: np.ndarray = field(default_factory=lambda: np.zeros((2, 3, 2), dtype=np.float32))

    # Inter-RG couplings (Ijspeert-style): inhibit across sides at hip; weak hierarchical within limb
    Gw_lr_hip: float = 1.0  # L↔R anti-phase at hip RGs (mutual inhibition)
    Gw_H_to_K: float = 0.2  # RG hip -> RG knee (same side)
    Gw_K_to_A: float = 0.2  # RG knee -> RG ankle (same side)

    # RG (Rhythm Generator) -> PF (Pattern Formation) feedforward (Rybak): per joint
    Gc_hip: float = 1.0
    Gc_knee: float = 1.0
    Gc_ankle: float = 1.0

    # Sensory gains per joint, side and pathway (Ia to RG, Ib to PF)
    # Gain for muscle stretch Ia feedback to RG
    K_Ia: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 3, 2), dtype=np.float32)
    )  # (side,joint,targets RG[E/F])
    # Gain for Golgi tendon Ib feedback to PF
    K_Ib: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 3, 2), dtype=np.float32)
    )  # (side,joint,targets PF[E/F])


@dataclass
class CPGConfig:
    neuron: NeuronParams = field(default_factory=NeuronParams)

    # Reversal potentials for synapses
    E_exc: float = 0.0  # Excitatory potentials for synapses
    E_inh: float = -70.0  # Inhibitory potentials for synapses

    # Intrinsic mutual inhibition inside RG half-center (E<->F) and PF (optional)
    RG_mutual_inh: float = 1.2  # Strength of mutual inhibition within RG half-centers
    PF_mutual_inh: float = 0.8  # Strength of mutual inhibition within PF

    # Output squashing (PF to muscle activation)
    # TODO: 0.1 and 0.9 ?
    out_min: float = 0.0
    out_max: float = 1.0


class TwoLayerCPG:
    """
    Two-layer CPG (RG -> PF) with bilateral hip/knee/ankle modules (E/F per module).
    - State: V and h for all neurons, vectorized.
    - Dynamics: Deng-style non-spiking neuron, fixed-step RK4.
    - Modulation: D, Gw, Gc, K_Ia, K_Ib.
    - Output: 26 muscle activations (MyoLeg26 order).
    """

    def __init__(self, cfg: Optional[CPGConfig] = None):
        self.cfg = cfg or CPGConfig()
        self.gains = CPGGains()
        # Indexing of neurons: layer x side x joint x E/F
        # layer: 0 = RG, 1 = PF
        self.n_layers, self.n_sides, self.n_joints, self.n_ef = 2, 2, 3, 2
        self.N = self.n_layers * self.n_sides * self.n_joints * self.n_ef

        # State vectors
        self.V = np.full(self.N, -55.0, dtype=np.float32)  # Initial membrane potentials
        self.h = self.cfg.neuron.h_inf(self.V).astype(
            np.float32
        )  # Initial activation variables
        self.t = 0.0

        # Precompute indices
        self._idx = self._build_index()

    # ---------- Indexing helpers ----------
    def _build_index(self) -> Dict[Tuple[int, int, int, int], int]:
        idx = {}
        k = 0
        for L in range(self.n_layers):
            for s in range(self.n_sides):
                for j in range(self.n_joints):
                    for ef in range(self.n_ef):
                        idx[(L, s, j, ef)] = k
                        k += 1
        return idx

    @staticmethod
    def _side_idx(side: str) -> int:
        return 0 if side == "r" else 1

    @staticmethod
    def _joint_idx(joint: str) -> int:
        return {"hip": 0, "knee": 1, "ankle": 2}[joint]

    @staticmethod
    def _ef_idx(ef: str) -> int:
        return 0 if ef == "E" else 1

    def _id(self, L: int, side: int, joint: int, ef: int) -> int:
        return self._idx[(L, side, joint, ef)]

    # ---------- Public API ----------
    def reset(self, V0: float = -55.0):
        self.V[:] = V0
        self.h[:] = self.cfg.neuron.h_inf(self.V)
        self.t = 0.0

    # TODO: Make this more efficient
    def apply_action(self, action: Dict[str, float]):
        """
        Update modulatory gains from an action dict. Map keys to indices in numpy arrays:
          - D_hip_E_r, D_hip_F_r, ..., D_ankle_F_l
          - Gw_lr_hip, Gw_H_to_K, Gw_K_to_A
          - Gc_hip, Gc_knee, Gc_ankle
          - K_Ia_<joint>_<E|F>_<side>
          - K_Ib_<joint>_<E|F>_<side>
        """
        # Descending drive
        for side in SIDES:
            s = self._side_idx(side)
            for joint in JOINTS:
                j = self._joint_idx(joint)
                for ef in EF:
                    e = self._ef_idx(ef)
                    key = f"D_{joint}_{ef}_{side}"
                    if key in action:
                        self.gains.D[s, j, e] = float(action[key])

        # Couplings
        for k in (
            "Gw_lr_hip",
            "Gw_H_to_K",
            "Gw_K_to_A",
            "Gc_hip",
            "Gc_knee",
            "Gc_ankle",
        ):
            if k in action:
                setattr(self.gains, k, float(action[k]))

        # Feedback gains
        for path, store in (("K_Ia", "K_Ia"), ("K_Ib", "K_Ib")):
            tgt = getattr(self.gains, store)
            for side in SIDES:
                s = self._side_idx(side)
                for joint in JOINTS:
                    j = self._joint_idx(joint)
                    for ef in EF:
                        e = self._ef_idx(ef)
                        key = f"{path}_{joint}_{ef}_{side}"
                        if key in action:
                            tgt[s, j, e] = float(action[key])

    def step(
        self,
        dt_env: float,
        n_substeps: int,
        Ia_muscle: Optional[np.ndarray] = None,
        Ib_muscle: Optional[np.ndarray] = None,
    ):
        """
        Integrate the CPG by n_substeps of dt_cpg = dt_env/n_substeps.
        Ia_muscle, Ib_muscle: optional arrays (26,) aligned to MUSCLE_ORDER.
        """
        dt = float(dt_env) / max(1, int(n_substeps))
        for _ in range(max(1, int(n_substeps))):
            # Collect input currents (signals)
            I_app_RG, I_app_PF = self._assemble_inputs(Ia_muscle, Ib_muscle)
            # Solve next ODE step with RK4 (fixed step to ensure training timing)
            self._rk4_step(dt, I_app_RG, I_app_PF)
            self.t += dt
        return self.cpg_to_muscle_activation()

    # ---------- Internals: inputs and integration ----------
    def _assemble_inputs(
        self, Ia_muscle: Optional[np.ndarray], Ib_muscle: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build I_app to RG and PF layers from:
          - Descending drive D (per side/joint/E-F)
          - Ia feedback to RG
          - Ib feedback to PF
        Ia/Ib muscle signals (26,) are pooled per (side,joint,E/F) using anatomical mapping defined in this file
        """
        d = self.gains.D  # (2,3,2)
        Ia_rg = np.zeros_like(d)
        Ib_pf = np.zeros_like(d)

        # Apply Muscle Feedbacks
        if Ia_muscle is not None:
            Ia_rg += self._pool_feedback_to_modules(Ia_muscle) * self.gains.K_Ia
        if Ib_muscle is not None:
            Ib_pf += self._pool_feedback_to_modules(Ib_muscle) * self.gains.K_Ib

        # Final I_app
        I_app_RG = (d + Ia_rg).reshape(-1)  # (2*3*2,)
        I_app_PF = (Ib_pf).reshape(-1)
        return I_app_RG, I_app_PF

    # TODO: Make this more efficient
    def _pool_feedback_to_modules(self, muscle_vals: np.ndarray) -> np.ndarray:
        """
        Aggregate 26 muscle-level signals into (side, joint, E/F).
        Biarticular: split by BIART_WEIGHTS; mono: direct to module from MONO_MAP.
        Returns array (2,3,2).
        """
        out = np.zeros((2, 3, 2), dtype=np.float32)
        for idx, name in enumerate(MUSCLE_ORDER):
            val = float(muscle_vals[idx])
            # Get side and base name
            base, side = name.rsplit("_", 1)
            s = self._side_idx(side)

            # Biarticular
            if base in BIART_WEIGHTS:
                for (j_name, ef_name), w in BIART_WEIGHTS[base].items():
                    j = self._joint_idx(j_name)
                    e = self._ef_idx(ef_name)
                    out[s, j, e] += w * val
                continue

            # Monoarticular
            if base not in MONO_MAP:
                continue
            j_name, ef_name = MONO_MAP[base]
            j = self._joint_idx(j_name)
            e = self._ef_idx(ef_name)
            out[s, j, e] += val

        return out

    def _rk4_step(self, dt: float, I_app_RG: np.ndarray, I_app_PF: np.ndarray):
        """
        One RK4 step for all neurons. I_app vectors are per (side,joint,ef) and must be
        expanded to the layer indexing.
        I_app: External input currents to RG and PF layers
        I_syn: Synaptic currents from other neurons
        """
        n = self.cfg.neuron

        # Build per-neuron I_app (concatenate RG then PF order)
        I_rg = I_app_RG.astype(np.float32)
        I_pf = I_app_PF.astype(np.float32)
        I_app = np.concatenate([I_rg, I_pf], axis=0)  # shape (N,)

        def f(V, h):
            # Ionic + leak + synaptic + I_app
            m = n.m_inf(V)
            h_inf = n.h_inf(V)
            tau_h = n.tau_h(V)
            s_out = n.syn_out(V)  # presynaptic output

            # Synaptic currents
            I_syn = self._synaptic_current(s_out, V)

            eps = (
                np.finfo(tau_h.dtype).eps
                if hasattr(tau_h, "dtype")
                else np.finfo(float).eps
            )
            tau_safe = np.clip(tau_h, a_min=eps, a_max=None)

            # Differential equation for V and h of Neurons, solved by ODE-Solver (from Deng et al. 2022)
            dV = (
                n.Gm * (n.Er - V) + n.GNa * (n.ENa - V) * m * h + I_syn + I_app
            ) / n.Cm
            dh = (h_inf - h) / tau_safe
            return dV, dh

        # RK4 steps (less overhead than scipy.integrate for small systems)
        V, h = self.V, self.h
        k1_V, k1_h = f(V, h)
        k2_V, k2_h = f(V + 0.5 * dt * k1_V, h + 0.5 * dt * k1_h)
        k3_V, k3_h = f(V + 0.5 * dt * k2_V, h + 0.5 * dt * k2_h)
        k4_V, k4_h = f(V + dt * k3_V, h + dt * k3_h)

        self.V = V + (dt / 6.0) * (k1_V + 2 * k2_V + 2 * k3_V + k4_V)
        self.h = h + (dt / 6.0) * (k1_h + 2 * k2_h + 2 * k3_h + k4_h)

    # TODO: Vectorize this function
    def _synaptic_current(self, s_out: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Build synaptic currents using:
          - mutual inhibition within RG half-centers and PF half-centers
          - inter-RG couplings (hip L<->R; hip->knee; knee->ankle)
          - RG -> PF coupling (excitatory) per joint
        I_syn = sum g_ij * s_j * (E_syn - V_i)
        """
        N = self.N
        E_exc, E_inh = self.cfg.E_exc, self.cfg.E_inh
        I_syn = np.zeros(N, dtype=np.float32)

        # Helper to add current: i receives from j with gain g and E_rev
        def add_syn(i, j, g, E_rev):
            if g == 0.0:
                return
            I_syn[i] += (
                g * s_out[j] * (E_rev - V[i])
            )  # synaptic current based on potential difference, gain between the neurons and presynaptic output

        # Indices
        for s in range(self.n_sides):
            s_op = 1 - s  # opposite side
            # -------- Mutual inhibition within half-centers -------
            for j in range(self.n_joints):
                # RG layer
                iE = self._id(0, s, j, 0)
                iF = self._id(0, s, j, 1)
                add_syn(iE, iF, self.cfg.RG_mutual_inh, E_inh)
                add_syn(iF, iE, self.cfg.RG_mutual_inh, E_inh)
                # PF layer (optional mutual inhibition)
                # TODO: Necessary to have mutual inhibition in PF layer?
                pE = self._id(1, s, j, 0)
                pF = self._id(1, s, j, 1)
                add_syn(pE, pF, self.cfg.PF_mutual_inh, E_inh)
                add_syn(pF, pE, self.cfg.PF_mutual_inh, E_inh)

            # TODO: Check necessity of inter-RG couplings
            # -------- Inter-RG lateral inhibition at hip for anti-phase (Ijspeert-style)
            j_hip = 0
            for ef in range(self.n_ef):
                i_self = self._id(0, s, j_hip, ef)
                i_other = self._id(0, s_op, j_hip, ef)
                add_syn(i_self, i_other, self.gains.Gw_lr_hip, E_inh)

            # -------- Hierarchical RG coupling within limb: hip -> knee -> ankle
            # Use excitatory from E to E and F to F (promotes phasing), but can be tuned.
            for ef in range(self.n_ef):
                # hip -> knee
                add_syn(
                    self._id(0, s, 1, ef),
                    self._id(0, s, 0, ef),
                    self.gains.Gw_H_to_K,
                    E_exc,
                )
                # knee -> ankle
                add_syn(
                    self._id(0, s, 2, ef),
                    self._id(0, s, 1, ef),
                    self.gains.Gw_K_to_A,
                    E_exc,
                )

            # RG -> PF feedforward per joint (Rybak)
            # Get gains per joint
            Gc = [self.gains.Gc_hip, self.gains.Gc_knee, self.gains.Gc_ankle]
            for j in range(self.n_joints):
                for ef in range(self.n_ef):
                    add_syn(self._id(1, s, j, ef), self._id(0, s, j, ef), Gc[j], E_exc)

        return I_syn

    # ---------- Output mapping ----------
    # TODO: Make this more efficient
    def cpg_to_muscle_activation(self) -> np.ndarray:
        """
        Map PF activities to 26 muscle activations in [0,1] using anatomical grouping and
        biarticular mixing.
        """
        n = self.cfg.neuron
        # PF layer outputs
        pf_out = np.zeros((2, 3, 2), dtype=np.float32)
        for s, side in enumerate(SIDES):
            for j, joint in enumerate(JOINTS):
                for e, ef in enumerate(EF):
                    idx = self._id(1, s, j, e)
                    pf_out[s, j, e] = n.syn_out(self.V[idx])

        # Normalize to [0,1]
        pf_out = np.clip(pf_out, 0.0, 1.0)

        # Build muscle vector
        act = np.zeros(26, dtype=np.float32)
        for i, name in enumerate(MUSCLE_ORDER):
            base, side = name.rsplit("_", 1)
            s = self._side_idx(side)

            # Biarticulars: weighted sum
            if base in BIART_WEIGHTS:
                y = 0.0
                for (j_name, ef_name), w in BIART_WEIGHTS[base].items():
                    j = self._joint_idx(j_name)
                    e = self._ef_idx(ef_name)
                    y += w * pf_out[s, j, e]
                act[i] = y
                continue

            # Monoarticular
            if base in MONO_MAP:
                j_name, ef_name = MONO_MAP[base]
                j = self._joint_idx(j_name)
                e = self._ef_idx(ef_name)
                act[i] = pf_out[s, j, e]
            else:
                act[i] = 0.0  # unknown muscle -> off

        # Output scaling
        if not (self.cfg.out_min == 0.0 and self.cfg.out_max == 1.0):
            a = self.cfg.out_min
            b = self.cfg.out_max
            act = a + (b - a) * act
        return act
