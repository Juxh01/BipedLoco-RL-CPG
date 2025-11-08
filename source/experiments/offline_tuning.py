from __future__ import annotations

from typing import Dict, Tuple

import hydra
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from source.agents.CPG import CPGConfig, NeuronParams, TwoLayerCPG

# --------- Signal analysis helpers ---------


def _dominant_frequency_fft(signal: np.ndarray, dt: float) -> float | None:
    """Estimate dominant frequency using FFT (skip DC)."""
    x = np.asarray(signal, dtype=np.float64)
    if x.size < 8:
        return None
    x = x - float(np.mean(x))
    # Hann window to reduce leakage
    w = np.hanning(x.size)
    xw = x * w
    # zero-pad to next power of two for better resolution
    n = 1 << int(np.ceil(np.log2(xw.size)))
    X = np.fft.rfft(xw, n=n)
    freqs = np.fft.rfftfreq(n, d=dt)
    if freqs.size <= 1:
        return None
    power = X.real**2 + X.imag**2
    power[0] = 0.0
    k = int(np.argmax(power))
    f = float(freqs[k])
    return f if np.isfinite(f) and f > 0.0 else None


# --------- CPG evaluation (single-module, no feedback) ---------


def _setup_single_module_cpg(cfg: DictConfig) -> Tuple[TwoLayerCPG, Tuple[int, int]]:
    # Build neuron params from current trial values later; here use defaults, will be overwritten in objective
    cpg = TwoLayerCPG(CPGConfig())
    # Disable inter-joint couplings if requested
    if cfg.connectivity.disable_inter_joint:
        cpg.gains.Gw_lr_hip = 0.0
        cpg.gains.Gw_H_to_K = 0.0
        cpg.gains.Gw_K_to_A = 0.0
        cpg.gains.Gc_hip = 0.0
        cpg.gains.Gc_knee = 0.0
        cpg.gains.Gc_ankle = 0.0

    # Enable RG->PF only for selected joint
    joint = str(cfg.module.joint)
    j_idx = {"hip": 0, "knee": 1, "ankle": 2}[joint]
    gc_val = float(cfg.connectivity.Gc)
    if j_idx == 0:
        cpg.gains.Gc_hip = gc_val
    elif j_idx == 1:
        cpg.gains.Gc_knee = gc_val
    else:
        cpg.gains.Gc_ankle = gc_val

    # Mutual inhibitions
    cpg.cfg.RG_mutual_inh = float(cfg.connectivity.RG_mutual_inh)
    cpg.cfg.PF_mutual_inh = float(cfg.connectivity.PF_mutual_inh)

    # Set descending drive D only for selected module
    side = str(cfg.module.side)
    s_idx = 0 if side == "r" else 1
    D = float(cfg.module.drive)
    cpg.gains.D[:] = 0.0
    cpg.gains.D[s_idx, j_idx, 0] = D
    cpg.gains.D[s_idx, j_idx, 1] = D - 0.01

    # No sensory feedbacks
    cpg.gains.K_Ia[:] = 0.0
    cpg.gains.K_Ib[:] = 0.0

    return cpg, (s_idx, j_idx)


def _run_module_time_series(
    cpg: TwoLayerCPG, sm_idx: Tuple[int, int], cfg: DictConfig
) -> Dict[str, np.ndarray]:
    dt = float(cfg.sim.dt)
    warm = float(cfg.sim.warmup_time)
    meas = float(cfg.sim.measure_time)
    n_warm = int(np.ceil(warm / dt))
    n_meas = int(np.ceil(meas / dt))

    s_idx, j_idx = sm_idx

    # collect RG E/F and PF E/F outputs for the selected module only
    rg_e = []
    rg_f = []
    pf_e = []
    pf_f = []

    # Internal neuron states and signals for the selected module
    V_rg_e, V_rg_f, V_pf_e, V_pf_f = [], [], [], []
    h_rg_e, h_rg_f, h_pf_e, h_pf_f = [], [], [], []
    m_inf_rg_e, m_inf_rg_f, m_inf_pf_e, m_inf_pf_f = [], [], [], []
    h_inf_rg_e, h_inf_rg_f, h_inf_pf_e, h_inf_pf_f = [], [], [], []
    tau_h_rg_e, tau_h_rg_f, tau_h_pf_e, tau_h_pf_f = [], [], [], []
    s_rg_e, s_rg_f, s_pf_e, s_pf_f = [], [], [], []
    I_syn_rg_e, I_syn_rg_f, I_syn_pf_e, I_syn_pf_f = [], [], [], []
    I_app_rg_e, I_app_rg_f, I_app_pf_e, I_app_pf_f = [], [], [], []

    # warmup
    for _ in range(n_warm):
        cpg.step(dt_env=dt, n_substeps=1, Ia_muscle=None, Ib_muscle=None)

    # measure
    for _ in range(n_meas):
        cpg.step(dt_env=dt, n_substeps=1, Ia_muscle=None, Ib_muscle=None)
        # indices for outputs
        i_rg_e = cpg._id(0, s_idx, j_idx, 0)
        i_rg_f = cpg._id(0, s_idx, j_idx, 1)
        i_pf_e = cpg._id(1, s_idx, j_idx, 0)
        i_pf_f = cpg._id(1, s_idx, j_idx, 1)
        n = cpg.cfg.neuron
        rg_e.append(float(n.syn_out(cpg.V[i_rg_e])))
        rg_f.append(float(n.syn_out(cpg.V[i_rg_f])))
        pf_e.append(float(n.syn_out(cpg.V[i_pf_e])))
        pf_f.append(float(n.syn_out(cpg.V[i_pf_f])))

        # Internal states
        V_e_rg = float(cpg.V[i_rg_e])
        V_f_rg = float(cpg.V[i_rg_f])
        V_e_pf = float(cpg.V[i_pf_e])
        V_f_pf = float(cpg.V[i_pf_f])

        h_e_rg = float(cpg.h[i_rg_e])
        h_f_rg = float(cpg.h[i_rg_f])
        h_e_pf = float(cpg.h[i_pf_e])
        h_f_pf = float(cpg.h[i_pf_f])

        V_rg_e.append(V_e_rg)
        V_rg_f.append(V_f_rg)
        V_pf_e.append(V_e_pf)
        V_pf_f.append(V_f_pf)
        h_rg_e.append(h_e_rg)
        h_rg_f.append(h_f_rg)
        h_pf_e.append(h_e_pf)
        h_pf_f.append(h_f_pf)

        # Derived neuronal quantities for the selected neurons
        m_inf_rg_e.append(float(n.m_inf(np.array([V_e_rg]))[0]))
        m_inf_rg_f.append(float(n.m_inf(np.array([V_f_rg]))[0]))
        m_inf_pf_e.append(float(n.m_inf(np.array([V_e_pf]))[0]))
        m_inf_pf_f.append(float(n.m_inf(np.array([V_f_pf]))[0]))

        h_inf_rg_e.append(float(n.h_inf(np.array([V_e_rg]))[0]))
        h_inf_rg_f.append(float(n.h_inf(np.array([V_f_rg]))[0]))
        h_inf_pf_e.append(float(n.h_inf(np.array([V_e_pf]))[0]))
        h_inf_pf_f.append(float(n.h_inf(np.array([V_f_pf]))[0]))

        tau_h_rg_e.append(float(n.tau_h(np.array([V_e_rg]))[0]))
        tau_h_rg_f.append(float(n.tau_h(np.array([V_f_rg]))[0]))
        tau_h_pf_e.append(float(n.tau_h(np.array([V_e_pf]))[0]))
        tau_h_pf_f.append(float(n.tau_h(np.array([V_f_pf]))[0]))

        # Synaptic outputs and currents
        s_all = n.syn_out(cpg.V)
        s_rg_e.append(float(s_all[i_rg_e]))
        s_rg_f.append(float(s_all[i_rg_f]))
        s_pf_e.append(float(s_all[i_pf_e]))
        s_pf_f.append(float(s_all[i_pf_f]))

        I_syn_all = cpg._synaptic_current(s_all, cpg.V)
        I_syn_rg_e.append(float(I_syn_all[i_rg_e]))
        I_syn_rg_f.append(float(I_syn_all[i_rg_f]))
        I_syn_pf_e.append(float(I_syn_all[i_pf_e]))
        I_syn_pf_f.append(float(I_syn_all[i_pf_f]))

        # Applied external inputs (descending drive and feedback aggregation)
        I_app_RG, I_app_PF = cpg._assemble_inputs(Ia_muscle=None, Ib_muscle=None)
        I_app_RG = I_app_RG.reshape(2, 3, 2)
        I_app_PF = I_app_PF.reshape(2, 3, 2)
        I_app_rg_e.append(float(I_app_RG[s_idx, j_idx, 0]))
        I_app_rg_f.append(float(I_app_RG[s_idx, j_idx, 1]))
        I_app_pf_e.append(float(I_app_PF[s_idx, j_idx, 0]))
        I_app_pf_f.append(float(I_app_PF[s_idx, j_idx, 1]))

    return {
        "dt": np.array([dt], dtype=np.float32),
        "rg_e": np.array(rg_e, dtype=np.float32),
        "rg_f": np.array(rg_f, dtype=np.float32),
        "pf_e": np.array(pf_e, dtype=np.float32),
        "pf_f": np.array(pf_f, dtype=np.float32),
        # Internals
        "V_rg_e": np.array(V_rg_e, dtype=np.float32),
        "V_rg_f": np.array(V_rg_f, dtype=np.float32),
        "V_pf_e": np.array(V_pf_e, dtype=np.float32),
        "V_pf_f": np.array(V_pf_f, dtype=np.float32),
        "h_rg_e": np.array(h_rg_e, dtype=np.float32),
        "h_rg_f": np.array(h_rg_f, dtype=np.float32),
        "h_pf_e": np.array(h_pf_e, dtype=np.float32),
        "h_pf_f": np.array(h_pf_f, dtype=np.float32),
        "m_inf_rg_e": np.array(m_inf_rg_e, dtype=np.float32),
        "m_inf_rg_f": np.array(m_inf_rg_f, dtype=np.float32),
        "m_inf_pf_e": np.array(m_inf_pf_e, dtype=np.float32),
        "m_inf_pf_f": np.array(m_inf_pf_f, dtype=np.float32),
        "h_inf_rg_e": np.array(h_inf_rg_e, dtype=np.float32),
        "h_inf_rg_f": np.array(h_inf_rg_f, dtype=np.float32),
        "h_inf_pf_e": np.array(h_inf_pf_e, dtype=np.float32),
        "h_inf_pf_f": np.array(h_inf_pf_f, dtype=np.float32),
        "tau_h_rg_e": np.array(tau_h_rg_e, dtype=np.float32),
        "tau_h_rg_f": np.array(tau_h_rg_f, dtype=np.float32),
        "tau_h_pf_e": np.array(tau_h_pf_e, dtype=np.float32),
        "tau_h_pf_f": np.array(tau_h_pf_f, dtype=np.float32),
        "s_rg_e": np.array(s_rg_e, dtype=np.float32),
        "s_rg_f": np.array(s_rg_f, dtype=np.float32),
        "s_pf_e": np.array(s_pf_e, dtype=np.float32),
        "s_pf_f": np.array(s_pf_f, dtype=np.float32),
        "I_syn_rg_e": np.array(I_syn_rg_e, dtype=np.float32),
        "I_syn_rg_f": np.array(I_syn_rg_f, dtype=np.float32),
        "I_syn_pf_e": np.array(I_syn_pf_e, dtype=np.float32),
        "I_syn_pf_f": np.array(I_syn_pf_f, dtype=np.float32),
        "I_app_rg_e": np.array(I_app_rg_e, dtype=np.float32),
        "I_app_rg_f": np.array(I_app_rg_f, dtype=np.float32),
        "I_app_pf_e": np.array(I_app_pf_e, dtype=np.float32),
        "I_app_pf_f": np.array(I_app_pf_f, dtype=np.float32),
    }


def _compute_cpg_losses(
    ts: Dict[str, np.ndarray], cfg: DictConfig
) -> Dict[str, float | None]:
    """Hierarchical multi-objective loss per Diagnostic Analysis.

    Stage 1 (gate): Detect oscillation via variance of rg_e and rg_f.
      - If either variance < threshold: classify as fixed-point and return a large
        penalty wall plus an inverse-variance penalty to create a gradient.

    Stage 2 (island): If oscillating, optimize for anti-phase correlation and frequency.
      - Phase: Pearson correlation between rg_e and rg_f, target -1.0 (anti-phase)
      - Frequency: FFT-based dominant frequency of x_rg = rg_e - rg_f
    """
    dt = float(ts["dt"][0])
    rg_e_signal = np.asarray(ts["rg_e"], dtype=np.float64)
    rg_f_signal = np.asarray(ts["rg_f"], dtype=np.float64)

    # Config parameters
    var_threshold = float(getattr(cfg.loss, "variance_threshold", 1.0e-5))
    penalty_wall = float(getattr(cfg.loss, "invalid_penalty_wall", 1000.0))
    invalid_penalty = float(getattr(cfg.loss, "invalid_penalty", 10.0))
    w_phase = float(getattr(cfg.loss, "w_phase", 10.0))
    w_freq = float(getattr(cfg.loss, "w_freq", 1.0))
    freq_target = float(cfg.loss.freq_target)

    eps = 1e-9
    var_e = float(np.var(rg_e_signal))
    var_f = float(np.var(rg_f_signal))

    metrics: Dict[str, float | None] = {
        "variance_e": var_e,
        "variance_f": var_f,
    }

    # Stage 1: Fixed-point detection via variance gate
    if (var_e < var_threshold) or (var_f < var_threshold):
        L_amp = (1.0 / (var_e + eps)) + (1.0 / (var_f + eps))
        total_loss = penalty_wall + L_amp
        # Optional diagnostics
        metrics.update(
            {
                "freq": 0.0,
                "freq_loss": (0.0 - freq_target) ** 2,
                "phase_corr": 0.0,
                "phase_loss": 4.0,  # worst-case if used
                "total_loss": float(total_loss),
            }
        )
        return metrics

    # Stage 2: Oscillation present -> enforce anti-phase and frequency
    # Phase correlation (anti-phase target = -1)
    try:
        corr_matrix = np.corrcoef(rg_e_signal, rg_f_signal)
        # corrcoef returns 2x2 matrix; off-diagonal [0,1] is correlation between the two
        phase_corr = float(corr_matrix[0, 1])
    except Exception:
        phase_corr = 0.0
    L_phase = (phase_corr - (-1.0)) ** 2

    # Frequency term from x_rg
    x_rg = rg_e_signal - rg_f_signal
    f = _dominant_frequency_fft(x_rg, dt)
    if f is None or not np.isfinite(f) or f == 0.0:
        L_freq = invalid_penalty
        f_metric = None
    else:
        L_freq = (float(f) - freq_target) ** 2
        f_metric = float(f)

    total_loss = w_phase * L_phase + w_freq * L_freq
    metrics.update(
        {
            "freq": f_metric,
            "freq_loss": float(L_freq),
            "phase_corr": float(phase_corr),
            "phase_loss": float(L_phase),
            "total_loss": float(total_loss),
        }
    )
    return metrics


def _plot_signals(ts: Dict[str, np.ndarray], cfg: DictConfig) -> None:
    """Plot x_rg, rg_e, rg_f for the measured window and save to file in the run dir."""
    dt = float(ts["dt"][0])
    t = np.arange(0, ts["rg_e"].size, dtype=np.float32) * dt
    rg_e = ts["rg_e"]
    rg_f = ts["rg_f"]
    x_rg = rg_e - rg_f

    # Figure with two rows: x_rg on top, rg_e/rg_f on bottom
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
    ax0.plot(t, x_rg, label="x_rg = rg_e - rg_f", color="#1f77b4")
    ax0.set_ylabel("x_rg")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right")

    ax1.plot(t, rg_e, label="rg_e", color="#2ca02c")
    ax1.plot(t, rg_f, label="rg_f", color="#d62728")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("RG outputs")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    fname = "signals.png"
    try:
        if (
            hasattr(cfg, "output")
            and hasattr(cfg.output, "plot_filename")
            and cfg.output.plot_filename
        ):
            fname = str(cfg.output.plot_filename)
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def _plot_internal_signals(ts: Dict[str, np.ndarray], cfg: DictConfig) -> None:
    """Plot internal neuronal signals for the selected module (RG/PF, E/F):
    - Membrane potential V
    - Inactivation variable h
    - Steady-state activations m_inf, h_inf
    - Inactivation time constant tau_h
    - Synaptic output s (n.syn_out(V))
    - Applied input I_app (RG/PF)
    - Net synaptic current I_syn
    """
    dt = float(ts["dt"][0])
    T = int(ts["rg_e"].size)
    t = np.arange(0, T, dtype=np.float32) * dt

    # Helper to safely fetch
    def g(key: str) -> np.ndarray:
        return np.asarray(ts.get(key, np.zeros(T, dtype=np.float32)), dtype=np.float32)

    # Build figure with multiple stacked panels
    labels = ["RG_E", "RG_F", "PF_E", "PF_F"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    series = {
        "Membrane potential V [mV]": [
            g("V_rg_e"),
            g("V_rg_f"),
            g("V_pf_e"),
            g("V_pf_f"),
        ],
        "Inactivation h": [g("h_rg_e"), g("h_rg_f"), g("h_pf_e"), g("h_pf_f")],
        "m_inf(V)": [
            g("m_inf_rg_e"),
            g("m_inf_rg_f"),
            g("m_inf_pf_e"),
            g("m_inf_pf_f"),
        ],
        "h_inf(V)": [
            g("h_inf_rg_e"),
            g("h_inf_rg_f"),
            g("h_inf_pf_e"),
            g("h_inf_pf_f"),
        ],
        "tau_h(V) [s]": [
            g("tau_h_rg_e"),
            g("tau_h_rg_f"),
            g("tau_h_pf_e"),
            g("tau_h_pf_f"),
        ],
        "Synaptic output s(V)": [g("s_rg_e"), g("s_rg_f"), g("s_pf_e"), g("s_pf_f")],
        "Applied input I_app": [
            g("I_app_rg_e"),
            g("I_app_rg_f"),
            g("I_app_pf_e"),
            g("I_app_pf_f"),
        ],
        "Net synaptic current I_syn": [
            g("I_syn_rg_e"),
            g("I_syn_rg_f"),
            g("I_syn_pf_e"),
            g("I_syn_pf_f"),
        ],
    }

    n_rows = len(series)
    fig, axs = plt.subplots(n_rows, 1, sharex=True, figsize=(10, 2.3 * n_rows))
    if n_rows == 1:
        axs = [axs]

    for ax, (title, curves) in zip(axs, series.items()):
        for y, lab, col in zip(curves, labels, colors):
            ax.plot(t, y, label=lab, color=col, linewidth=1.2)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=4, fontsize=8)

    axs[-1].set_xlabel("time [s]")

    fname = "internals.png"
    try:
        if (
            hasattr(cfg, "output")
            and hasattr(cfg.output, "internal_plot_filename")
            and cfg.output.internal_plot_filename
        ):
            fname = str(cfg.output.internal_plot_filename)
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def _build_neuron_params_from_cfg(cfg: DictConfig) -> NeuronParams:
    n = cfg.neuron
    return NeuronParams(
        Cm=float(n.Cm),
        Gm=float(n.Gm),
        Er=float(n.Er),
        ENa=float(n.ENa),
        GNa=float(n.GNa),
        V_half_m=float(n.V_half_m),
        k_m=float(n.k_m),
        V_half_h=float(n.V_half_h),
        k_h=float(n.k_h),
        tau_h_min=float(n.tau_h_min),
        tau_h_max=float(n.tau_h_max),
        V_tau=float(n.V_tau),
        k_tau=float(n.k_tau),
        V_th_syn=float(n.V_th_syn),
        k_syn=float(n.k_syn),
    )


@hydra.main(
    config_path="../../configs", config_name="offline_tuning.yaml", version_base="1.1"
)
def main(cfg: DictConfig) -> float:
    """Single-run objective for Hydra-Optuna sweeper: return frequency-only loss."""
    # Build neuron params from config (sweeper overrides these fields per trial)
    p = _build_neuron_params_from_cfg(cfg)

    # Build and configure single-module CPG
    cpg = TwoLayerCPG(CPGConfig(neuron=p))
    cpg, sm_idx = _setup_single_module_cpg(cfg)
    cpg.cfg.neuron = p

    # simulate and compute frequency-only loss
    ts = _run_module_time_series(cpg, sm_idx, cfg)
    metrics = _compute_cpg_losses(ts, cfg)
    # Save plot for this run
    try:
        do_plot = True
        if hasattr(cfg, "output") and hasattr(cfg.output, "plot_signals"):
            do_plot = bool(cfg.output.plot_signals)
        if do_plot:
            _plot_signals(ts, cfg)
        # Optionally also plot internal signals
        do_plot_internal = True
        if hasattr(cfg, "output") and hasattr(cfg.output, "plot_internal_signals"):
            do_plot_internal = bool(cfg.output.plot_internal_signals)
        if do_plot_internal:
            _plot_internal_signals(ts, cfg)
    except Exception as e:
        # Avoid breaking optimization on plotting errors
        print(f"[plot] failed: {e}")

    # Display key outputs (dominant freq and loss). Hydra-Optuna will also print best at the end.
    print(
        {
            "dominant_freq_hz": metrics.get("freq", None),
            "target_hz": float(cfg.loss.freq_target),
            "freq_loss": metrics.get("freq_loss", None),
            "phase_corr": metrics.get("phase_corr", None),
            "phase_loss": metrics.get("phase_loss", None),
            "variance_e": metrics.get("variance_e", None),
            "variance_f": metrics.get("variance_f", None),
            "total_loss": metrics.get("total_loss", None),
        }
    )

    # Return total loss for the sweeper
    total = metrics.get("total_loss", None)
    if total is None or not np.isfinite(total):
        return float(getattr(cfg.loss, "invalid_penalty_wall", 1000.0))
    return float(total)


if __name__ == "__main__":
    main()
