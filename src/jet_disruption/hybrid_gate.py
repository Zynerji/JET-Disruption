"""Hybrid checker gate combining three orthogonal signals:

A. Coherence-of-derivative gate (sign(dz/dt) consensus across channels)
B. Kuramoto order parameter R from analytic-phase synchronization
C. Diagnostic Hamiltonian conservation breakdown |dH/dt|

Final alert = K-of-3 vote (default K=2). Rationale: independent failure
modes — A misses slow precursors, B misses non-oscillatory ones,
C misses balanced (energy-preserving) precursors. Requiring agreement
between two indicators suppresses false alarms from each gate's noise.

Methodology import inspired by Knopp 2026 ("Golden Pendulum") — coupled
oscillator structure used as inductive bias for the detector. Does NOT
validate the Lagrangian itself; only uses the synchronisation/conservation
picture as a feature engineer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catcher_v2 import ConsensusResult, PerChannelScore


# ----------------------------------------------------------------- gate A

def coherent_novelty_mask(per_channel: list[PerChannelScore],
                            coherence_window: int = 5,
                            min_agreeing: int = 2) -> np.ndarray:
    """For each channel, mark |z| > thr AND sign(dz/dt) matches the majority
    of other above-threshold channels within `coherence_window` samples.

    Returns: (T, C) bool array of "coherent novelty" flags.
    """
    T = len(per_channel[0].z_score)
    C = len(per_channel)
    Z = np.stack([s.z_score for s in per_channel], axis=1)  # (T, C)
    is_novel = np.stack([s.is_novel for s in per_channel], axis=1)  # (T, C)
    # Local derivative direction
    dZ = np.diff(Z, axis=0, prepend=Z[:1, :])
    sign_dZ = np.sign(dZ).astype(np.int8)  # (T, C), {-1, 0, +1}

    # For each (t, c) we look at the smoothed sign over a small window
    # to get robust direction
    if coherence_window > 1:
        kernel = np.ones(coherence_window, dtype=np.float32) / coherence_window
        smooth_sign = np.zeros_like(sign_dZ, dtype=np.float32)
        for c in range(C):
            smooth_sign[:, c] = np.convolve(sign_dZ[:, c].astype(np.float32),
                                              kernel, mode="same")
        sign_dZ = np.sign(smooth_sign).astype(np.int8)

    # Vectorised: at each t, count novel+plus and novel+minus across c
    novel_plus = is_novel & (sign_dZ > 0)
    novel_minus = is_novel & (sign_dZ < 0)
    plus_count = novel_plus.sum(axis=1)
    minus_count = novel_minus.sum(axis=1)
    # Majority direction at each t (broadcast)
    plus_majority = (plus_count >= minus_count) & (plus_count >= min_agreeing)
    minus_majority = (minus_count > plus_count) & (minus_count >= min_agreeing)
    coherent = np.zeros((T, C), dtype=bool)
    coherent[plus_majority] = novel_plus[plus_majority]
    coherent[minus_majority] = novel_minus[minus_majority]
    return coherent


# ----------------------------------------------------------------- gate B

def kuramoto_order_parameter(per_channel: list[PerChannelScore],
                              smooth_samples: int = 10) -> np.ndarray:
    """Compute Kuramoto-like R(t) over channels from analytic phases of
    the z-score trajectories.

    Each channel is treated as one oscillator. Its analytic phase is
    computed via scipy Hilbert transform on the z-score time series.
    R(t) = |(1/C) sum_c exp(i phi_c(t))|

    Returns: (T,) array of R(t) in [0, 1].
    """
    from scipy.signal import hilbert
    Z = np.stack([s.z_score for s in per_channel], axis=1)  # (T, C)
    # Bandpass-pre-filter would help; for simplicity, Hilbert on raw z
    # works because z fluctuates around 0 with finite bandwidth.
    analytic = hilbert(Z, axis=0)  # complex (T, C)
    phases = np.angle(analytic)  # (T, C)
    R_t = np.abs(np.mean(np.exp(1j * phases), axis=1))  # (T,)

    if smooth_samples > 1:
        kernel = np.ones(smooth_samples, dtype=np.float32) / smooth_samples
        R_t = np.convolve(R_t, kernel, mode="same")
    return R_t.astype(np.float32)


# ----------------------------------------------------------------- gate C

def diagnostic_hamiltonian(per_channel: list[PerChannelScore],
                            weights: np.ndarray | None = None) -> np.ndarray:
    """Compute H(t) = sum_c w_c * z_c(t)^2.

    A "quiescent" plasma has approximately constant H (rolling baseline);
    a disruption spikes H. We return H(t) and let the caller compute
    |dH/dt| or relative-to-baseline excursion.

    Returns: (T,) array of H(t).
    """
    C = len(per_channel)
    if weights is None:
        weights = np.ones(C, dtype=np.float32) / C
    w = np.asarray(weights, dtype=np.float32)
    Z = np.stack([s.z_score for s in per_channel], axis=1)  # (T, C)
    # Clip extreme z to avoid float32 overflow when post-disruption diagnostics blow up.
    Z_clipped = np.clip(Z, -1e6, 1e6).astype(np.float64)
    H = (w[None, :] * (Z_clipped ** 2)).sum(axis=1)
    return H.astype(np.float32)


def dH_dt_normalized(H: np.ndarray, baseline_window: int = 200) -> np.ndarray:
    """Compute |dH/dt| normalised by trailing baseline of |dH/dt|."""
    dH = np.gradient(H)
    abs_dH = np.abs(dH)
    # Robust rolling baseline of abs_dH
    from numpy.lib.stride_tricks import sliding_window_view as swv
    T = len(H)
    if T <= baseline_window:
        return np.zeros(T, dtype=np.float32)
    n = T - baseline_window
    views = swv(abs_dH, baseline_window)[:n]
    med = np.median(views, axis=1)
    mad = np.median(np.abs(views - med[:, None]), axis=1)
    sigma = 1.4826 * mad + 1e-12
    out = np.zeros(T, dtype=np.float32)
    out[baseline_window:baseline_window + n] = (
        (abs_dH[baseline_window:baseline_window + n] - med) / sigma
    ).astype(np.float32)
    return out


# ------------------------------------------------------------ hybrid alert

@dataclass
class HybridResult:
    gate_a_consensus: np.ndarray  # (T,) fraction of coherent-novel channels
    gate_a_active: np.ndarray     # (T,) bool: A vote active
    gate_b_R: np.ndarray          # (T,) Kuramoto R
    gate_b_active: np.ndarray     # (T,) bool: B vote active
    gate_c_dH_z: np.ndarray       # (T,) |dH/dt| z-score
    gate_c_active: np.ndarray     # (T,) bool: C vote active
    votes: np.ndarray             # (T,) number of active gates 0-3
    alert_active: np.ndarray      # (T,) bool: votes >= K_threshold
    alert_start_idx: int | None
    consensus_count: np.ndarray   # (T,) compat field for scoring
    consensus_fraction: np.ndarray
    per_channel: list
    channel_names: tuple[str, ...]


def hybrid_alert(per_channel: list[PerChannelScore],
                 channel_names: tuple[str, ...],
                 weights: np.ndarray | None = None,
                 dt: float = 1e-3,
                 # Gate A
                 a_coherence_window: int = 5,
                 a_min_agreeing: int = 3,
                 a_fraction_threshold: float = 0.30,
                 # Gate B
                 b_R_excess: float = 0.15,  # absolute R rise vs baseline
                 b_baseline_samples: int = 300,
                 # Gate C
                 c_dH_z_threshold: float = 3.0,
                 c_baseline_samples: int = 300,
                 # Combiner
                 k_of_3: int = 2,
                 min_duration_ms: float = 5.0) -> HybridResult:
    """Run all 3 gates and combine via K-of-3 vote."""
    T = len(per_channel[0].z_score)
    C = len(per_channel)

    # --- Gate A: coherence-of-derivative ---
    coh_mask = coherent_novelty_mask(per_channel,
                                       coherence_window=a_coherence_window,
                                       min_agreeing=a_min_agreeing)  # (T, C)
    a_consensus = coh_mask.sum(axis=1).astype(np.float32) / C
    a_active = a_consensus >= a_fraction_threshold

    # --- Gate B: Kuramoto R ---
    R_t = kuramoto_order_parameter(per_channel, smooth_samples=10)
    # Vectorised rolling-median baseline (excludes last 10 samples to avoid
    # self-correlation). Pad start with zeros.
    from numpy.lib.stride_tricks import sliding_window_view as swv
    R_baseline = np.zeros_like(R_t)
    bw = b_baseline_samples
    skip = 10
    if T > bw + skip:
        # For time t in [bw, T-1], baseline = R[t-bw : t-skip]
        # length = bw - skip (constant)
        view_len = bw - skip
        all_views = swv(R_t, view_len)  # (T - view_len + 1, view_len)
        # The window ending at index (t - skip) is row (t - skip - view_len + 1)
        # = t - bw + 1. Valid for t >= bw - 1 (so row 0 corresponds to t = bw - 1)
        valid_t_start = bw
        valid_t_end = T  # exclusive
        n_valid = valid_t_end - valid_t_start
        rows = np.arange(n_valid) + (valid_t_start - bw + 1)
        rows = np.clip(rows, 0, len(all_views) - 1)
        R_baseline[valid_t_start:valid_t_end] = np.median(all_views[rows], axis=1)
    b_active = (R_t - R_baseline) >= b_R_excess

    # --- Gate C: diagnostic Hamiltonian ---
    H = diagnostic_hamiltonian(per_channel, weights=weights)
    dH_z = dH_dt_normalized(H, baseline_window=c_baseline_samples)
    c_active = dH_z >= c_dH_z_threshold

    # --- Combine ---
    votes = (a_active.astype(np.int8) + b_active.astype(np.int8)
              + c_active.astype(np.int8))
    above = votes >= k_of_3

    # Sustained run
    min_samples = max(1, int(min_duration_ms * 1e-3 / dt))
    alert = np.zeros(T, dtype=bool)
    alert_start = None
    run_start = None
    for t in range(T):
        if above[t]:
            if run_start is None:
                run_start = t
            elif t - run_start + 1 >= min_samples:
                alert[t] = True
                if alert_start is None:
                    alert_start = run_start
        else:
            run_start = None

    return HybridResult(
        gate_a_consensus=a_consensus, gate_a_active=a_active,
        gate_b_R=R_t, gate_b_active=b_active,
        gate_c_dH_z=dH_z, gate_c_active=c_active,
        votes=votes, alert_active=alert, alert_start_idx=alert_start,
        consensus_count=votes.astype(np.int32),
        consensus_fraction=votes.astype(np.float32) / 3.0,
        per_channel=per_channel, channel_names=channel_names,
    )
