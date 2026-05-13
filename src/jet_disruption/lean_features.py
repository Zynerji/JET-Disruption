"""Lean feature set v5 — informed by failure analysis.

Drops: absolute-R features (no discrimination), per-channel peakz (fire
after disruption, fragile signal direction).

Keeps + adds:
  - weighted_consensus_200ms (peak)            — main workhorse
  - weighted_consensus_200ms_rate              — RATE of rise (last 100ms)
  - weighted_consensus_200ms_above_thresh_ms   — duration above 0.3
  - weighted_consensus_200ms_integral_500ms    — area under curve
  - R_excess_200ms                              — synchronization rise
  - R_excess_500ms                              — same, slower scale
  - n_novel_500ms                               — count tail
  - in_flattop / in_rampdown                    — phase context
  - 9 features total
"""
from __future__ import annotations

import math
import numpy as np

from .catcher_v2 import per_channel_scores
from .hybrid_gate import kuramoto_order_parameter


LEAN_FEATURE_NAMES = (
    "weighted_consensus_200ms",
    "weighted_consensus_200ms_rate",
    "weighted_consensus_200ms_above_0_3_ms",
    "weighted_consensus_200ms_integral_500ms",
    "R_excess_200ms",
    "R_excess_500ms",
    "n_novel_500ms",
    "in_flattop",
    "in_rampdown",
)


def _compute_R_excess(pcs: list, baseline_samples: int = 300) -> np.ndarray:
    R_t = kuramoto_order_parameter(pcs, smooth_samples=10)
    from numpy.lib.stride_tricks import sliding_window_view as swv
    T = len(R_t)
    R_baseline = np.zeros_like(R_t)
    bw = baseline_samples; skip = 10
    if T > bw + skip:
        view_len = bw - skip
        all_views = swv(R_t, view_len)
        rows = np.arange(T - bw) + 1
        rows = np.clip(rows, 0, len(all_views) - 1)
        R_baseline[bw:] = np.median(all_views[rows], axis=1)
    return (R_t - R_baseline).astype(np.float32)


def compute_lean_features(diag: np.ndarray, dt: float,
                            weights: np.ndarray,
                            time_s: np.ndarray,
                            ramp_up_tmax: float | None,
                            flat_top_tmax: float | None) -> np.ndarray:
    """Return (T, 9) lean feature matrix."""
    T = diag.shape[0]
    C = diag.shape[1]
    w = np.asarray(weights, dtype=np.float32)
    w_total = float(w.sum()) + 1e-12

    # Per-channel z at scale 200ms baseline (the dominant scale)
    pcs_200 = per_channel_scores(diag, dt,
                                   baseline_window_ms=200.0,
                                   assess_window_ms=5.0,
                                   z_threshold=3.0)
    is_novel_200 = np.stack([s.is_novel for s in pcs_200], axis=1)
    weighted_200 = (is_novel_200.astype(np.float32) * w[None, :]).sum(axis=1) / w_total

    # Rate: derivative of weighted_consensus_200 over 100ms = ~462 samples at 4638 Hz
    rate_window = max(1, int(0.1 / dt))
    weighted_rate = np.zeros_like(weighted_200)
    if T > rate_window + 1:
        weighted_rate[rate_window:] = (
            weighted_200[rate_window:] - weighted_200[:-rate_window]
        ) / (rate_window * dt)

    # Duration above 0.3 (ms): rolling count of samples in last 500ms where weighted > 0.3
    above_03 = (weighted_200 > 0.3).astype(np.float32)
    look_window = max(1, int(0.5 / dt))
    duration_ms = np.zeros_like(weighted_200)
    if T > look_window:
        from numpy.lib.stride_tricks import sliding_window_view as swv
        view = swv(above_03, look_window)
        duration_ms[look_window-1:] = view.sum(axis=1) * dt * 1000.0

    # Integral of weighted_consensus over last 500ms
    integral_500 = np.zeros_like(weighted_200)
    if T > look_window:
        from numpy.lib.stride_tricks import sliding_window_view as swv
        view = swv(weighted_200, look_window)
        integral_500[look_window-1:] = view.sum(axis=1) * dt

    # R-excess at 200 and 500 ms scales
    pcs_500 = per_channel_scores(diag, dt,
                                   baseline_window_ms=500.0,
                                   assess_window_ms=5.0,
                                   z_threshold=3.0)
    R_excess_200 = _compute_R_excess(pcs_200, baseline_samples=300)
    R_excess_500 = _compute_R_excess(pcs_500, baseline_samples=600)

    # n_novel at 500ms scale
    is_novel_500 = np.stack([s.is_novel for s in pcs_500], axis=1)
    n_novel_500 = is_novel_500.sum(axis=1).astype(np.float32)

    # Phase indicators
    in_flattop = np.zeros(T, dtype=np.float32)
    in_rampdown = np.zeros(T, dtype=np.float32)
    if (ramp_up_tmax is not None and not math.isnan(ramp_up_tmax)
            and flat_top_tmax is not None and not math.isnan(flat_top_tmax)):
        in_flattop[(time_s >= ramp_up_tmax) & (time_s <= flat_top_tmax)] = 1.0
        in_rampdown[time_s > flat_top_tmax] = 1.0

    out = np.stack([
        weighted_200,
        weighted_rate,
        duration_ms,
        integral_500,
        R_excess_200,
        R_excess_500,
        n_novel_500,
        in_flattop,
        in_rampdown,
    ], axis=1).astype(np.float32)
    return out
