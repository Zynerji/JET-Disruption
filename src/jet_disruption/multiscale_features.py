"""Multi-scale temporal features for the learned classifier.

Different disruption types have different precursor timescales:
  - Fast MHD instability: 10-30 ms
  - Radiation collapse: 50-100 ms
  - Density-limit / current-decay: 100-500 ms

We compute per-channel rolling robust z-scores at 3 baseline scales
(50 ms, 200 ms, 500 ms), then derive per-window features from each
scale and concatenate. We also add time-derivatives of the key gate
signals so the classifier can pick up RATE of synchronization rise.

Drops the dead gates (A coherence, C Hamiltonian, max_channel_z) that
the single-scale learned classifier showed carried zero coefficient.

Feature vector per window (3 scales × 4 features + 2 derivatives = 14):
  - weighted_consensus @ {50ms, 200ms, 500ms}
  - R @ {50ms, 200ms, 500ms}
  - R_excess @ {50ms, 200ms, 500ms}
  - n_channels_novel @ {50ms, 200ms, 500ms}
  - d/dt weighted_consensus @ 200ms
  - d/dt R_excess @ 200ms
"""
from __future__ import annotations

import numpy as np

from .catcher_v2 import per_channel_scores
from .hybrid_gate import kuramoto_order_parameter


MULTISCALE_FEATURE_NAMES = (
    "weighted_consensus_50ms",  "weighted_consensus_200ms",  "weighted_consensus_500ms",
    "R_50ms",                   "R_200ms",                   "R_500ms",
    "R_excess_50ms",            "R_excess_200ms",            "R_excess_500ms",
    "n_novel_50ms",             "n_novel_200ms",             "n_novel_500ms",
    "d_weighted_consensus_dt",  "d_R_excess_dt",
)


def _compute_one_scale_features(diag: np.ndarray, dt: float, weights: np.ndarray,
                                  baseline_ms: float,
                                  assess_ms: float = 5.0,
                                  z_threshold: float = 3.0,
                                  smooth_R_samples: int = 10):
    """Return weighted_consensus, R, R_excess, n_novel arrays at one scale."""
    pcs = per_channel_scores(diag, dt,
                              baseline_window_ms=baseline_ms,
                              assess_window_ms=assess_ms,
                              z_threshold=z_threshold)
    T = len(pcs[0].z_score)
    C = len(pcs)
    is_novel = np.stack([s.is_novel for s in pcs], axis=1)  # (T, C)

    # Weighted consensus
    w = np.asarray(weights, dtype=np.float32)
    w_total = float(w.sum()) + 1e-12
    weighted = (is_novel.astype(np.float32) * w[None, :]).sum(axis=1) / w_total

    # Kuramoto R
    R_t = kuramoto_order_parameter(pcs, smooth_samples=smooth_R_samples)

    # R-excess vs trailing-window median (scale-matched baseline)
    from numpy.lib.stride_tricks import sliding_window_view as swv
    R_baseline = np.zeros_like(R_t)
    bw = max(50, int(baseline_ms * 1e-3 / dt * 1.5))  # 1.5x scale for baseline
    skip = 10
    if T > bw + skip:
        view_len = bw - skip
        all_views = swv(R_t, view_len)
        n_valid = T - bw
        rows = np.arange(n_valid) + 1
        rows = np.clip(rows, 0, len(all_views) - 1)
        R_baseline[bw:] = np.median(all_views[rows], axis=1)
    R_excess = (R_t - R_baseline).astype(np.float32)

    n_novel = is_novel.sum(axis=1).astype(np.float32)

    return weighted, R_t.astype(np.float32), R_excess, n_novel


def compute_multiscale_features(diag: np.ndarray, dt: float,
                                  weights: np.ndarray,
                                  scales_ms: tuple[float, ...] = (50.0, 200.0, 500.0)
                                  ) -> np.ndarray:
    """Return (T, 14) multi-scale feature matrix."""
    feat_columns = []
    weighted_200ms = None
    R_excess_200ms = None
    for sc in scales_ms:
        weighted, R_t, R_excess, n_novel = _compute_one_scale_features(
            diag, dt, weights, baseline_ms=sc,
        )
        feat_columns.append((weighted, R_t, R_excess, n_novel))
        if sc == 200.0:
            weighted_200ms = weighted
            R_excess_200ms = R_excess

    # Assemble (T, 14) — first 3 scales × 4 features, then 2 derivatives
    out = np.zeros((diag.shape[0], 14), dtype=np.float32)
    # weighted_consensus per scale (cols 0, 1, 2)
    for i, fc in enumerate(feat_columns):
        out[:, i] = fc[0]
    # R per scale (cols 3, 4, 5)
    for i, fc in enumerate(feat_columns):
        out[:, 3 + i] = fc[1]
    # R_excess per scale (cols 6, 7, 8)
    for i, fc in enumerate(feat_columns):
        out[:, 6 + i] = fc[2]
    # n_novel per scale (cols 9, 10, 11)
    for i, fc in enumerate(feat_columns):
        out[:, 9 + i] = fc[3]
    # Derivatives (cols 12, 13)
    if weighted_200ms is not None:
        out[:, 12] = np.gradient(weighted_200ms).astype(np.float32)
    if R_excess_200ms is not None:
        out[:, 13] = np.gradient(R_excess_200ms).astype(np.float32)

    return out
