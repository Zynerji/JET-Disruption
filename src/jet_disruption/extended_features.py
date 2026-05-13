"""Extended features = multi-scale collective + per-channel peak features.

Adds 10 per-channel peak |z| over a 500ms trailing window. The intuition:
the existing collective features can't distinguish a disruption with a
strong locked-mode (saddle coil) signature from a normal plasma with a
strong density fluctuation. Per-channel breakdown gives that.

Total = 14 multi-scale + 10 per-channel peak = 24 features.
"""
from __future__ import annotations

import numpy as np

from .catcher_v2 import per_channel_scores
from .multiscale_features import (
    compute_multiscale_features, MULTISCALE_FEATURE_NAMES,
)


def _per_channel_peak_z_500ms(diag: np.ndarray, dt: float,
                                channel_names: tuple,
                                window_ms: float = 500.0,
                                assess_ms: float = 5.0,
                                z_threshold: float = 3.0) -> np.ndarray:
    """Compute (T, C) per-channel peak |z| over trailing window_ms."""
    pcs = per_channel_scores(diag, dt,
                              baseline_window_ms=window_ms,
                              assess_window_ms=assess_ms,
                              z_threshold=z_threshold)
    T = len(pcs[0].z_score)
    C = len(pcs)
    Z = np.stack([s.z_score for s in pcs], axis=1)  # (T, C)
    # Sliding-window max |z| over last w samples
    w = int(window_ms * 1e-3 / dt)
    if T <= w:
        return np.zeros((T, C), dtype=np.float32)
    from numpy.lib.stride_tricks import sliding_window_view as swv
    abs_Z = np.abs(Z)
    out = np.zeros((T, C), dtype=np.float32)
    for c in range(C):
        view = swv(abs_Z[:, c], w)  # (T-w+1, w)
        out[w-1:, c] = view.max(axis=1)
    return out


def per_channel_feature_names(channel_names: tuple) -> tuple:
    return tuple(f"peakz_500ms_{c}" for c in channel_names)


EXTENDED_FEATURE_NAMES_TEMPLATE = (
    MULTISCALE_FEATURE_NAMES,  # 14
    "<per_channel_peakz_500ms>",  # 10 (filled at runtime)
)


def compute_extended_features(diag: np.ndarray, dt: float,
                                weights: np.ndarray,
                                channel_names: tuple,
                                scales_ms: tuple = (50.0, 200.0, 500.0)
                                ) -> tuple[np.ndarray, tuple]:
    """Return (T, 14 + C) extended feature matrix + name tuple."""
    multi = compute_multiscale_features(diag, dt, weights, scales_ms=scales_ms)
    per_ch = _per_channel_peak_z_500ms(diag, dt, channel_names)
    out = np.concatenate([multi, per_ch], axis=1)
    feat_names = tuple(MULTISCALE_FEATURE_NAMES) + per_channel_feature_names(channel_names)
    return out.astype(np.float32), feat_names
