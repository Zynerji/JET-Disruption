"""Cross-diagnostic consensus catcher for tokamak time series.

Ported and adapted from systrophe.catcher_v2. Where Systrophe operates on
a parameter axis with vector outputs, here we operate on a time axis with
multi-channel diagnostic data. The cross-channel consensus filter is the
disruption-prediction structure: a regime change is a moment when MANY
channels simultaneously deflect from baseline, not just one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PerChannelScore:
    """Per-channel novelty score time series."""
    z_score: np.ndarray  # (T,) robust z-score vs trailing baseline
    is_novel: np.ndarray  # (T,) bool, |z| above threshold


@dataclass
class ConsensusResult:
    """Consensus catcher result."""
    consensus_fraction: np.ndarray  # (T,) fraction of channels novel at each t
    consensus_count: np.ndarray  # (T,) count of channels novel at each t
    alert_active: np.ndarray  # (T,) bool, ≥ frac_threshold for ≥ min_duration
    alert_start_idx: int | None  # first sample where alert fires, or None
    per_channel: list[PerChannelScore]
    channel_names: tuple[str, ...]


def robust_zscore_rolling(x: np.ndarray, baseline_window: int,
                          assess_window: int) -> np.ndarray:
    """Vectorized rolling robust z-score.

    For each sample t:
      baseline: x[t - baseline_window : t - assess_window]   (length B-A)
      now:      x[t - assess_window : t]                       (length A)
    Z = (median(now) - median(baseline)) / (1.4826 * MAD(baseline))

    Uses sliding-window views + np.median over axis=1 for ~100x speedup over
    the per-t Python loop.
    """
    from numpy.lib.stride_tricks import sliding_window_view as swv
    T = len(x)
    B = baseline_window
    A = assess_window
    if T <= B:
        return np.zeros(T, dtype=np.float32)

    # Compute z at times t = B, B+1, ..., T-1.  i = t - B, range 0..T-1-B,
    # length n = T - B.
    n = T - B
    all_views = swv(x, B)                  # (T-B+1, B): row i = x[i:i+B]
    baselines = all_views[:n, : B - A]     # (n, B-A) = x[i : i + B - A]
    now_views = swv(x, A)                  # (T-A+1, A): row j = x[j:j+A]
    nows = now_views[B - A : B - A + n]    # (n, A) = x[i+B-A : i+B]

    bl_med = np.median(baselines, axis=1)
    bl_mad = np.median(np.abs(baselines - bl_med[:, None]), axis=1)
    now_med = np.median(nows, axis=1)
    sigma = 1.4826 * bl_mad + 1e-12
    z_vals = (now_med - bl_med) / sigma  # (n,)

    z = np.zeros(T, dtype=np.float32)
    z[B : B + n] = z_vals.astype(np.float32)
    return z


def per_channel_scores(diag: np.ndarray, dt: float,
                       baseline_window_ms: float = 200.0,
                       assess_window_ms: float = 20.0,
                       z_threshold: float = 3.0) -> list[PerChannelScore]:
    """Compute per-channel rolling z-score + novelty flag.

    diag: (T, C) array.
    dt: sample period in seconds.
    """
    T, C = diag.shape
    bl_w = int(baseline_window_ms * 1e-3 / dt)
    as_w = int(assess_window_ms * 1e-3 / dt)
    scores = []
    for c in range(C):
        z = robust_zscore_rolling(diag[:, c], bl_w, as_w)
        is_novel = np.abs(z) > z_threshold
        scores.append(PerChannelScore(z_score=z, is_novel=is_novel))
    return scores


def consensus_alert(per_channel: list[PerChannelScore],
                    channel_names: tuple[str, ...],
                    fraction_threshold: float = 0.3,
                    min_duration_ms: float = 10.0,
                    dt: float = 1e-3) -> ConsensusResult:
    """Combine per-channel novelty flags into a consensus alert.

    Alert fires when fraction-above-threshold ≥ fraction_threshold for
    a sustained ≥ min_duration_ms.

    fraction_threshold: 0.3 = 30% of channels (≥ 5 of 14 by default)
    """
    T = len(per_channel[0].z_score)
    C = len(per_channel)
    count = np.zeros(T, dtype=np.int32)
    for s in per_channel:
        count += s.is_novel.astype(np.int32)
    frac = count.astype(np.float32) / C

    min_samples = int(min_duration_ms * 1e-3 / dt)
    above = frac >= fraction_threshold

    # Find sustained runs
    alert = np.zeros(T, dtype=bool)
    run_start = None
    alert_start_idx = None
    for t in range(T):
        if above[t]:
            if run_start is None:
                run_start = t
            elif t - run_start + 1 >= min_samples:
                alert[t] = True
                if alert_start_idx is None:
                    alert_start_idx = run_start
        else:
            run_start = None

    return ConsensusResult(
        consensus_fraction=frac,
        consensus_count=count,
        alert_active=alert,
        alert_start_idx=alert_start_idx,
        per_channel=per_channel,
        channel_names=channel_names,
    )


def run_catcher(diag: np.ndarray, time_s: np.ndarray,
                channel_names: tuple[str, ...],
                baseline_window_ms: float = 200.0,
                assess_window_ms: float = 20.0,
                z_threshold: float = 3.0,
                fraction_threshold: float = 0.30,
                min_duration_ms: float = 10.0) -> ConsensusResult:
    """One-shot: per-channel scoring + consensus aggregation."""
    dt = float(time_s[1] - time_s[0])
    pcs = per_channel_scores(diag, dt,
                              baseline_window_ms=baseline_window_ms,
                              assess_window_ms=assess_window_ms,
                              z_threshold=z_threshold)
    return consensus_alert(pcs, channel_names,
                            fraction_threshold=fraction_threshold,
                            min_duration_ms=min_duration_ms, dt=dt)
