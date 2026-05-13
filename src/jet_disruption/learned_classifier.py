"""Learned-feature classifier for tokamak disruption alerts.

Replaces the K-of-N hand voting with a small logistic regression over
per-window scalar features extracted from the three hybrid gates plus
the weighted-consensus signal.

Per-window features (window = 20 ms before t):
  - weighted_consensus       (gate from weighting.py)
  - gate_a_consensus         (coherence-of-derivative)
  - gate_b_R                 (Kuramoto order parameter)
  - gate_b_R_excess          (R above rolling baseline)
  - gate_c_dH_z              (Hamiltonian d/dt z-score)
  - max_channel_z            (single-channel max |z|)
  - n_channels_novel         (raw count of |z|>3 channels)

At inference, the classifier scores every window in the analysis range,
and we alert when the score exceeds a learned threshold for a sustained
min_duration_ms.

Training data is sampled from labelled MAST shots:
  positive windows: 20 ms-spaced windows in the precursor zone
                    [t_disrupt - 100ms, t_disrupt - 5ms]
  negative windows: 20 ms-spaced windows in flat-top quiescent zone
                    [flat_top_tmin + 50ms, t_disrupt - 200ms] for
                    disruptive shots; full flat-top for non-disruptive.

The classifier is sklearn LogisticRegression(class_weight='balanced').
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .catcher_v2 import per_channel_scores, PerChannelScore
from .hybrid_gate import (
    coherent_novelty_mask, kuramoto_order_parameter,
    diagnostic_hamiltonian, dH_dt_normalized,
)
from .weighting import weighted_consensus_alert


FEATURE_NAMES = (
    "weighted_consensus",
    "gate_a_consensus",
    "gate_b_R",
    "gate_b_R_excess",
    "gate_c_dH_z",
    "max_channel_z",
    "n_channels_novel",
)


def compute_features(per_channel: list[PerChannelScore],
                      weights: np.ndarray,
                      dt: float,
                      a_coherence_window: int = 5,
                      a_min_agreeing: int = 3,
                      b_baseline_samples: int = 300,
                      c_baseline_samples: int = 300) -> np.ndarray:
    """Compute (T, 7) feature matrix for a single shot."""
    T = len(per_channel[0].z_score)
    C = len(per_channel)
    Z = np.stack([s.z_score for s in per_channel], axis=1)  # (T, C)
    is_novel = np.stack([s.is_novel for s in per_channel], axis=1)

    # Weighted consensus signal (re-derive without thresholding)
    w = np.asarray(weights, dtype=np.float32)
    w_total = float(w.sum()) + 1e-12
    weighted = (is_novel.astype(np.float32) * w[None, :]).sum(axis=1) / w_total

    # Gate A
    coh = coherent_novelty_mask(per_channel,
                                  coherence_window=a_coherence_window,
                                  min_agreeing=a_min_agreeing)
    a_consensus = coh.sum(axis=1).astype(np.float32) / C

    # Gate B
    R_t = kuramoto_order_parameter(per_channel, smooth_samples=10)
    # Rolling baseline R (reuse logic from hybrid_alert in compact form)
    from numpy.lib.stride_tricks import sliding_window_view as swv
    R_baseline = np.zeros_like(R_t)
    bw = b_baseline_samples
    skip = 10
    if T > bw + skip:
        view_len = bw - skip
        all_views = swv(R_t, view_len)
        valid_t_start = bw
        n_valid = T - valid_t_start
        rows = np.arange(n_valid) + (valid_t_start - bw + 1)
        rows = np.clip(rows, 0, len(all_views) - 1)
        R_baseline[valid_t_start:] = np.median(all_views[rows], axis=1)
    R_excess = (R_t - R_baseline).astype(np.float32)

    # Gate C
    H = diagnostic_hamiltonian(per_channel, weights=w)
    dH_z = dH_dt_normalized(H, baseline_window=c_baseline_samples)

    # Aggregate features
    max_z = np.abs(Z).max(axis=1)
    n_novel = is_novel.sum(axis=1).astype(np.float32)

    feats = np.stack([
        weighted,
        a_consensus,
        R_t,
        R_excess,
        dH_z,
        max_z,
        n_novel,
    ], axis=1).astype(np.float32)
    return feats


def sample_training_windows(shot, features: np.ndarray, dt: float,
                              baseline_window_ms: float = 50.0,
                              precursor_lead_ms: float = 100.0,
                              precursor_min_ahead_ms: float = 5.0,
                              quiescent_buffer_ms: float = 200.0,
                              stride_ms: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Sample feature rows + binary labels from one shot.

    Returns (X, y) where each row of X is a per-window feature vector and
    y is 1 for precursor windows, 0 for quiescent windows.
    """
    import math
    ft_min = getattr(shot, "flat_top_tmin", None)
    ft_max = getattr(shot, "flat_top_tmax", None)
    if (ft_min is None or ft_max is None
            or (isinstance(ft_min, float) and math.isnan(ft_min))
            or (isinstance(ft_max, float) and math.isnan(ft_max))):
        return np.empty((0, features.shape[1])), np.empty(0, dtype=np.int8)
    earliest = int((ft_min + baseline_window_ms * 1e-3) / dt)
    stride = max(1, int(stride_ms * 1e-3 / dt))

    X_list, y_list = [], []
    if shot.is_disruptive and shot.t_disrupt_s is not None:
        td_idx = int(shot.t_disrupt_s / dt)
        pre_start = int((shot.t_disrupt_s - precursor_lead_ms * 1e-3) / dt)
        pre_end = int((shot.t_disrupt_s - precursor_min_ahead_ms * 1e-3) / dt)
        pre_start = max(pre_start, earliest)
        # Positive windows
        for t in range(pre_start, pre_end, stride):
            if 0 <= t < len(features):
                X_list.append(features[t])
                y_list.append(1)
        # Negative windows: flat-top zone well before precursor
        quie_end = pre_start - int(quiescent_buffer_ms * 1e-3 / dt)
        for t in range(earliest, max(earliest + 1, quie_end), stride):
            if 0 <= t < len(features):
                X_list.append(features[t])
                y_list.append(0)
    else:
        # Non-disruptive: all flat-top is quiescent
        ft_end = int(ft_max / dt)
        for t in range(earliest, min(ft_end, len(features)), stride):
            X_list.append(features[t])
            y_list.append(0)

    if not X_list:
        return np.empty((0, features.shape[1])), np.empty(0, dtype=np.int8)
    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.int8)


@dataclass
class TrainedClassifier:
    model: object  # sklearn LogisticRegression
    feature_means: np.ndarray
    feature_stds: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES


def train_classifier(shots_with_features: list[tuple],
                      C_reg: float = 1.0) -> TrainedClassifier:
    """Train logistic regression on (shot, features_array) pairs."""
    from sklearn.linear_model import LogisticRegression

    Xs, ys = [], []
    for shot, feats in shots_with_features:
        dt = float(shot.time_s[1] - shot.time_s[0])
        X, y = sample_training_windows(shot, feats, dt)
        if len(X) > 0:
            Xs.append(X)
            ys.append(y)
    if not Xs:
        raise ValueError("No training windows extracted.")
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)

    mu = X_all.mean(axis=0)
    sigma = X_all.std(axis=0) + 1e-6
    X_norm = (X_all - mu) / sigma

    model = LogisticRegression(class_weight="balanced", C=C_reg,
                                max_iter=2000, solver="liblinear")
    model.fit(X_norm, y_all)
    return TrainedClassifier(model=model, feature_means=mu, feature_stds=sigma)


def classifier_alert(features: np.ndarray, clf: TrainedClassifier,
                      time_s: np.ndarray, dt: float,
                      score_threshold: float = 0.5,
                      min_duration_ms: float = 5.0):
    """Score every time-step with the classifier, find sustained alert."""
    from .catcher_v2 import ConsensusResult

    X_norm = (features - clf.feature_means) / clf.feature_stds
    probs = clf.model.predict_proba(X_norm)[:, 1]  # P(disrupting)
    above = probs >= score_threshold
    min_samples = max(1, int(min_duration_ms * 1e-3 / dt))
    T = len(probs)
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
        consensus_fraction=probs.astype(np.float32),
        consensus_count=(probs * 10).astype(np.int32),
        alert_active=alert,
        alert_start_idx=alert_start_idx,
        per_channel=[],
        channel_names=(),
    )
