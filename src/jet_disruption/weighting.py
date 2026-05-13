"""Per-channel weighting trained from labelled shots.

Vanilla unweighted consensus treats every channel equally — a single noisy
channel that fires frequently in quiescent shots pulls FAR up without
adding TPR. We compute a per-channel weight = log-likelihood ratio of
firing in pre-disruption windows vs flat-top quiescent windows, learned
from a training subset, then use a weighted-sum consensus rule:

    weighted_score(t) = sum_c w_c * is_novel_c(t) / sum_c w_c

A channel that fires often in precursor windows but rarely in quiescent
windows gets a positive weight; a channel that fires in both gets ~0;
a channel that fires only in quiescent gets a negative weight (capped at 0
for the consensus rule).

We learn weights on a held-out training subset of shots and evaluate
on the test subset — a clean train/test split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catcher_v2 import per_channel_scores, ConsensusResult


@dataclass
class WeightingResult:
    weights: np.ndarray            # (C,) learned per-channel weight
    fire_rate_precursor: np.ndarray  # (C,) P(is_novel in precursor window)
    fire_rate_quiescent: np.ndarray  # (C,) P(is_novel in quiescent window)
    channel_names: tuple[str, ...]


def _precursor_quiescent_windows(shot, ft_min, ft_max, t_disrupt,
                                  baseline_window_ms: float = 50.0,
                                  precursor_lead_ms: float = 100.0,
                                  post_buffer_ms: float = 20.0):
    """Return (precursor_slice, quiescent_slice) index ranges for a shot.

    precursor: last precursor_lead_ms before t_disrupt (only for disruptive)
    quiescent: from (ft_min + baseline_window_ms) to (t_disrupt - 200ms),
               i.e., the analyzable flat-top window before the precursor.
    """
    import math
    dt = float(shot.time_s[1] - shot.time_s[0])
    earliest_valid = int((ft_min + baseline_window_ms * 1e-3) / dt)
    if t_disrupt is not None and not math.isnan(t_disrupt):
        td_idx = int(t_disrupt / dt)
        prec_start = int((t_disrupt - precursor_lead_ms * 1e-3) / dt)
        prec_start = max(prec_start, earliest_valid)
        quie_end = max(prec_start - int(0.2 / dt), earliest_valid)
        prec = slice(prec_start, td_idx)
        quie = slice(earliest_valid, quie_end)
    else:
        # Non-disruptive: all of flat-top window is quiescent
        ft_end = int((ft_max + post_buffer_ms * 1e-3) / dt)
        quie = slice(earliest_valid, ft_end)
        prec = slice(0, 0)  # empty
    return prec, quie


def train_channel_weights(shots, channel_names: tuple[str, ...],
                            baseline_window_ms: float = 50.0,
                            assess_window_ms: float = 5.0,
                            z_threshold: float = 3.0,
                            precursor_lead_ms: float = 100.0) -> WeightingResult:
    """Compute per-channel weight = log( P(fire | precursor) /
                                          P(fire | quiescent) ).

    Smoothed with Laplace +1/+2 to avoid zero/inf.
    """
    C = len(channel_names)
    fire_prec = np.zeros(C, dtype=np.float64)
    fire_quie = np.zeros(C, dtype=np.float64)
    n_prec_samples = np.zeros(C, dtype=np.float64)
    n_quie_samples = np.zeros(C, dtype=np.float64)

    for s in shots:
        dt = float(s.time_s[1] - s.time_s[0])
        pcs = per_channel_scores(s.diagnostics, dt,
                                   baseline_window_ms=baseline_window_ms,
                                   assess_window_ms=assess_window_ms,
                                   z_threshold=z_threshold)
        import math as _math
        ft_min = getattr(s, "flat_top_tmin", None)
        ft_max = getattr(s, "flat_top_tmax", None)
        if (ft_min is None or ft_max is None
                or (isinstance(ft_min, float) and _math.isnan(ft_min))
                or (isinstance(ft_max, float) and _math.isnan(ft_max))):
            continue
        prec_slc, quie_slc = _precursor_quiescent_windows(
            s, ft_min, ft_max, s.t_disrupt_s,
            baseline_window_ms=baseline_window_ms,
            precursor_lead_ms=precursor_lead_ms,
        )
        for c in range(C):
            prec_arr = pcs[c].is_novel[prec_slc]
            quie_arr = pcs[c].is_novel[quie_slc]
            fire_prec[c] += int(prec_arr.sum())
            fire_quie[c] += int(quie_arr.sum())
            n_prec_samples[c] += len(prec_arr)
            n_quie_samples[c] += len(quie_arr)

    # Laplace smoothing
    p_prec = (fire_prec + 1) / (n_prec_samples + 2)
    p_quie = (fire_quie + 1) / (n_quie_samples + 2)
    weights = np.log(p_prec / p_quie)
    # Clip negative weights to 0: a channel that fires more in quiescent than
    # precursor should not actively pull the consensus down — just not
    # contribute.
    weights = np.maximum(weights, 0.0)
    # If all weights end up 0 (degenerate), fall back to uniform 1.0
    if weights.max() == 0:
        weights = np.ones(C, dtype=np.float64)

    return WeightingResult(
        weights=weights.astype(np.float32),
        fire_rate_precursor=(fire_prec / np.maximum(n_prec_samples, 1)).astype(np.float32),
        fire_rate_quiescent=(fire_quie / np.maximum(n_quie_samples, 1)).astype(np.float32),
        channel_names=channel_names,
    )


def weighted_consensus_alert(per_channel: list,
                              channel_names: tuple[str, ...],
                              weights: np.ndarray,
                              weighted_fraction_threshold: float = 0.3,
                              min_duration_ms: float = 5.0,
                              dt: float = 1e-3) -> ConsensusResult:
    """Weighted consensus rule.

    weighted_consensus(t) = sum_c (w_c * is_novel_c(t)) / sum_c w_c
    Alert fires when weighted_consensus(t) >= weighted_fraction_threshold
    for >= min_duration_ms.
    """
    T = len(per_channel[0].z_score)
    C = len(per_channel)
    w = np.asarray(weights, dtype=np.float32)
    w_total = float(w.sum()) + 1e-12

    is_novel = np.stack([s.is_novel for s in per_channel], axis=1)  # (T, C)
    weighted = (is_novel.astype(np.float32) * w[None, :]).sum(axis=1) / w_total
    above = weighted >= weighted_fraction_threshold

    min_samples = int(min_duration_ms * 1e-3 / dt)
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
        consensus_fraction=weighted,
        consensus_count=(weighted * C).astype(np.int32),  # for compat
        alert_active=alert,
        alert_start_idx=alert_start_idx,
        per_channel=per_channel,
        channel_names=channel_names,
    )
