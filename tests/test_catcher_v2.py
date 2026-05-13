"""Unit tests for catcher_v2 rolling z-score + consensus filter."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jet_disruption.catcher_v2 import (
    robust_zscore_rolling, per_channel_scores,
    consensus_alert, run_catcher,
)


def test_zscore_quiescent_returns_zero():
    """Pure Gaussian noise should give |z| << 3 almost everywhere."""
    rng = np.random.default_rng(0)
    x = 1.0 + 0.1 * rng.standard_normal(2000)
    z = robust_zscore_rolling(x, baseline_window=200, assess_window=20)
    # Only last 1800 samples are valid; check max |z| in the tail
    valid = z[200:]
    assert valid.std() < 1.5, f"quiescent z-std too high: {valid.std():.3f}"
    assert (np.abs(valid) > 4.0).sum() < 10  # tail count < 10/1800


def test_zscore_detects_step():
    """A clear step should produce |z| >> 3 at and after the step."""
    rng = np.random.default_rng(1)
    x = np.concatenate([
        1.0 + 0.05 * rng.standard_normal(800),
        2.5 + 0.05 * rng.standard_normal(200),
    ])
    z = robust_zscore_rolling(x, baseline_window=200, assess_window=20)
    # z should spike at the step (around idx 800)
    post_step = z[810:990]
    assert post_step.max() > 5.0, f"step not detected: max z={post_step.max():.2f}"


def test_zscore_output_shape():
    x = np.ones(1000)
    z = robust_zscore_rolling(x, 200, 20)
    assert z.shape == x.shape
    assert z[:200].max() == 0  # no z computed in baseline window


def test_zscore_no_off_by_one():
    """Regression test for the off-by-one bug in vectorised version."""
    for T in (500, 1000, 5000, 10000):
        x = np.zeros(T)
        z = robust_zscore_rolling(x, 200, 20)
        assert z.shape == (T,), f"shape mismatch at T={T}: got {z.shape}"


def test_consensus_filter_requires_duration():
    """Consensus must persist for min_duration to fire."""
    T = 1000
    C = 4
    rng = np.random.default_rng(2)
    diag = 0.01 * rng.standard_normal((T, C))
    # Spike all channels at one sample
    diag[500, :] = 10.0
    dt = 1e-3

    pcs = per_channel_scores(diag, dt,
                              baseline_window_ms=200, assess_window_ms=20,
                              z_threshold=3.0)
    # Single-sample spike: with min_duration_ms=10 (10 samples), should NOT fire
    r = consensus_alert(pcs, ("a", "b", "c", "d"),
                         fraction_threshold=0.5,
                         min_duration_ms=10.0, dt=dt)
    assert r.alert_start_idx is None, "single-sample spike fired the alert"


def test_consensus_filter_fires_sustained():
    T = 1000
    C = 4
    rng = np.random.default_rng(3)
    diag = 0.01 * rng.standard_normal((T, C))
    # Sustained 30-ms spike across 3 of 4 channels
    diag[500:530, :3] = 5.0
    dt = 1e-3
    pcs = per_channel_scores(diag, dt,
                              baseline_window_ms=200, assess_window_ms=20,
                              z_threshold=3.0)
    r = consensus_alert(pcs, ("a", "b", "c", "d"),
                         fraction_threshold=0.5,
                         min_duration_ms=10.0, dt=dt)
    assert r.alert_start_idx is not None
    assert r.alert_start_idx >= 500
    assert r.alert_start_idx < 540


def test_run_catcher_end_to_end():
    """Generate disruptive shot, expect alert."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from jet_disruption.synthetic_shot import generate_shot
    shot = generate_shot(0, is_disruptive=True, seed=10)
    r = run_catcher(shot.diagnostics, shot.time_s,
                     channel_names=shot.diagnostic_names)
    assert r.alert_start_idx is not None
    alert_t = shot.time_s[r.alert_start_idx]
    assert alert_t <= shot.t_disrupt_s, "alert after disruption — predictive failed"
    assert (shot.t_disrupt_s - alert_t) * 1000 < 200, "alert > 200ms ahead?"
    assert r.consensus_count.max() >= 4, "too few channels co-firing"


def test_run_catcher_quiescent_no_alert():
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from jet_disruption.synthetic_shot import generate_shot
    shot = generate_shot(0, is_disruptive=False, seed=20)
    r = run_catcher(shot.diagnostics, shot.time_s,
                     channel_names=shot.diagnostic_names)
    assert r.alert_start_idx is None, (
        f"quiescent shot fired alert at idx {r.alert_start_idx}, "
        f"max consensus {r.consensus_count.max()}/14"
    )
