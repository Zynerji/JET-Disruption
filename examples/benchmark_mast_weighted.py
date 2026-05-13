"""Weighted-consensus benchmark on real MAST data.

5-fold cross-validation: in each fold, learn per-channel weights from the
training shots, evaluate the weighted catcher on the test shots, aggregate
TPR/FAR across folds.

Compare against the unweighted baseline to quantify the weighting lift.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from jet_disruption.mast_loader import load_mast_dataset, MAST_CHANNELS
from jet_disruption.catcher_v2 import per_channel_scores, consensus_alert
from jet_disruption.weighting import (
    train_channel_weights, weighted_consensus_alert,
)
from jet_disruption.scoring import score_shot, score_dataset


def mask_to_analysis_window(result, time_s, flat_top_tmin, flat_top_tmax,
                              t_disrupt,
                              baseline_window_ms: float = 50.0,
                              post_flat_top_buffer_ms: float = 20.0):
    if (flat_top_tmin is None or flat_top_tmin <= 0
            or math.isnan(flat_top_tmin) or flat_top_tmax is None
            or math.isnan(flat_top_tmax)):
        result.alert_start_idx = None
        return result
    if result.alert_start_idx is None:
        return result
    dt = float(time_s[1] - time_s[0])
    earliest = int((flat_top_tmin + baseline_window_ms * 1e-3) / dt)
    latest = int((flat_top_tmax + post_flat_top_buffer_ms * 1e-3) / dt)
    if t_disrupt is not None and not math.isnan(t_disrupt):
        latest = min(latest, int(t_disrupt / dt))
    new_start = None
    for i in range(earliest, min(latest + 1, len(result.alert_active))):
        if result.alert_active[i]:
            new_start = i
            break
    result.alert_start_idx = new_start
    return result


def main():
    print("MAST Weighted-Consensus Benchmark (5-fold CV)")
    print("=" * 70)

    print("Loading MAST shots ...")
    t0 = time.time()
    shots = load_mast_dataset("data/T_lead_0ms")
    print(f"  {len(shots)} shots in {time.time() - t0:.1f}s")
    n_disr = sum(1 for s in shots if s.is_disruptive)
    n_quie = sum(1 for s in shots if not s.is_disruptive)
    print(f"  disruptive: {n_disr}, non-disruptive: {n_quie}")

    print()
    print("Pre-computing per-channel scores for all shots ...")
    t0 = time.time()
    pre = {}
    for s in shots:
        dt = float(s.time_s[1] - s.time_s[0])
        pre[s.shot_id] = per_channel_scores(
            s.diagnostics, dt,
            baseline_window_ms=50.0, assess_window_ms=5.0, z_threshold=3.0,
        )
    print(f"  {time.time() - t0:.1f}s")

    # 5-fold CV
    print()
    print("5-fold CV ...")
    rng = np.random.default_rng(42)
    indices = np.arange(len(shots))
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    frac_thresholds = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
    warn_ms = 200.0

    # Aggregate across folds: collect every test-shot score
    fold_weights = []
    fold_per_threshold = {ft: [] for ft in frac_thresholds}
    unweighted_per_threshold = {ft: [] for ft in frac_thresholds}

    for fi, test_idx in enumerate(folds):
        train_shots = [shots[i] for i in indices if i not in set(test_idx)]
        test_shots = [shots[i] for i in test_idx]
        print(f"  Fold {fi+1}/5: train={len(train_shots)}, test={len(test_shots)}")
        # Train weights
        wr = train_channel_weights(train_shots, MAST_CHANNELS,
                                    baseline_window_ms=50.0,
                                    assess_window_ms=5.0,
                                    z_threshold=3.0,
                                    precursor_lead_ms=100.0)
        fold_weights.append(wr.weights.tolist())
        # Evaluate at every threshold
        for ft in frac_thresholds:
            for s in test_shots:
                dt = float(s.time_s[1] - s.time_s[0])
                # Weighted
                rw = weighted_consensus_alert(
                    pre[s.shot_id], s.diagnostic_names,
                    weights=wr.weights,
                    weighted_fraction_threshold=ft,
                    min_duration_ms=5.0, dt=dt,
                )
                rw = mask_to_analysis_window(rw, s.time_s,
                                              s.flat_top_tmin, s.flat_top_tmax,
                                              s.t_disrupt_s)
                fold_per_threshold[ft].append(
                    score_shot(s, rw, warning_window_ms=warn_ms))
                # Unweighted (for comparison, same threshold scale)
                ru = consensus_alert(
                    pre[s.shot_id], s.diagnostic_names,
                    fraction_threshold=ft,
                    min_duration_ms=5.0, dt=dt,
                )
                ru = mask_to_analysis_window(ru, s.time_s,
                                              s.flat_top_tmin, s.flat_top_tmax,
                                              s.t_disrupt_s)
                unweighted_per_threshold[ft].append(
                    score_shot(s, ru, warning_window_ms=warn_ms))

    # Aggregate TPR/FAR
    print()
    print(f"Results (warn={warn_ms:.0f} ms)")
    print(f"{'frac':>6} | {'unweighted':>20s} | {'weighted':>20s} | weighting_lift")
    print("-" * 80)
    summary = []
    for ft in frac_thresholds:
        ds_w = score_dataset(fold_per_threshold[ft])
        ds_u = score_dataset(unweighted_per_threshold[ft])
        lift_tpr = (ds_w.tpr - ds_u.tpr) * 100
        lift_far = (ds_u.far - ds_w.far) * 100  # positive = weighting reduced FAR
        u_str = f"TPR={ds_u.tpr*100:.0f}% FAR={ds_u.far*100:.0f}%"
        w_str = f"TPR={ds_w.tpr*100:.0f}% FAR={ds_w.far*100:.0f}%"
        lift_str = f"dTPR={lift_tpr:+.0f}pp dFAR_reduced={lift_far:+.0f}pp"
        print(f"{ft:>6.2f} | {u_str:>20s} | {w_str:>20s} | {lift_str}")
        summary.append({
            "fraction_threshold": ft,
            "unweighted": {"tpr": ds_u.tpr, "far": ds_u.far,
                           "median_ttd_ms": ds_u.median_ttd_ms,
                           "n_tp": ds_u.n_tp, "n_fp": ds_u.n_fp,
                           "n_fn": ds_u.n_fn, "n_tn": ds_u.n_tn},
            "weighted":   {"tpr": ds_w.tpr, "far": ds_w.far,
                           "median_ttd_ms": ds_w.median_ttd_ms,
                           "n_tp": ds_w.n_tp, "n_fp": ds_w.n_fp,
                           "n_fn": ds_w.n_fn, "n_tn": ds_w.n_tn},
        })

    # Report mean weights across folds
    weights_mean = np.mean(np.asarray(fold_weights), axis=0)
    print()
    print("Mean learned weights across 5 folds (sorted by importance):")
    order = np.argsort(-weights_mean)
    for i in order:
        print(f"  {MAST_CHANNELS[i]:>30s}: weight={weights_mean[i]:.3f}")

    out = Path(__file__).parent / "mast_weighted_results.json"
    out.write_text(json.dumps({
        "dataset": "MAST disruption detection (Sharma 2025, Zenodo 16032053)",
        "n_shots": len(shots),
        "n_disruptive": n_disr,
        "n_non_disruptive": n_quie,
        "n_folds": 5,
        "config": {"baseline_window_ms": 50.0, "assess_window_ms": 5.0,
                   "z_threshold": 3.0, "min_duration_ms": 5.0,
                   "warning_window_ms": warn_ms,
                   "precursor_lead_ms": 100.0},
        "channels": list(MAST_CHANNELS),
        "fold_weights": fold_weights,
        "mean_weights": weights_mean.tolist(),
        "summary": summary,
    }, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
