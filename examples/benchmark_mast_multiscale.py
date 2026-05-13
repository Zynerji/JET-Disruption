"""Multi-scale + derivative learned classifier on MAST.

3 baseline scales (50/200/500 ms) × 4 features each + 2 derivatives = 14.
Drops the dead gates A (coherence) and C (Hamiltonian) from the
single-scale benchmark.
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
from jet_disruption.catcher_v2 import per_channel_scores
from jet_disruption.weighting import train_channel_weights
from jet_disruption.multiscale_features import (
    compute_multiscale_features, MULTISCALE_FEATURE_NAMES,
)
from jet_disruption.learned_classifier import (
    train_classifier, classifier_alert, sample_training_windows,
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
    print("MAST Multi-scale Classifier (3 scales x 4 feats + 2 derivs)")
    print("=" * 70)

    print("Loading MAST shots...")
    t0 = time.time()
    shots = load_mast_dataset("data/T_lead_0ms")
    print(f"  {len(shots)} shots in {time.time() - t0:.1f}s")

    rng = np.random.default_rng(42)
    indices = np.arange(len(shots))
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    warn_ms = 200.0
    score_thresholds = (0.50, 0.70, 0.85, 0.95)
    fold_results = {f"multi_p{thr:.2f}": [] for thr in score_thresholds}
    fold_coefs = []

    for fi, test_idx in enumerate(folds):
        train_idx = [i for i in indices if i not in set(test_idx)]
        train_shots = [shots[i] for i in train_idx]
        test_shots = [shots[i] for i in test_idx]
        print(f"\n  Fold {fi+1}/5: train={len(train_shots)}, test={len(test_shots)}")

        # Per-channel weights (uses default 50 ms scale)
        t1 = time.time()
        wr = train_channel_weights(train_shots, MAST_CHANNELS,
                                    baseline_window_ms=50.0,
                                    assess_window_ms=5.0,
                                    z_threshold=3.0,
                                    precursor_lead_ms=100.0)
        print(f"    weights: {time.time() - t1:.1f}s")

        # Compute multi-scale features for ALL shots in this fold
        t1 = time.time()
        feats_cache = {}
        for s in shots:
            dt = float(s.time_s[1] - s.time_s[0])
            feats_cache[s.shot_id] = compute_multiscale_features(
                s.diagnostics, dt, wr.weights,
                scales_ms=(50.0, 200.0, 500.0),
            )
        print(f"    multi-scale features: {time.time() - t1:.1f}s")

        # Train classifier (LogReg balanced, C=1)
        t1 = time.time()
        clf = train_classifier(
            [(s, feats_cache[s.shot_id]) for s in train_shots],
            C_reg=1.0,
        )
        fold_coefs.append(clf.model.coef_[0].tolist())
        print(f"    classifier: {time.time() - t1:.1f}s")

        # Evaluate at multiple thresholds
        for s in test_shots:
            dt = float(s.time_s[1] - s.time_s[0])
            feats = feats_cache[s.shot_id]
            for thr in score_thresholds:
                r = classifier_alert(feats, clf, s.time_s, dt,
                                       score_threshold=thr,
                                       min_duration_ms=5.0)
                r = mask_to_analysis_window(r, s.time_s,
                                              s.flat_top_tmin, s.flat_top_tmax,
                                              s.t_disrupt_s)
                fold_results[f"multi_p{thr:.2f}"].append(
                    score_shot(s, r, warning_window_ms=warn_ms))

    print()
    print(f"Results (5-fold CV, warning window {warn_ms:.0f} ms)")
    print("-" * 70)
    summary = []
    for thr in score_thresholds:
        ds = score_dataset(fold_results[f"multi_p{thr:.2f}"])
        ttd = (f"{ds.median_ttd_ms:.1f}" if ds.median_ttd_ms is not None
               else "n/a")
        print(f"  multi-scale p={thr:.2f}: TPR={ds.tpr*100:>4.0f}% "
              f"FAR={ds.far*100:>4.0f}%  TTD_med={ttd}ms  "
              f"({ds.n_tp}/{ds.n_tp+ds.n_fn} TP, "
              f"{ds.n_fp}/{ds.n_fp+ds.n_tn} FP)")
        summary.append({
            "threshold": thr, "tpr": ds.tpr, "far": ds.far,
            "median_ttd_ms": ds.median_ttd_ms,
            "n_tp": ds.n_tp, "n_fn": ds.n_fn,
            "n_fp": ds.n_fp, "n_tn": ds.n_tn,
        })

    print()
    print("Mean learned coefficients across 5 folds:")
    coef_mean = np.mean(np.asarray(fold_coefs), axis=0)
    order = np.argsort(-np.abs(coef_mean))
    for i in order:
        print(f"  {MULTISCALE_FEATURE_NAMES[i]:>26s}: coef={coef_mean[i]:+.3f}")

    out = Path(__file__).parent / "mast_multiscale_results.json"
    out.write_text(json.dumps({
        "dataset": "MAST disruption (Sharma 2025, Zenodo 16032053)",
        "n_shots": len(shots),
        "n_folds": 5,
        "scales_ms": [50.0, 200.0, 500.0],
        "feature_names": list(MULTISCALE_FEATURE_NAMES),
        "fold_coefs": fold_coefs,
        "mean_coefs": coef_mean.tolist(),
        "summary": summary,
    }, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
