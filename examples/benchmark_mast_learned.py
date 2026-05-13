"""5-fold CV benchmark of the learned-feature classifier on MAST.

Pipeline (per fold):
  1. Train per-channel LLR weights on training shots.
  2. Compute (T, 7) feature matrix for every shot using those weights.
  3. Sample per-window training rows: positive = precursor, negative = quiescent.
  4. Train logistic regression with class_weight='balanced'.
  5. Apply classifier on test shots; alert when P(disrupting) > threshold
      for sustained duration.

Compares against the best hand-built configs:
  - weighted_frac0.50 (best balance from weighting benchmark)
  - hybrid_K2
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
from jet_disruption.hybrid_gate import hybrid_alert
from jet_disruption.learned_classifier import (
    compute_features, train_classifier, classifier_alert,
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
    print("MAST Learned-Classifier Benchmark (5-fold CV)")
    print("=" * 70)

    print("Loading MAST shots...")
    t0 = time.time()
    shots = load_mast_dataset("data/T_lead_0ms")
    print(f"  {len(shots)} shots in {time.time() - t0:.1f}s")

    print()
    print("Per-channel rolling z-scores (one-time)...")
    t0 = time.time()
    pre = {}
    for s in shots:
        dt = float(s.time_s[1] - s.time_s[0])
        pre[s.shot_id] = per_channel_scores(
            s.diagnostics, dt,
            baseline_window_ms=50.0, assess_window_ms=5.0, z_threshold=3.0,
        )
    print(f"  {time.time() - t0:.1f}s")

    rng = np.random.default_rng(42)
    indices = np.arange(len(shots))
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    warn_ms = 200.0
    score_thresholds = (0.30, 0.50, 0.70, 0.85, 0.95)

    fold_results = {f"learned_p{thr:.2f}": [] for thr in score_thresholds}
    fold_results["weighted_frac0.50"] = []
    fold_results["hybrid_K2"] = []

    fold_coefs = []

    for fi, test_idx in enumerate(folds):
        train_idx = [i for i in indices if i not in set(test_idx)]
        train_shots = [shots[i] for i in train_idx]
        test_shots = [shots[i] for i in test_idx]
        print(f"  Fold {fi+1}/5: train={len(train_shots)}, test={len(test_shots)}")

        # 1. Per-channel weights
        wr = train_channel_weights(train_shots, MAST_CHANNELS,
                                    baseline_window_ms=50.0,
                                    assess_window_ms=5.0,
                                    z_threshold=3.0,
                                    precursor_lead_ms=100.0)

        # 2. Compute features for ALL shots (train + test) once per fold
        t_feat0 = time.time()
        feats_cache = {}
        for s in shots:
            dt = float(s.time_s[1] - s.time_s[0])
            feats_cache[s.shot_id] = compute_features(
                pre[s.shot_id], wr.weights, dt,
                a_coherence_window=5, a_min_agreeing=3,
                b_baseline_samples=300, c_baseline_samples=300,
            )
        print(f"    features: {time.time() - t_feat0:.1f}s")

        # 3-4. Train classifier
        t_clf0 = time.time()
        clf = train_classifier(
            [(s, feats_cache[s.shot_id]) for s in train_shots],
            C_reg=1.0,
        )
        fold_coefs.append(clf.model.coef_[0].tolist())
        print(f"    classifier: {time.time() - t_clf0:.1f}s, "
              f"coefs={[f'{c:+.2f}' for c in clf.model.coef_[0]]}")

        # 5. Evaluate on test shots
        for s in test_shots:
            dt = float(s.time_s[1] - s.time_s[0])
            feats = feats_cache[s.shot_id]
            pcs = pre[s.shot_id]

            # Learned classifier at multiple thresholds
            for thr in score_thresholds:
                ra = classifier_alert(feats, clf, s.time_s, dt,
                                       score_threshold=thr,
                                       min_duration_ms=5.0)
                ra = mask_to_analysis_window(ra, s.time_s,
                                              s.flat_top_tmin, s.flat_top_tmax,
                                              s.t_disrupt_s)
                fold_results[f"learned_p{thr:.2f}"].append(
                    score_shot(s, ra, warning_window_ms=warn_ms))

            # Baselines for context
            rw = weighted_consensus_alert(pcs, s.diagnostic_names,
                                            weights=wr.weights,
                                            weighted_fraction_threshold=0.50,
                                            min_duration_ms=5.0, dt=dt)
            rw = mask_to_analysis_window(rw, s.time_s,
                                          s.flat_top_tmin, s.flat_top_tmax,
                                          s.t_disrupt_s)
            fold_results["weighted_frac0.50"].append(
                score_shot(s, rw, warning_window_ms=warn_ms))

            rh = hybrid_alert(pcs, s.diagnostic_names,
                                weights=wr.weights, dt=dt,
                                k_of_3=2, min_duration_ms=5.0)
            rh = mask_to_analysis_window(rh, s.time_s,
                                          s.flat_top_tmin, s.flat_top_tmax,
                                          s.t_disrupt_s)
            fold_results["hybrid_K2"].append(
                score_shot(s, rh, warning_window_ms=warn_ms))

    print()
    print(f"Results (5-fold CV, warning window {warn_ms:.0f} ms)")
    print("-" * 70)
    summary = []
    for name in (["weighted_frac0.50", "hybrid_K2"]
                  + [f"learned_p{thr:.2f}" for thr in score_thresholds]):
        ds = score_dataset(fold_results[name])
        ttd = (f"{ds.median_ttd_ms:.1f}" if ds.median_ttd_ms is not None
               else "n/a")
        print(f"  {name:>22s}: TPR={ds.tpr*100:>4.0f}% "
              f"FAR={ds.far*100:>4.0f}%  TTD_med={ttd}ms  "
              f"({ds.n_tp}/{ds.n_tp+ds.n_fn} TP, "
              f"{ds.n_fp}/{ds.n_fp+ds.n_tn} FP)")
        summary.append({
            "name": name, "tpr": ds.tpr, "far": ds.far,
            "median_ttd_ms": ds.median_ttd_ms,
            "n_tp": ds.n_tp, "n_fn": ds.n_fn,
            "n_fp": ds.n_fp, "n_tn": ds.n_tn,
        })

    print()
    print("Mean learned coefficients across 5 folds:")
    coef_mean = np.mean(np.asarray(fold_coefs), axis=0)
    from jet_disruption.learned_classifier import FEATURE_NAMES
    order = np.argsort(-np.abs(coef_mean))
    for i in order:
        print(f"  {FEATURE_NAMES[i]:>22s}: coef={coef_mean[i]:+.3f}")

    out = Path(__file__).parent / "mast_learned_results.json"
    out.write_text(json.dumps({
        "dataset": "MAST disruption (Sharma 2025, Zenodo 16032053)",
        "n_shots": len(shots),
        "n_folds": 5,
        "feature_names": list(FEATURE_NAMES),
        "fold_coefs": fold_coefs,
        "mean_coefs": coef_mean.tolist(),
        "summary": summary,
    }, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
