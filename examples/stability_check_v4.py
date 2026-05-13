"""Stability check: re-run GBM v4 with 3 different RNG seeds for fold splits.

Confirms the 73%/17%/78ms TPR/FAR/TTD result is robust to which shots
land in which fold, not a fold-split lottery winner.
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
from jet_disruption.weighting import train_channel_weights
from jet_disruption.extended_features import compute_phase_aware_features
from jet_disruption.scoring import score_shot, score_dataset
from jet_disruption.catcher_v2 import ConsensusResult


def mask_to_analysis_window(result, time_s, flat_top_tmin, flat_top_tmax,
                              t_disrupt, baseline_window_ms=50.0,
                              post_flat_top_buffer_ms=20.0):
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


def sample_dense(shot, features, dt, baseline_window_ms=50.0,
                   precursor_lead_ms=50.0, precursor_min_ahead_ms=2.0,
                   quiescent_buffer_ms=150.0, stride_ms=2.0):
    ft_min = getattr(shot, "flat_top_tmin", None)
    ft_max = getattr(shot, "flat_top_tmax", None)
    if (ft_min is None or ft_max is None
            or (isinstance(ft_min, float) and math.isnan(ft_min))
            or (isinstance(ft_max, float) and math.isnan(ft_max))):
        return np.empty((0, features.shape[1])), np.empty(0, dtype=np.int8)
    earliest = int((ft_min + baseline_window_ms * 1e-3) / dt)
    stride = max(1, int(stride_ms * 1e-3 / dt))
    Xs, ys = [], []
    if shot.is_disruptive and shot.t_disrupt_s is not None:
        td = shot.t_disrupt_s
        pre_start = max(earliest, int((td - precursor_lead_ms * 1e-3) / dt))
        pre_end = int((td - precursor_min_ahead_ms * 1e-3) / dt)
        for t in range(pre_start, pre_end, stride):
            if 0 <= t < len(features):
                Xs.append(features[t]); ys.append(1)
        quie_end = pre_start - int(quiescent_buffer_ms * 1e-3 / dt)
        for t in range(earliest, max(earliest + 1, quie_end), stride):
            if 0 <= t < len(features):
                Xs.append(features[t]); ys.append(0)
    else:
        ft_end = len(features)
        for t in range(earliest, ft_end, stride):
            Xs.append(features[t]); ys.append(0)
    if not Xs:
        return np.empty((0, features.shape[1])), np.empty(0, dtype=np.int8)
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int8)


def gbm_alert(features, model, mu, sigma, time_s, dt,
               score_threshold=0.5, min_duration_ms=5.0):
    X = (features - mu) / sigma
    probs = model.predict_proba(X)[:, 1]
    above = probs >= score_threshold
    min_samples = max(1, int(min_duration_ms * 1e-3 / dt))
    T = len(probs); alert = np.zeros(T, dtype=bool)
    run_start = None; alert_start_idx = None
    for t in range(T):
        if above[t]:
            if run_start is None: run_start = t
            elif t - run_start + 1 >= min_samples:
                alert[t] = True
                if alert_start_idx is None: alert_start_idx = run_start
        else:
            run_start = None
    return ConsensusResult(
        consensus_fraction=probs.astype(np.float32),
        consensus_count=(probs * 10).astype(np.int32),
        alert_active=alert, alert_start_idx=alert_start_idx,
        per_channel=[], channel_names=(),
    )


def run_seed(seed, shots, warn_ms=200.0, score_thresholds=(0.85, 0.90, 0.95)):
    print(f"\n=== Seed {seed} ===", flush=True)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(shots))
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)
    fold_results = {f"p{thr:.2f}": [] for thr in score_thresholds}

    for fi, test_idx in enumerate(folds):
        train_idx = [i for i in indices if i not in set(test_idx)]
        train_shots = [shots[i] for i in train_idx]
        test_shots = [shots[i] for i in test_idx]
        print(f"  Fold {fi+1}/5: train={len(train_shots)}, test={len(test_shots)}",
              flush=True)

        wr = train_channel_weights(train_shots, MAST_CHANNELS,
                                    baseline_window_ms=50.0, assess_window_ms=5.0,
                                    z_threshold=3.0, precursor_lead_ms=100.0)

        feats_cache = {}
        for s in shots:
            dt = float(s.time_s[1] - s.time_s[0])
            f, _ = compute_phase_aware_features(
                s.diagnostics, dt, wr.weights,
                channel_names=MAST_CHANNELS, time_s=s.time_s,
                ramp_up_tmax=getattr(s, "ramp_up_tmax", None),
                flat_top_tmin=getattr(s, "flat_top_tmin", None),
                flat_top_tmax=getattr(s, "flat_top_tmax", None),
                scales_ms=(50.0, 200.0, 500.0),
            )
            feats_cache[s.shot_id] = f

        Xs, ys = [], []
        for s in train_shots:
            X, y = sample_dense(s, feats_cache[s.shot_id],
                                  float(s.time_s[1] - s.time_s[0]))
            if len(X) > 0:
                Xs.append(X); ys.append(y)
        X_all = np.concatenate(Xs); y_all = np.concatenate(ys)
        n_pos = int((y_all == 1).sum()); n_neg = int((y_all == 0).sum())
        mu = X_all.mean(axis=0); sigma = X_all.std(axis=0) + 1e-6
        sigma = np.maximum(sigma, 1e-3)
        X_norm = (X_all - mu) / sigma

        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.05,
            subsample=0.7, min_samples_split=20, min_samples_leaf=10,
            random_state=42,
        )
        cls_weights = np.where(y_all == 1,
                                 len(y_all) / (2 * n_pos + 1e-9),
                                 len(y_all) / (2 * n_neg + 1e-9))
        model.fit(X_norm, y_all, sample_weight=cls_weights)

        for s in test_shots:
            dt = float(s.time_s[1] - s.time_s[0])
            feats = feats_cache[s.shot_id]
            for thr in score_thresholds:
                r = gbm_alert(feats, model, mu, sigma, s.time_s, dt,
                               score_threshold=thr, min_duration_ms=5.0)
                r = mask_to_analysis_window(r, s.time_s,
                                              s.flat_top_tmin, s.flat_top_tmax,
                                              s.t_disrupt_s)
                fold_results[f"p{thr:.2f}"].append(
                    score_shot(s, r, warning_window_ms=warn_ms))

    out = {}
    for thr in score_thresholds:
        ds = score_dataset(fold_results[f"p{thr:.2f}"])
        out[f"p{thr:.2f}"] = {"tpr": ds.tpr, "far": ds.far,
                               "median_ttd_ms": ds.median_ttd_ms,
                               "n_tp": ds.n_tp, "n_fn": ds.n_fn,
                               "n_fp": ds.n_fp, "n_tn": ds.n_tn}
        ttd = (f"{ds.median_ttd_ms:.1f}" if ds.median_ttd_ms is not None else "n/a")
        print(f"  p={thr:.2f}: TPR={ds.tpr*100:.0f}% FAR={ds.far*100:.0f}% TTD={ttd}ms",
              flush=True)
    return out


def main():
    print("MAST GBM v4 Stability Check (3 RNG seeds)", flush=True)
    print("=" * 70, flush=True)
    shots = load_mast_dataset("data/T_lead_0ms")
    print(f"  {len(shots)} shots", flush=True)

    seeds = [42, 7, 2024]
    score_thresholds = (0.85, 0.90, 0.95)
    all_results = {}
    for seed in seeds:
        all_results[seed] = run_seed(seed, shots, score_thresholds=score_thresholds)

    # Aggregate stats
    print()
    print("Stability summary (mean +/- std across seeds):", flush=True)
    print(f"{'threshold':>12s} | {'TPR':>16s} | {'FAR':>16s} | {'TTD (ms)':>16s}", flush=True)
    print("-" * 70, flush=True)
    summary = []
    for thr in score_thresholds:
        tprs = [all_results[s][f"p{thr:.2f}"]["tpr"] for s in seeds]
        fars = [all_results[s][f"p{thr:.2f}"]["far"] for s in seeds]
        ttds = [all_results[s][f"p{thr:.2f}"]["median_ttd_ms"] for s in seeds
                if all_results[s][f"p{thr:.2f}"]["median_ttd_ms"] is not None]
        m_tpr, s_tpr = np.mean(tprs)*100, np.std(tprs)*100
        m_far, s_far = np.mean(fars)*100, np.std(fars)*100
        m_ttd = np.mean(ttds); s_ttd = np.std(ttds)
        print(f"  p={thr:.2f}      | {m_tpr:.1f}% +/- {s_tpr:.1f}% | "
              f"{m_far:.1f}% +/- {s_far:.1f}% | "
              f"{m_ttd:.1f} +/- {s_ttd:.1f}", flush=True)
        summary.append({
            "threshold": thr,
            "tpr_mean": m_tpr, "tpr_std": s_tpr,
            "far_mean": m_far, "far_std": s_far,
            "ttd_mean": m_ttd, "ttd_std": s_ttd,
        })

    out = Path(__file__).parent / "mast_stability_results.json"
    out.write_text(json.dumps({
        "dataset": "MAST disruption (Sharma 2025)",
        "seeds": seeds,
        "per_seed": all_results,
        "summary": summary,
    }, indent=2, default=str))
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
