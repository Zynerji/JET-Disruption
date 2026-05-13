"""5-fold CV benchmark of the A+B+C hybrid gate on MAST data.

Compares:
  - unweighted consensus (baseline)
  - weighted consensus (per-channel LLR weights)
  - hybrid 2-of-3 (A coherence + B Kuramoto + C Hamiltonian)
  - hybrid 3-of-3 (stricter)

Output: results table + per-shot timing for hybrid alerts.
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
    print("MAST Hybrid Gate Benchmark (A+B+C, 5-fold CV)")
    print("=" * 70)

    print("Loading MAST shots...")
    t0 = time.time()
    shots = load_mast_dataset("data/T_lead_0ms")
    print(f"  {len(shots)} shots in {time.time() - t0:.1f}s")

    print()
    print("Per-channel rolling z-scores...")
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
    # Tested configurations
    configs = [
        ("unweighted_frac0.50", "unweighted", 0.50),
        ("weighted_frac0.30",  "weighted",   0.30),
        ("weighted_frac0.50",  "weighted",   0.50),
        ("hybrid_K2",          "hybrid",     2),
        ("hybrid_K3",          "hybrid",     3),
    ]
    fold_results = {name: [] for name, *_ in configs}
    fold_weights = []

    for fi, test_idx in enumerate(folds):
        train_shots = [shots[i] for i in indices if i not in set(test_idx)]
        test_shots = [shots[i] for i in test_idx]
        print(f"  Fold {fi+1}/5: train={len(train_shots)}, test={len(test_shots)}")
        wr = train_channel_weights(train_shots, MAST_CHANNELS,
                                    baseline_window_ms=50.0,
                                    assess_window_ms=5.0,
                                    z_threshold=3.0,
                                    precursor_lead_ms=100.0)
        fold_weights.append(wr.weights.tolist())

        for s in test_shots:
            dt = float(s.time_s[1] - s.time_s[0])
            pcs = pre[s.shot_id]

            # Unweighted at frac=0.50
            ru = consensus_alert(pcs, s.diagnostic_names,
                                  fraction_threshold=0.50,
                                  min_duration_ms=5.0, dt=dt)
            ru = mask_to_analysis_window(ru, s.time_s,
                                          s.flat_top_tmin, s.flat_top_tmax,
                                          s.t_disrupt_s)
            fold_results["unweighted_frac0.50"].append(
                score_shot(s, ru, warning_window_ms=warn_ms))

            # Weighted at frac=0.30 (best lift)
            rw30 = weighted_consensus_alert(pcs, s.diagnostic_names,
                                              weights=wr.weights,
                                              weighted_fraction_threshold=0.30,
                                              min_duration_ms=5.0, dt=dt)
            rw30 = mask_to_analysis_window(rw30, s.time_s,
                                              s.flat_top_tmin, s.flat_top_tmax,
                                              s.t_disrupt_s)
            fold_results["weighted_frac0.30"].append(
                score_shot(s, rw30, warning_window_ms=warn_ms))

            # Weighted at frac=0.50
            rw50 = weighted_consensus_alert(pcs, s.diagnostic_names,
                                              weights=wr.weights,
                                              weighted_fraction_threshold=0.50,
                                              min_duration_ms=5.0, dt=dt)
            rw50 = mask_to_analysis_window(rw50, s.time_s,
                                              s.flat_top_tmin, s.flat_top_tmax,
                                              s.t_disrupt_s)
            fold_results["weighted_frac0.50"].append(
                score_shot(s, rw50, warning_window_ms=warn_ms))

            # Hybrid K=2
            rh2 = hybrid_alert(pcs, s.diagnostic_names,
                                weights=wr.weights, dt=dt,
                                k_of_3=2, min_duration_ms=5.0)
            rh2 = mask_to_analysis_window(rh2, s.time_s,
                                            s.flat_top_tmin, s.flat_top_tmax,
                                            s.t_disrupt_s)
            fold_results["hybrid_K2"].append(
                score_shot(s, rh2, warning_window_ms=warn_ms))

            # Hybrid K=3
            rh3 = hybrid_alert(pcs, s.diagnostic_names,
                                weights=wr.weights, dt=dt,
                                k_of_3=3, min_duration_ms=5.0)
            rh3 = mask_to_analysis_window(rh3, s.time_s,
                                            s.flat_top_tmin, s.flat_top_tmax,
                                            s.t_disrupt_s)
            fold_results["hybrid_K3"].append(
                score_shot(s, rh3, warning_window_ms=warn_ms))

    print()
    print(f"Results (warning window {warn_ms:.0f} ms, 5-fold CV)")
    print("-" * 70)
    summary = []
    for name, *_ in configs:
        ds = score_dataset(fold_results[name])
        ttd = (f"{ds.median_ttd_ms:.1f}" if ds.median_ttd_ms is not None
               else "n/a")
        print(f"  {name:>26s}: TPR={ds.tpr*100:>4.0f}% "
              f"FAR={ds.far*100:>4.0f}%  TTD_med={ttd}ms"
              f"  ({ds.n_tp}/{ds.n_tp+ds.n_fn} TP, "
              f"{ds.n_fp}/{ds.n_fp+ds.n_tn} FP)")
        summary.append({
            "name": name,
            "tpr": ds.tpr, "far": ds.far,
            "median_ttd_ms": ds.median_ttd_ms,
            "n_tp": ds.n_tp, "n_fn": ds.n_fn,
            "n_fp": ds.n_fp, "n_tn": ds.n_tn,
        })

    out = Path(__file__).parent / "mast_hybrid_results.json"
    out.write_text(json.dumps({
        "dataset": "MAST disruption detection (Sharma 2025, Zenodo 16032053)",
        "n_shots": len(shots),
        "n_folds": 5,
        "warning_window_ms": warn_ms,
        "channels": list(MAST_CHANNELS),
        "fold_weights": fold_weights,
        "summary": summary,
    }, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
