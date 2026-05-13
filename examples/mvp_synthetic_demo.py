"""MVP: synthetic JET-like shots scored by the cross-diagnostic consensus
catcher in two regimes:

  CLEAN: canonical strong precursor for every disruptive shot, no glitches
         in quiescent shots. Validates the method on the easy case.
  MIXED: precursor strength uniformly in [0.4, 1.0] (weak precursors
         included); 0-3 random noise glitches injected in quiescent shots
         (mimics JET artefacts that fool naive single-channel detectors).
         Harder, more realistic.

Sweeps fraction_threshold to map the TPR/FAR trade-off in each mode.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from jet_disruption.synthetic_shot import generate_dataset
from jet_disruption.catcher_v2 import (
    per_channel_scores, consensus_alert,
)
from jet_disruption.scoring import score_shot, score_dataset


def run_mode(shots, label: str, frac_thresholds, warning_window_ms: float):
    print(f"\n=== {label} (n={len(shots)}) ===")
    # Pre-compute per-channel scores once per shot
    t0 = time.time()
    pre = {}
    for s in shots:
        dt = float(s.time_s[1] - s.time_s[0])
        pre[s.shot_id] = per_channel_scores(s.diagnostics, dt,
                                              baseline_window_ms=200.0,
                                              assess_window_ms=20.0,
                                              z_threshold=3.0)
    print(f"  per-channel scoring: {time.time() - t0:.1f}s")

    sweep = []
    for ft in frac_thresholds:
        scores_ft = []
        for s in shots:
            r = consensus_alert(pre[s.shot_id], s.diagnostic_names,
                                 fraction_threshold=ft,
                                 min_duration_ms=10.0,
                                 dt=float(s.time_s[1] - s.time_s[0]))
            scores_ft.append(score_shot(s, r,
                                         warning_window_ms=warning_window_ms))
        ds = score_dataset(scores_ft)
        sweep.append({
            "fraction_threshold": ft,
            "tpr": ds.tpr, "far": ds.far,
            "n_tp": ds.n_tp, "n_fn": ds.n_fn,
            "n_fp": ds.n_fp, "n_tn": ds.n_tn,
            "median_ttd_ms": ds.median_ttd_ms,
            "mean_ttd_ms": ds.mean_ttd_ms,
        })
        ttd_str = (f"{ds.median_ttd_ms:.1f}" if ds.median_ttd_ms is not None
                   else "n/a")
        print(f"  frac={ft:.2f}: TPR={ds.tpr:.3f} ({ds.n_tp}/{ds.n_tp+ds.n_fn})"
              f"  FAR={ds.far:.3f} ({ds.n_fp}/{ds.n_fp+ds.n_tn})"
              f"  TTD_med={ttd_str}")
    return sweep


def main():
    print("JET-Disruption MVP: synthetic-data consensus catcher")
    print("=" * 70)

    n_disr, n_quie = 50, 50
    frac_thresholds = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
    warning_window_ms = 100.0

    print(f"Generating clean dataset: {n_disr} disr + {n_quie} quie...")
    t0 = time.time()
    shots_clean = generate_dataset(n_disruptive=n_disr, n_quiescent=n_quie,
                                    base_seed=42, mode="clean")
    print(f"  done in {time.time() - t0:.1f}s")

    print(f"Generating mixed dataset: weak precursors + glitches...")
    t0 = time.time()
    shots_mixed = generate_dataset(n_disruptive=n_disr, n_quiescent=n_quie,
                                    base_seed=43, mode="mixed")
    print(f"  done in {time.time() - t0:.1f}s")

    sweep_clean = run_mode(shots_clean, "CLEAN", frac_thresholds,
                            warning_window_ms)
    sweep_mixed = run_mode(shots_mixed, "MIXED", frac_thresholds,
                            warning_window_ms)

    out = Path(__file__).parent / "mvp_synthetic_results.json"
    out.write_text(json.dumps({
        "config": {"n_disruptive": n_disr, "n_quiescent": n_quie,
                   "z_threshold": 3.0, "min_duration_ms": 10.0,
                   "warning_window_ms": warning_window_ms,
                   "baseline_window_ms": 200.0, "assess_window_ms": 20.0},
        "clean_sweep": sweep_clean,
        "mixed_sweep": sweep_mixed,
    }, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
