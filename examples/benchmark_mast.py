"""Benchmark the cross-diagnostic consensus catcher on real MAST shots.

Dataset: Sharma 2025, "MAST disruption detection dataset" (Zenodo 16032053).
500 shots, 10 plasma diagnostics, 4638 Hz sample rate.

We run the per-channel rolling z-score scorer once per shot, then sweep the
consensus-fraction threshold and the warning window to map TPR / FAR.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from jet_disruption.mast_loader import load_mast_dataset, MAST_CHANNELS
from jet_disruption.catcher_v2 import per_channel_scores, consensus_alert
from jet_disruption.scoring import score_shot, score_dataset


def mask_to_analysis_window(result, time_s, flat_top_tmin, flat_top_tmax,
                              t_disrupt,
                              baseline_window_ms: float = 50.0,
                              post_flat_top_buffer_ms: float = 20.0):
    """Restrict alert detection to a physically meaningful window.

    Window:
      start = flat_top_tmin + baseline_window_ms  (baseline fully post-ramp)
      end   = min(t_disrupt or inf, flat_top_tmax + post_flat_top_buffer_ms)

    Rationale: plasma ramp-up and controlled ramp-down both produce
    multi-channel deflections similar to disruption precursors. We exclude
    them. For disruptive shots, we let the window extend to t_disrupt (since
    disruptions can happen at or just past flat-top end). For non-disruptive
    shots, the window ends at flat_top_tmax + a small buffer.
    """
    import math
    if (flat_top_tmin is None or flat_top_tmin <= 0
            or math.isnan(flat_top_tmin)
            or flat_top_tmax is None
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
    # Find first alert in [earliest, latest]
    new_start = None
    for i in range(earliest, min(latest + 1, len(result.alert_active))):
        if result.alert_active[i]:
            new_start = i
            break
    result.alert_start_idx = new_start
    return result


def main():
    print("MAST Benchmark: consensus catcher on real shots")
    print("=" * 70)

    print("Loading MAST shots from data/T_lead_0ms/ ...")
    t0 = time.time()
    shots = load_mast_dataset("data/T_lead_0ms")
    print(f"  loaded {len(shots)} shots in {time.time() - t0:.1f}s")

    n_disr = sum(1 for s in shots if s.is_disruptive)
    n_quie = sum(1 for s in shots if not s.is_disruptive)
    print(f"  disruptive: {n_disr}, non-disruptive: {n_quie}")

    # Compute per-channel z-scores once per shot
    # MAST sample rate ~4638 Hz, so baseline 200 ms -> 928 samples,
    # assess 20 ms -> 93 samples. We rescale baseline-window-ms by the
    # actual dt of each shot. catcher_v2 handles this via dt argument.
    print()
    print("Computing per-channel rolling robust z-scores ...")
    print("  (baseline 50 ms, assess 5 ms, |z|>3.0 = novel)")
    t0 = time.time()
    pre = {}
    for s in shots:
        dt = float(s.time_s[1] - s.time_s[0])
        pcs = per_channel_scores(
            s.diagnostics, dt,
            baseline_window_ms=50.0,
            assess_window_ms=5.0,
            z_threshold=3.0,
        )
        pre[s.shot_id] = pcs
    print(f"  done in {time.time() - t0:.1f}s")

    # Threshold + warning-window sweep
    print()
    print("Sweep over fraction threshold and warning window ...")
    frac_thresholds = (0.20, 0.30, 0.40, 0.50, 0.60)
    warning_windows_ms = (30.0, 50.0, 100.0, 200.0)

    sweep = []
    for ft in frac_thresholds:
        for ww in warning_windows_ms:
            scores = []
            for s in shots:
                dt = float(s.time_s[1] - s.time_s[0])
                r = consensus_alert(
                    pre[s.shot_id], s.diagnostic_names,
                    fraction_threshold=ft,
                    min_duration_ms=5.0,  # 5 ms sustain
                    dt=dt,
                )
                r = mask_to_analysis_window(r, s.time_s,
                                             s.flat_top_tmin, s.flat_top_tmax,
                                             s.t_disrupt_s)
                scores.append(score_shot(s, r, warning_window_ms=ww))
            ds = score_dataset(scores)
            sweep.append({
                "fraction_threshold": ft,
                "warning_window_ms": ww,
                "tpr": ds.tpr, "far": ds.far,
                "n_tp": ds.n_tp, "n_fn": ds.n_fn,
                "n_fp": ds.n_fp, "n_tn": ds.n_tn,
                "median_ttd_ms": ds.median_ttd_ms,
                "mean_ttd_ms": ds.mean_ttd_ms,
            })

    # Print as a 5x4 grid
    print()
    print("TPR / FAR grid (rows = frac threshold, cols = warning window ms)")
    print(f"{'frac':>6} | " + " | ".join(f"{ww:>8.0f}ms" for ww in warning_windows_ms))
    print("-" * 70)
    for ft in frac_thresholds:
        cells = []
        for ww in warning_windows_ms:
            entry = next(e for e in sweep
                         if e["fraction_threshold"] == ft
                         and e["warning_window_ms"] == ww)
            cells.append(f"{entry['tpr']*100:>3.0f}/{entry['far']*100:<3.0f}")
        print(f"{ft:>6.2f} | " + " | ".join(f"{c:>9s}" for c in cells))

    print()
    print("Best Pareto points (high TPR + low FAR):")
    # Find Pareto-best
    pareto = []
    for e in sweep:
        dominated = False
        for o in sweep:
            if (o["tpr"] > e["tpr"] and o["far"] <= e["far"]
                  or o["tpr"] >= e["tpr"] and o["far"] < e["far"]):
                dominated = True
                break
        if not dominated and e["tpr"] > 0:
            pareto.append(e)
    pareto.sort(key=lambda x: (-x["tpr"], x["far"]))
    for p in pareto[:10]:
        ttd = (f"{p['median_ttd_ms']:.1f}" if p['median_ttd_ms'] is not None
               else "n/a")
        print(f"  frac={p['fraction_threshold']:.2f}, "
              f"warn={p['warning_window_ms']:.0f}ms: "
              f"TPR={p['tpr']:.3f} FAR={p['far']:.3f} "
              f"TTD_med={ttd}ms")

    out = Path(__file__).parent / "mast_benchmark_results.json"
    out.write_text(json.dumps({
        "dataset": "MAST disruption detection (Sharma 2025, Zenodo 16032053)",
        "n_shots": len(shots),
        "n_disruptive": n_disr,
        "n_non_disruptive": n_quie,
        "config": {"baseline_window_ms": 50.0, "assess_window_ms": 5.0,
                   "z_threshold": 3.0, "min_duration_ms": 5.0},
        "channels": list(MAST_CHANNELS),
        "sweep": sweep,
    }, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
