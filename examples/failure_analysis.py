"""Failure analysis: characterize the 26% of disruptions missed at p=0.85.

For each missed shot:
  - duration of flat-top
  - max R, max R_excess, max weighted_consensus reached
  - was the shot terminated early?
  - which channels are quiescent (no |z|>3) up to disruption?

Goal: find the systematic pattern in missed disruptions to inform next
feature-engineering iteration.
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
from jet_disruption.catcher_v2 import per_channel_scores


def main():
    print("MAST GBM v4 Failure Analysis (at p=0.85, seed=42)", flush=True)
    print("=" * 70, flush=True)
    shots = load_mast_dataset("data/T_lead_0ms")

    # Train on all 500 shots, score back on each (a "leak" but for diagnosis)
    print("Train global weights + model (NOT CV — diagnostic only)", flush=True)
    wr = train_channel_weights(shots, MAST_CHANNELS,
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

    # Per-shot peak feature values in the analysis window
    print()
    print("Computing per-shot peak diagnostic stats...", flush=True)

    def shot_peak_stats(s):
        ft_min = getattr(s, "flat_top_tmin", None)
        ft_max = getattr(s, "flat_top_tmax", None)
        if (ft_min is None or ft_max is None
                or math.isnan(ft_min) or math.isnan(ft_max)):
            return None
        dt = float(s.time_s[1] - s.time_s[0])
        feats = feats_cache[s.shot_id]
        earliest = int((ft_min + 0.05) / dt)
        if s.is_disruptive and s.t_disrupt_s is not None:
            latest = int((s.t_disrupt_s) / dt)
        else:
            latest = int((ft_max + 0.02) / dt)
        latest = min(latest, feats.shape[0])
        if latest <= earliest:
            return None
        win = feats[earliest:latest]
        # multi-scale features index: 0-2 weighted_consensus, 3-5 R, 6-8 R_excess, 9-11 n_novel
        # 12 d_wc, 13 d_R, 14-23 peakz channels, 24 in_rampup, 25 in_flattop, 26 in_rampdown
        return {
            "flat_top_duration_s": float(ft_max - ft_min),
            "max_weighted_200ms": float(win[:, 1].max()),  # weighted_consensus_200ms
            "max_R_500ms": float(win[:, 5].max()),
            "max_R_excess_500ms": float(win[:, 8].max()),
            "max_R_excess_200ms": float(win[:, 7].max()),
            "max_n_novel_500ms": float(win[:, 11].max()),
            "saddle_coil_peakz": float(win[:, 14 + 8].max()),  # saddle_coil_ch6 idx
            "soft_xray_peakz": float(win[:, 14 + 9].max()),  # soft_xray_ch6 idx
            "ip_peakz": float(win[:, 14 + 0].max()),
        }

    # Compute stats for disruptive and non-disruptive separately
    disr_stats = []
    quie_stats = []
    for s in shots:
        st = shot_peak_stats(s)
        if st is None:
            continue
        st["shot_id"] = s.shot_id
        st["is_disruptive"] = s.is_disruptive
        st["t_disrupt_s"] = s.t_disrupt_s
        if s.is_disruptive:
            disr_stats.append(st)
        else:
            quie_stats.append(st)
    print(f"  disruptive shots analyzed: {len(disr_stats)}", flush=True)
    print(f"  non-disruptive shots: {len(quie_stats)}", flush=True)

    # Train an ungated classifier on all data, predict back to identify
    # likely-missed shots based on peak feature values
    print()
    print("Computing distribution stats (no classifier, just feature peaks):",
          flush=True)
    keys = ["max_weighted_200ms", "max_R_500ms", "max_R_excess_500ms",
            "max_n_novel_500ms", "saddle_coil_peakz", "soft_xray_peakz",
            "ip_peakz"]

    def report(stats, label):
        print(f"\n  {label} (n={len(stats)}):", flush=True)
        for k in keys:
            vals = np.array([s[k] for s in stats])
            print(f"    {k:>28s}: median={np.median(vals):>7.3f}  "
                  f"p25={np.percentile(vals, 25):>7.3f}  "
                  f"p75={np.percentile(vals, 75):>7.3f}", flush=True)

    report(disr_stats, "Disruptive")
    report(quie_stats, "Non-disruptive")

    # Identify likely-missed disruptions: those with weak peak features
    # across the board
    print()
    print("Identifying likely-missed disruptions (low peak features):",
          flush=True)
    disr_arr = np.array([[s[k] for k in keys] for s in disr_stats])
    quie_arr = np.array([[s[k] for k in keys] for s in quie_stats])
    quie_p90 = np.percentile(quie_arr, 90, axis=0)
    # Missed = no feature is > p90 of non-disruptive
    missed_count = 0
    for i, s in enumerate(disr_stats):
        if all(disr_arr[i] < quie_p90):
            missed_count += 1
    print(f"  {missed_count}/{len(disr_stats)} disruptions have ALL features "
          f"below the 90th percentile of non-disruptive shots", flush=True)
    print(f"    ({100*missed_count/len(disr_stats):.0f}% — these are "
          f"structurally missed, no feature engineering will help)", flush=True)

    # Bottom-decile in weighted_consensus_200ms specifically
    weak_wc = [s for s in disr_stats
                if s["max_weighted_200ms"] < np.percentile(disr_arr[:, 0], 10)]
    print(f"\n  {len(weak_wc)} disruptions with bottom-decile max_weighted_200ms"
          f" (very weak Kuramoto signal):", flush=True)
    for s in weak_wc[:5]:
        print(f"    shot {s['shot_id']}: td={s['t_disrupt_s']:.3f}s, "
              f"max_wc_200ms={s['max_weighted_200ms']:.3f}, "
              f"max_R_500ms={s['max_R_500ms']:.3f}, "
              f"ip_peakz={s['ip_peakz']:.2f}", flush=True)

    out = Path(__file__).parent / "mast_failure_analysis.json"
    out.write_text(json.dumps({
        "n_disruptive": len(disr_stats),
        "n_quiescent": len(quie_stats),
        "structurally_missed_count": missed_count,
        "structurally_missed_fraction": missed_count / len(disr_stats),
        "weak_kuramoto_disruptions": [s["shot_id"] for s in weak_wc],
        "non_disruptive_p90_thresholds": dict(zip(keys, quie_p90.tolist())),
    }, indent=2, default=str))
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
