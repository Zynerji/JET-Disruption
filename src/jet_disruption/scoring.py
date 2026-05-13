"""Score consensus catcher output: TPR, FAR, time-to-disruption (TTD)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catcher_v2 import ConsensusResult
from .synthetic_shot import SyntheticShot


@dataclass
class ShotScore:
    shot_id: int
    is_disruptive: bool
    alert_fired: bool
    alert_time_s: float | None
    t_disrupt_s: float | None
    ttd_ms: float | None  # time-to-disruption when alert fires (positive = ahead)
    false_alert: bool  # alert fired but shot is not disruptive
    missed: bool  # disruptive but no alert


def score_shot(shot: SyntheticShot, result: ConsensusResult,
               require_pre_disruption: bool = True,
               warning_window_ms: float = 100.0) -> ShotScore:
    """Score a single shot.

    require_pre_disruption: if True, alert must fire BEFORE t_disrupt to count
                            as TP (we want predictive, not coincident).
    warning_window_ms: alert must fire within this window before disruption.
    """
    alert_idx = result.alert_start_idx
    alert_t = float(shot.time_s[alert_idx]) if alert_idx is not None else None
    fired = alert_idx is not None

    is_disr = shot.is_disruptive
    td = shot.t_disrupt_s
    ttd = None
    if fired and is_disr and td is not None:
        ttd = (td - alert_t) * 1000.0  # ms; positive = ahead

    if is_disr and td is not None:
        # True positive: alert fires within [t_disrupt - warning_window, t_disrupt]
        is_tp = (fired and alert_t is not None
                  and (td - alert_t) * 1000.0 >= 0
                  and (td - alert_t) * 1000.0 <= warning_window_ms)
        missed = not is_tp
        false_alert = False
    else:
        # Non-disruptive shot
        false_alert = fired
        missed = False

    return ShotScore(
        shot_id=shot.shot_id,
        is_disruptive=is_disr,
        alert_fired=fired,
        alert_time_s=alert_t,
        t_disrupt_s=td,
        ttd_ms=ttd,
        false_alert=false_alert,
        missed=missed,
    )


@dataclass
class DatasetScore:
    n_disruptive: int
    n_quiescent: int
    n_tp: int  # disruptive + alert in window
    n_fn: int  # disruptive + no alert or alert outside window
    n_fp: int  # quiescent + alert
    n_tn: int  # quiescent + no alert
    tpr: float
    far: float  # fraction of non-disruptive shots with false alert
    median_ttd_ms: float | None
    mean_ttd_ms: float | None
    shot_scores: list[ShotScore]


def score_dataset(shot_scores: list[ShotScore]) -> DatasetScore:
    n_disr = sum(1 for s in shot_scores if s.is_disruptive)
    n_quie = sum(1 for s in shot_scores if not s.is_disruptive)
    tps = [s for s in shot_scores if s.is_disruptive and not s.missed]
    fns = [s for s in shot_scores if s.is_disruptive and s.missed]
    fps = [s for s in shot_scores if not s.is_disruptive and s.false_alert]
    tns = [s for s in shot_scores if not s.is_disruptive and not s.false_alert]
    tpr = len(tps) / n_disr if n_disr else 0.0
    far = len(fps) / n_quie if n_quie else 0.0
    ttds = [s.ttd_ms for s in tps if s.ttd_ms is not None]
    return DatasetScore(
        n_disruptive=n_disr,
        n_quiescent=n_quie,
        n_tp=len(tps), n_fn=len(fns), n_fp=len(fps), n_tn=len(tns),
        tpr=tpr, far=far,
        median_ttd_ms=float(np.median(ttds)) if ttds else None,
        mean_ttd_ms=float(np.mean(ttds)) if ttds else None,
        shot_scores=shot_scores,
    )
