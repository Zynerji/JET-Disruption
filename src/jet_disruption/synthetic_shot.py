"""Synthetic JET-like plasma shot generator.

Generates a multi-diagnostic time series with realistic noise characteristics
and an optional disruption precursor pattern. The precursor model is the
simplest one consistent with JET literature: 50-80 ms before disruption,
locked-mode amplitude rises monotonically, edge density collapses, MHD
fluctuation envelope spikes, and the radiated power undergoes a step
increase from radiation-coupled instability.

Diagnostic channels (14, matching common JET subset):
  - locked_mode      (LM): n=1 locked-mode amplitude        [mT]
  - mhd_rms          (MHD): magnetic fluctuation envelope    [mT]
  - density_core     (DCO): line-integrated core density     [1e19 m^-3]
  - density_edge     (DED): line-integrated edge density     [1e19 m^-3]
  - prad_total       (PRAD): total radiated power            [MW]
  - prad_core_frac   (PRC): fraction of P_rad from core       [-]
  - ip_normalised    (IP): plasma current / reference        [-]
  - q95              (Q95): edge safety factor               [-]
  - betap            (BP): poloidal beta                     [-]
  - li               (LI): internal inductance               [-]
  - dwmhd_dt         (DW): time-derivative of stored energy  [MW]
  - sxr_core         (SXR): soft-X-ray core emissivity        [a.u.]
  - neutron_rate     (NR): neutron emission rate              [a.u.]
  - elm_marker       (ELM): ELM event indicator               [-]

Sampling: 1 kHz (1 ms per sample). Default shot length: 10 s = 10000 samples.
Disruption window: 50-80 ms before t_disrupt.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


DIAGNOSTIC_NAMES = (
    "locked_mode", "mhd_rms", "density_core", "density_edge",
    "prad_total", "prad_core_frac", "ip_normalised", "q95",
    "betap", "li", "dwmhd_dt", "sxr_core", "neutron_rate", "elm_marker",
)


@dataclass
class SyntheticShot:
    shot_id: int
    is_disruptive: bool
    t_disrupt_s: float | None  # None for non-disruptive
    time_s: np.ndarray  # (T,)
    diagnostics: np.ndarray  # (T, 14)
    diagnostic_names: tuple[str, ...] = DIAGNOSTIC_NAMES


def _baseline_quiescent(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Quiescent baseline values for 14 diagnostics over time array t."""
    T = len(t)
    D = 14
    out = np.zeros((T, D), dtype=np.float32)

    # Slow drifts + low-amplitude noise per channel
    out[:, 0] = 0.05 + 0.005 * rng.standard_normal(T)  # locked_mode
    out[:, 1] = 0.15 + 0.03 * np.abs(rng.standard_normal(T))  # mhd_rms
    out[:, 2] = 6.0 + 0.2 * rng.standard_normal(T)  # density_core
    out[:, 3] = 3.0 + 0.15 * rng.standard_normal(T)  # density_edge
    out[:, 4] = 8.0 + 0.5 * rng.standard_normal(T)  # prad_total
    out[:, 5] = 0.55 + 0.05 * rng.standard_normal(T)  # prad_core_frac
    out[:, 6] = 1.0 + 0.01 * rng.standard_normal(T)  # ip_normalised
    out[:, 7] = 3.2 + 0.05 * rng.standard_normal(T)  # q95
    out[:, 8] = 0.6 + 0.02 * rng.standard_normal(T)  # betap
    out[:, 9] = 0.95 + 0.02 * rng.standard_normal(T)  # li
    out[:, 10] = 0.0 + 0.5 * rng.standard_normal(T)  # dwmhd_dt
    out[:, 11] = 1.0 + 0.05 * rng.standard_normal(T)  # sxr_core
    out[:, 12] = 1.0 + 0.05 * rng.standard_normal(T)  # neutron_rate

    # ELM markers: sparse spikes every ~100 ms
    elms_at = rng.uniform(0, t[-1], size=max(1, int(t[-1] * 10)))
    elm = np.zeros(T)
    for et in elms_at:
        idx = int(et / (t[1] - t[0]))
        if 0 <= idx < T:
            elm[idx] = 1.0
    out[:, 13] = elm

    return out


def _apply_disruption_precursor(diag: np.ndarray, t: np.ndarray,
                                 t_disrupt: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """Apply a precursor pattern to the diagnostics array in-place.

    Pattern starts at t_disrupt - 80ms; severity grows linearly to t_disrupt.
    """
    diag = diag.copy()
    dt = t[1] - t[0]
    pre_start = t_disrupt - 0.080  # 80 ms warning
    pre_idx_start = int(pre_start / dt)
    pre_idx_end = int(t_disrupt / dt)
    if pre_idx_start < 0:
        pre_idx_start = 0
    if pre_idx_end > len(t):
        pre_idx_end = len(t)
    if pre_idx_end <= pre_idx_start:
        return diag

    n = pre_idx_end - pre_idx_start
    ramp = np.linspace(0, 1, n)

    # Locked mode: exponential growth from 0.05 to ~0.5 mT
    diag[pre_idx_start:pre_idx_end, 0] += 0.5 * (np.exp(3 * ramp) - 1) / (np.exp(3) - 1)
    # MHD fluctuations: spike (5x baseline) in last 30 ms
    last30 = int(0.030 / dt)
    if pre_idx_end > last30:
        spike_start = pre_idx_end - last30
        diag[spike_start:pre_idx_end, 1] *= 5.0 * rng.uniform(0.8, 1.2, pre_idx_end - spike_start)
    # Density edge: collapse from 3.0 to 1.5 over the window
    diag[pre_idx_start:pre_idx_end, 3] -= 1.5 * ramp
    # P_rad: step increase + 50% in last 50ms
    last50 = int(0.050 / dt)
    if pre_idx_end > last50:
        step_start = pre_idx_end - last50
        diag[step_start:pre_idx_end, 4] *= 1.5
        diag[step_start:pre_idx_end, 5] += 0.15  # core radiation fraction climbs
    # q95 dropping (current penetration anomaly)
    diag[pre_idx_start:pre_idx_end, 7] -= 0.4 * ramp
    # dWmhd/dt: large negative going (stored energy collapsing)
    diag[pre_idx_start:pre_idx_end, 10] -= 5.0 * ramp ** 2
    # SXR: drop in last 30 ms
    if pre_idx_end > last30:
        diag[pre_idx_end - last30:pre_idx_end, 11] *= 0.7
    # Neutron rate: drops in last 20 ms
    last20 = int(0.020 / dt)
    if pre_idx_end > last20:
        diag[pre_idx_end - last20:pre_idx_end, 12] *= 0.5

    # Disruption itself: discontinuity at t_disrupt+
    if pre_idx_end < len(t):
        diag[pre_idx_end:, 0] = 0.05
        diag[pre_idx_end:, 6] *= 0.0  # current quench
        diag[pre_idx_end:, 8] *= 0.0
        diag[pre_idx_end:, 10] = 0.0
        diag[pre_idx_end:, 11] *= 0.0
        diag[pre_idx_end:, 12] *= 0.0

    return diag


def _inject_glitch(diag: np.ndarray, t: np.ndarray,
                    rng: np.random.Generator,
                    n_glitches: int = 2) -> np.ndarray:
    """Inject noise glitches in quiescent shots to make FAR test non-trivial.

    A glitch = a 5-15 ms spike on 1-3 random channels at a random time.
    Mimics common JET artefacts: locked-mode bumps from ELMs, MHD pickup,
    bolometer noise.
    """
    diag = diag.copy()
    dt = t[1] - t[0]
    T = len(t)
    for _ in range(n_glitches):
        t_g = rng.uniform(0.5, t[-1] - 0.2)
        idx = int(t_g / dt)
        dur = int(rng.uniform(0.005, 0.015) / dt)
        chans = rng.choice(14, size=rng.integers(1, 4), replace=False)
        for c in chans:
            scale = rng.uniform(2.5, 5.0)
            sign = rng.choice([-1, 1])
            for k in range(dur):
                if 0 <= idx + k < T:
                    diag[idx + k, c] += sign * scale * np.abs(rng.standard_normal())
    return diag


def generate_shot(shot_id: int,
                  is_disruptive: bool,
                  duration_s: float = 10.0,
                  sample_rate_hz: float = 1000.0,
                  seed: int | None = None,
                  precursor_strength: float = 1.0,
                  glitches_in_quiescent: int = 0) -> SyntheticShot:
    """Generate one synthetic JET-like shot.

    precursor_strength: 0.0 = no precursor (essentially quiescent before
                        the sudden disruption-step), 1.0 = canonical pattern,
                        intermediate values give weak/partial precursors.
    glitches_in_quiescent: number of noise spikes to inject (for quiescent
                            shots, to test FAR).
    """
    rng = np.random.default_rng(seed if seed is not None else shot_id)
    T = int(duration_s * sample_rate_hz)
    t = np.arange(T) / sample_rate_hz
    diag = _baseline_quiescent(t, rng)

    t_disrupt = None
    if is_disruptive:
        t_disrupt = float(rng.uniform(duration_s - 2.0, duration_s - 0.5))
        if precursor_strength > 0:
            full = _apply_disruption_precursor(diag, t, t_disrupt, rng)
            # Blend: weighted average of baseline and full-precursor
            diag = (1 - precursor_strength) * diag + precursor_strength * full
            # Still apply the post-disruption collapse (independent of strength)
            dt = t[1] - t[0]
            pre_end = int(t_disrupt / dt)
            if pre_end < T:
                diag[pre_end:, 0] = 0.05
                diag[pre_end:, 6] *= 0.0
                diag[pre_end:, 8] *= 0.0
                diag[pre_end:, 10] = 0.0
                diag[pre_end:, 11] *= 0.0
                diag[pre_end:, 12] *= 0.0
    else:
        if glitches_in_quiescent > 0:
            diag = _inject_glitch(diag, t, rng, n_glitches=glitches_in_quiescent)

    return SyntheticShot(
        shot_id=shot_id,
        is_disruptive=is_disruptive,
        t_disrupt_s=t_disrupt,
        time_s=t,
        diagnostics=diag.astype(np.float32),
    )


def generate_dataset(n_disruptive: int = 50,
                     n_quiescent: int = 50,
                     duration_s: float = 10.0,
                     sample_rate_hz: float = 1000.0,
                     base_seed: int = 0,
                     mode: str = "clean") -> list[SyntheticShot]:
    """Generate a balanced synthetic dataset.

    mode: "clean"  - canonical strong precursor for all disruptive shots,
                     no glitches in quiescent. (Method validation.)
          "mixed"  - vary precursor strength uniformly in [0.4, 1.0] for
                     disruptive shots; inject 0-3 glitches in quiescent.
                     (Harder test, more realistic.)
    """
    rng = np.random.default_rng(base_seed)
    shots = []
    for i in range(n_disruptive):
        if mode == "mixed":
            ps = float(rng.uniform(0.4, 1.0))
        else:
            ps = 1.0
        shots.append(generate_shot(shot_id=base_seed * 100 + i,
                                    is_disruptive=True,
                                    duration_s=duration_s,
                                    sample_rate_hz=sample_rate_hz,
                                    seed=base_seed * 100 + i,
                                    precursor_strength=ps))
    for i in range(n_quiescent):
        if mode == "mixed":
            ng = int(rng.integers(0, 4))
        else:
            ng = 0
        shots.append(generate_shot(shot_id=base_seed * 100 + 50 + i,
                                    is_disruptive=False,
                                    duration_s=duration_s,
                                    sample_rate_hz=sample_rate_hz,
                                    seed=base_seed * 100 + 50 + i,
                                    glitches_in_quiescent=ng))
    return shots
