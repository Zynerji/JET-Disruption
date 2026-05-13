"""Load MAST tokamak disruption-detection dataset (Zenodo 16032053).

Dataset: 500 MAST shots, 10 plasma diagnostics, ~0.43 s @ 4638 Hz, with
ground-truth disruption times in NetCDF attrs.

Reference:
  Sharma, P. (2025). MAST disruption detection dataset.
  Zenodo. https://doi.org/10.5281/zenodo.16032053
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .synthetic_shot import SyntheticShot  # share the dataclass

MAST_CHANNELS = (
    "ip", "power_radiated", "n_e_core", "t_e_core", "li",
    "vloop_dynamic", "dalpha_signal_HM10_T", "dalpha_signal_HU10_T",
    "saddle_coil_ch6", "soft_xray_ch6",
)


@dataclass
class MastShot:
    """MAST shot with flat-top region metadata.

    Subset / variant of SyntheticShot that exposes ramp-up vs flat-top
    boundaries so the catcher can be restricted to the analyzable region.
    """
    shot_id: int
    is_disruptive: bool
    t_disrupt_s: float | None
    time_s: np.ndarray
    diagnostics: np.ndarray
    diagnostic_names: tuple[str, ...] = MAST_CHANNELS
    flat_top_tmin: float | None = None
    flat_top_tmax: float | None = None
    ramp_up_tmax: float | None = None


def load_mast_shot(nc_path: str | Path) -> MastShot:
    """Load one MAST NetCDF shot."""
    import xarray as xr

    ds = xr.open_dataset(str(nc_path))
    try:
        t = np.asarray(ds.time.values, dtype=np.float32)
        T = len(t)
        C = len(MAST_CHANNELS)
        diag = np.zeros((T, C), dtype=np.float32)
        for c, name in enumerate(MAST_CHANNELS):
            diag[:, c] = np.asarray(ds[name].values, dtype=np.float32)
        td_attr = ds.attrs.get("disruption.td")
        td = (float(td_attr)
              if td_attr is not None
                  and not (isinstance(td_attr, float) and math.isnan(td_attr))
              else None)
        is_disr = td is not None
        ft_min = float(ds.attrs.get("pulse_regions.flat_top.tmin", 0.0))
        ft_max = float(ds.attrs.get("pulse_regions.flat_top.tmax", t[-1]))
        ru_max = float(ds.attrs.get("pulse_regions.ramp_up.tmax", 0.0))
        shot_id_str = Path(nc_path).stem.replace("shot_", "")
        try:
            shot_id = int(shot_id_str)
        except ValueError:
            shot_id = hash(shot_id_str) & 0x7fffffff
    finally:
        ds.close()

    # Replace NaN / Inf with channel medians
    for c in range(C):
        col = diag[:, c]
        bad = ~np.isfinite(col)
        if bad.any():
            good_median = float(np.median(col[~bad])) if (~bad).any() else 0.0
            col[bad] = good_median
            diag[:, c] = col

    return MastShot(
        shot_id=shot_id,
        is_disruptive=is_disr,
        t_disrupt_s=td,
        time_s=t,
        diagnostics=diag,
        flat_top_tmin=ft_min,
        flat_top_tmax=ft_max,
        ramp_up_tmax=ru_max,
    )


def load_mast_dataset(dir_path: str | Path = "data/T_lead_0ms",
                       max_shots: int | None = None) -> list[MastShot]:
    """Load all (or first max_shots) shots from a MAST extraction directory."""
    dir_path = Path(dir_path)
    nc_files = sorted(dir_path.glob("shot_*.nc"))
    if max_shots is not None:
        nc_files = nc_files[:max_shots]
    shots = []
    for fp in nc_files:
        try:
            shots.append(load_mast_shot(fp))
        except Exception as e:
            print(f"  load error {fp.name}: {e}")
    return shots
