"""Deterministic metric computations: interval inference, hours-above-ref, wet-bulb."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from weather_tool.config import INTERVAL_CHANGE_RATIO, INTERVAL_CHANGE_TOL


# ── Interval inference ─────────────────────────────────────────────────────

def infer_interval(timestamps: pd.Series) -> dict[str, Any]:
    """Infer the sampling interval from a sorted Series of unique tz-aware timestamps.

    Returns
    -------
    dict with keys:
        dt_minutes          – median diff in minutes (float)
        p10, p90            – 10th/90th percentile diffs
        interval_change_flag – True if >20 % of diffs deviate from median by >10 %
        unique_diff_counts  – dict[float, int] of rounded-minute diff → count
    """
    if len(timestamps) < 2:
        return {
            "dt_minutes": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "interval_change_flag": False,
            "unique_diff_counts": {},
        }

    sorted_ts = timestamps.sort_values().reset_index(drop=True)
    diffs_td = sorted_ts.diff().dropna()
    diffs_min = diffs_td.dt.total_seconds() / 60.0

    # Discard zero/negative
    diffs_min = diffs_min[diffs_min > 0]

    if len(diffs_min) == 0:
        return {
            "dt_minutes": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "interval_change_flag": False,
            "unique_diff_counts": {},
        }

    dt_minutes = float(np.median(diffs_min))
    p10 = float(np.percentile(diffs_min, 10))
    p90 = float(np.percentile(diffs_min, 90))

    # Interval change detection
    if dt_minutes > 0:
        deviations = np.abs(diffs_min - dt_minutes) / dt_minutes
        frac_deviant = float((deviations > INTERVAL_CHANGE_TOL).sum()) / len(diffs_min)
        interval_change_flag = frac_deviant > INTERVAL_CHANGE_RATIO
    else:
        interval_change_flag = False

    # Unique diff counts (rounded to 1 decimal)
    rounded = diffs_min.round(1)
    counts = rounded.value_counts().sort_index()
    unique_diff_counts = {float(k): int(v) for k, v in counts.items()}

    return {
        "dt_minutes": dt_minutes,
        "p10": p10,
        "p90": p90,
        "interval_change_flag": interval_change_flag,
        "unique_diff_counts": unique_diff_counts,
    }


# ── Hours above reference temperature ────────────────────────────────────

def hours_above_ref(temps: pd.Series, ref_temp: float, dt_minutes: float) -> float:
    """Compute hours above *ref_temp* using a step-function approach.

    Parameters
    ----------
    temps : pd.Series of float (may contain NaN)
    ref_temp : reference temperature threshold (strict >)
    dt_minutes : inferred sampling interval in minutes

    Returns
    -------
    float – total hours where temp > ref_temp
    """
    if math.isnan(dt_minutes) or dt_minutes <= 0:
        return 0.0
    valid = temps.dropna()
    count_above = int((valid > ref_temp).sum())
    return count_above * (dt_minutes / 60.0)


# ── Wet-bulb temperature (Stull 2011 approximation) ──────────────────────

def _stull_wetbulb_c(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Stull (2011) wet-bulb approximation (vectorized).

    Parameters
    ----------
    t_c : dry-bulb temperature in °C (numpy array, no NaNs)
    rh  : relative humidity in % (numpy array, no NaNs)

    Returns
    -------
    numpy array of wet-bulb temperature in °C

    Reference: Stull, R. (2011). Wet-bulb temperature from relative humidity
    and air temperature. J. Appl. Meteor. Climatol., 50, 2267–2269.
    Accuracy: ±0.35 °C for T ∈ [−20, 50] °C, RH ∈ [5, 99] %.
    """
    return (
        t_c * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
        + np.arctan(t_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


def compute_wetbulb_f(df: pd.DataFrame) -> pd.Series:
    """Compute wet-bulb temperature in °F using the Stull (2011) approximation.

    Preferred path: uses ``tmpf`` (or ``temp``) + ``relh`` columns.
    Fallback path:  derives RH from ``tmpf``/``temp`` + ``dwpf`` via Magnus
                    approximation, then applies Stull.
    Returns NaN for rows where required inputs are missing.

    Parameters
    ----------
    df : DataFrame that may contain columns tmpf/temp, relh, dwpf.

    Returns
    -------
    pd.Series of float64 (wetbulb_f), same index as *df*.
    """
    # --- dry-bulb in Fahrenheit ---
    if "tmpf" in df.columns:
        t_f = pd.to_numeric(df["tmpf"], errors="coerce")
    elif "temp" in df.columns:
        t_f = pd.to_numeric(df["temp"], errors="coerce")
    else:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    t_c = (t_f - 32.0) * 5.0 / 9.0

    # --- relative humidity ---
    if "relh" in df.columns:
        rh = pd.to_numeric(df["relh"], errors="coerce")
    elif "dwpf" in df.columns:
        # Magnus approximation: RH = 100 · e_s(Td) / e_s(T)
        td_f = pd.to_numeric(df["dwpf"], errors="coerce")
        td_c = (td_f - 32.0) * 5.0 / 9.0
        e_td = np.exp(17.625 * td_c / (243.04 + td_c))
        e_t = np.exp(17.625 * t_c / (243.04 + t_c))
        rh = (100.0 * e_td / e_t).clip(0.0, 100.0)  # clamp: dew point > dry-bulb is unphysical
    else:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    # --- compute Stull only where both inputs are valid ---
    valid = ~(t_c.isna() | rh.isna())
    twb_c = np.full(len(df), np.nan)
    if valid.any():
        twb_c[valid.values] = _stull_wetbulb_c(
            t_c.values[valid.values], rh.values[valid.values]
        )

    twb_f = twb_c * 9.0 / 5.0 + 32.0
    # Physics constraint: wet-bulb ≤ dry-bulb (Stull can overshoot at extreme RH)
    twb_f = np.minimum(twb_f, np.asarray(t_f, dtype=float))
    return pd.Series(twb_f, index=df.index, dtype="float64")


# ── Expected records for a time window ───────────────────────────────────

def expected_records(
    slice_start: pd.Timestamp,
    slice_end: pd.Timestamp,
    dt_minutes: float,
) -> int:
    """Number of records expected in [slice_start, slice_end] given dt_minutes."""
    if math.isnan(dt_minutes) or dt_minutes <= 0:
        return 0
    total_minutes = (slice_end - slice_start).total_seconds() / 60.0
    return int(math.floor(total_minutes / dt_minutes)) + 1
