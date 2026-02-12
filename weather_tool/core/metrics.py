"""Deterministic metric computations: interval inference, hours-above-ref."""

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
