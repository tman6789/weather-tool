"""Data quality checks and flag computation."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from weather_tool.core.metrics import expected_records

# Fields to auto-track for per-field NaN/missing% diagnostics when present in df
_TRACKABLE_FIELDS = ["tmpf", "dwpf", "relh", "sknt", "drct", "gust", "wetbulb_f"]


def compute_quality(
    df: pd.DataFrame,
    slice_start: pd.Timestamp,
    slice_end: pd.Timestamp,
    dt_minutes: float,
    interval_change_flag: bool,
    extra_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Compute quality metrics for a year-slice DataFrame.

    Parameters
    ----------
    df : DataFrame with columns timestamp, temp, _is_dup (plus optional extra fields).
        Already filtered to the year-slice.
    slice_start, slice_end : boundaries of this year-slice.
    dt_minutes : inferred median interval.
    interval_change_flag : from interval inference.
    extra_fields : list of additional field column names to compute per-field
        NaN counts and missing% for.  If None, auto-detected from
        _TRACKABLE_FIELDS present in df.

    Returns
    -------
    dict with quality fields matching the summary schema.
    """
    n_records_total = len(df)
    n_records_with_temp = int(df["temp"].notna().sum())
    nan_temp_count = n_records_total - n_records_with_temp

    # Unique timestamps (including rows with NaN temp) — after removing dups
    unique_ts = df.loc[~df["_is_dup"], "timestamp"]
    n_unique_timestamps = len(unique_ts)
    duplicate_count = int(df["_is_dup"].sum())

    expected = expected_records(slice_start, slice_end, dt_minutes)
    interval_unknown = expected == 0  # dt_minutes NaN/≤0 → can't compute missing%
    if expected > 0:
        missing_pct: float | None = max(0.0, min(1.0, 1.0 - n_unique_timestamps / expected))
    else:
        missing_pct = float("nan")

    result: dict[str, Any] = {
        "n_records_total": n_records_total,
        "n_records_with_temp": n_records_with_temp,
        "nan_temp_count": nan_temp_count,
        "n_unique_timestamps": n_unique_timestamps,
        "duplicate_count": duplicate_count,
        "missing_pct": round(missing_pct, 6) if not math.isnan(missing_pct) else float("nan"),
        "interval_change_flag": interval_change_flag,
        "interval_unknown_flag": interval_unknown,
    }

    # Per-field NaN counts and field_missing_pct
    if extra_fields is None:
        extra_fields = [f for f in _TRACKABLE_FIELDS if f in df.columns]

    for field in extra_fields:
        if field not in df.columns:
            continue
        nan_count = int(df[field].isna().sum())
        result[f"nan_count_{field}"] = nan_count
        result[f"field_missing_pct_{field}"] = round(
            nan_count / n_unique_timestamps if n_unique_timestamps > 0 else 0.0,
            6,
        )

    # Wet-bulb availability %
    if "wetbulb_f" in df.columns and n_records_with_temp > 0:
        wb_avail = int(df["wetbulb_f"].notna().sum())
        result["wetbulb_availability_pct"] = round(
            wb_avail / n_records_with_temp * 100, 2
        )

    return result
