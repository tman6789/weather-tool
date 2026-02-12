"""Data quality checks and flag computation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from weather_tool.core.metrics import expected_records, infer_interval


def compute_quality(
    df: pd.DataFrame,
    slice_start: pd.Timestamp,
    slice_end: pd.Timestamp,
    dt_minutes: float,
    interval_change_flag: bool,
) -> dict[str, Any]:
    """Compute quality metrics for a year-slice DataFrame.

    Parameters
    ----------
    df : DataFrame with columns timestamp, temp, _is_dup
        Already filtered to the year-slice.
    slice_start, slice_end : boundaries of this year-slice.
    dt_minutes : inferred median interval.
    interval_change_flag : from interval inference.

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
    if expected > 0:
        missing_pct = max(0.0, min(1.0, 1.0 - n_unique_timestamps / expected))
    else:
        missing_pct = 0.0

    return {
        "n_records_total": n_records_total,
        "n_records_with_temp": n_records_with_temp,
        "nan_temp_count": nan_temp_count,
        "n_unique_timestamps": n_unique_timestamps,
        "duplicate_count": duplicate_count,
        "missing_pct": round(missing_pct, 6),
        "interval_change_flag": interval_change_flag,
    }
