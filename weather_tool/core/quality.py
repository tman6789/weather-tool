"""Data quality checks and flag computation."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

# Fields to auto-track for per-field NaN/missing% diagnostics when present in df
_TRACKABLE_FIELDS = ["tmpf", "dwpf", "relh", "sknt", "drct", "gust", "wetbulb_f"]


def compute_year_coverage(
    timestamps: pd.Series,
    is_dup: pd.Series,
    slice_start: pd.Timestamp,
    slice_end: pd.Timestamp,
) -> dict[str, Any]:
    """Canonical hourly coverage for one year-slice.

    expected_hours = distinct calendar hours in [slice_start.floor('h'), slice_end.floor('h')]
    observed_hours = distinct hourly buckets (dedup-only observations inside window)
    missing_pct    = missing_hours / expected_hours  (0–1 scale; NaN if expected==0)

    Returns dict with: expected_hours (int), observed_hours (int),
    missing_hours (int), missing_pct (float).
    """
    start_h = slice_start.floor("h")
    end_h   = slice_end.floor("h")   # "23:59:59".floor("h") → "23:00:00" ✓

    if end_h >= start_h:
        expected_hours = int((end_h - start_h).total_seconds() // 3600) + 1
    else:
        expected_hours = 0

    dedup_ts = timestamps[~is_dup]
    if len(dedup_ts) > 0 and expected_hours > 0:
        bucketed      = dedup_ts.dt.floor("h")
        in_window     = (bucketed >= start_h) & (bucketed <= end_h)
        observed_hours = int(bucketed[in_window].nunique())
    else:
        observed_hours = 0

    missing_hours = max(0, expected_hours - observed_hours)

    if expected_hours > 0:
        missing_pct: float = round(missing_hours / expected_hours, 6)
    else:
        missing_pct = float("nan")

    return {
        "expected_hours": expected_hours,
        "observed_hours": observed_hours,
        "missing_hours":  missing_hours,
        "missing_pct":    missing_pct,
    }


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
    n_records_total     = len(df)
    n_records_with_temp = int(df["temp"].notna().sum())  # all rows — backward compat

    dedup = df.loc[~df["_is_dup"]]
    n_unique_timestamps = len(dedup)
    duplicate_count     = int(df["_is_dup"].sum())
    # FIX: NaN temp counted on dedup rows only (distinct timestamps with missing value)
    nan_temp_count      = int(dedup["temp"].isna().sum())

    # Hourly coverage — replaces dt_minutes-based expected_records
    coverage = compute_year_coverage(df["timestamp"], df["_is_dup"], slice_start, slice_end)

    interval_unknown = math.isnan(dt_minutes) or dt_minutes <= 0

    result: dict[str, Any] = {
        "n_records_total":       n_records_total,
        "n_records_with_temp":   n_records_with_temp,
        "nan_temp_count":        nan_temp_count,
        "n_unique_timestamps":   n_unique_timestamps,
        "duplicate_count":       duplicate_count,
        **coverage,              # expected_hours, observed_hours, missing_hours, missing_pct
        "interval_change_flag":  interval_change_flag,
        "interval_unknown_flag": interval_unknown,
    }

    # Per-field NaN counts and field_missing_pct (operate on dedup rows for consistency)
    if extra_fields is None:
        extra_fields = [f for f in _TRACKABLE_FIELDS if f in df.columns]

    for field in extra_fields:
        if field not in df.columns:
            continue
        nan_count = int(dedup[field].isna().sum())
        result[f"nan_count_{field}"] = nan_count
        result[f"field_missing_pct_{field}"] = round(
            nan_count / n_unique_timestamps if n_unique_timestamps > 0 else 0.0,
            6,
        )

    # Wet-bulb availability % — use deduplicated rows for both numerator and
    # denominator, consistent with all other per-year metrics.
    if "wetbulb_f" in df.columns:
        n_dedup_with_temp = int(dedup["temp"].notna().sum())
        if n_dedup_with_temp > 0:
            wb_avail = int(dedup["wetbulb_f"].notna().sum())
            result["wetbulb_availability_pct"] = round(
                wb_avail / n_dedup_with_temp * 100, 2
            )

    return result
