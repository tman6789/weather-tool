"""Extreme value analysis — rolling persistence, exceedance hours, design day."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ── Rolling persistence ──────────────────────────────────────────────────────


def compute_rolling_max(
    series: pd.Series,
    timestamps: pd.Series,
    window_hours: int,
    dt_minutes: float,
    min_frac: float = 0.80,
) -> float:
    """Max rolling *window_hours*-h mean with completeness guard.

    Resamples the series onto a regular *dt_minutes* grid, then applies an
    integer-step rolling window.  Works on any numeric series (temp, wetbulb,
    etc.).

    Returns NaN if dt_minutes is invalid, the series is empty, or no window
    meets the completeness threshold.
    """
    if math.isnan(dt_minutes) or dt_minutes <= 0 or len(series) == 0:
        return float("nan")

    valid = series.dropna()
    if len(valid) == 0:
        return float("nan")

    indexed = pd.Series(series.values, index=pd.DatetimeIndex(timestamps.values))
    indexed = indexed.sort_index()

    dt_round = max(1, round(dt_minutes))
    freq_str = f"{dt_round}min"
    resampled = indexed.resample(freq_str).mean()

    n_steps = int(round(window_hours * 60.0 / dt_round))
    if n_steps < 1:
        return float("nan")
    min_periods = max(1, int(min_frac * n_steps))

    rolling_mean = resampled.rolling(window=n_steps, min_periods=min_periods).mean()
    valid_means = rolling_mean.dropna()
    if len(valid_means) == 0:
        return float("nan")
    return round(float(valid_means.max()), 2)


def compute_rolling_window_max(
    series: pd.Series,
    timestamps: pd.Series,
    window_hours: int,
    dt_minutes: float,
    min_frac: float = 0.80,
) -> float:
    """Max of rolling *window_hours*-h trailing MAX (peak-of-peak) with completeness guard.

    Like compute_rolling_max() but the rolling aggregation uses .max() instead of
    .mean(), returning the highest single value observed in any sufficiently complete
    window.  min_frac=0.80 (non-strict): at least 80 % of window slots must be
    populated; any window with fewer populated slots is NaN-dropped.
    Returns NaN if dt_minutes is invalid, series is empty, or no window passes the
    completeness threshold.
    """
    if math.isnan(dt_minutes) or dt_minutes <= 0 or len(series) == 0:
        return float("nan")

    valid = series.dropna()
    if len(valid) == 0:
        return float("nan")

    indexed = pd.Series(series.values, index=pd.DatetimeIndex(timestamps.values))
    indexed = indexed.sort_index()

    dt_round = max(1, round(dt_minutes))
    resampled = indexed.resample(f"{dt_round}min").mean()

    n_steps = int(round(window_hours * 60.0 / dt_round))
    if n_steps < 1:
        return float("nan")
    min_periods = max(1, int(min_frac * n_steps))

    rolling_max = resampled.rolling(window=n_steps, min_periods=min_periods).max()
    valid_max = rolling_max.dropna()
    if len(valid_max) == 0:
        return float("nan")
    return round(float(valid_max.max()), 2)


# ── Exceedance hours ─────────────────────────────────────────────────────────


def compute_exceedance_hours(
    series: pd.Series,
    threshold: float,
    dt_minutes: float,
) -> float:
    """Hours where *series* >= *threshold* (NaN excluded).

    Uses the step-function convention: count × (dt_minutes / 60).
    Returns 0.0 if dt_minutes is invalid.
    """
    if math.isnan(dt_minutes) or dt_minutes <= 0:
        return 0.0
    if math.isnan(threshold):
        return float("nan")
    valid = series.dropna()
    count = int((valid >= threshold).sum())
    return round(count * (dt_minutes / 60.0), 2)


# ── Per-year extreme orchestrator ────────────────────────────────────────────


def compute_extreme_yearly(
    dedup: pd.DataFrame,
    dt_minutes: float,
    t_p99: float,
    wb_p99: float | None,
    min_completeness_frac: float = 0.80,
) -> dict[str, Any]:
    """Compute all extreme / persistence metrics for one year.

    Parameters
    ----------
    dedup : deduplicated DataFrame with at least ``temp`` and ``timestamp``.
    dt_minutes : effective sampling interval for this year.
    t_p99 : year-local 99th percentile dry-bulb (for exceedance calc).
    wb_p99 : year-local 99th percentile wet-bulb, or None if unavailable.
    min_completeness_frac : rolling window completeness threshold.

    Returns
    -------
    dict ready to merge into the yearly summary row.
    """
    result: dict[str, Any] = {}

    ts = dedup["timestamp"]

    # Dry-bulb persistence
    result["tdb_mean_24h_max"] = compute_rolling_max(
        dedup["temp"], ts, 24, dt_minutes, min_completeness_frac
    )
    result["tdb_mean_72h_max"] = compute_rolling_max(
        dedup["temp"], ts, 72, dt_minutes, min_completeness_frac
    )

    # Exceedance — year-local Tdb threshold
    if not math.isnan(t_p99):
        result["exceedance_hours_tdb_p99"] = compute_exceedance_hours(
            dedup["temp"], t_p99, dt_minutes
        )
    else:
        result["exceedance_hours_tdb_p99"] = float("nan")

    # Twb — only if wetbulb present with valid data
    has_wb = "wetbulb_f" in dedup.columns and dedup["wetbulb_f"].notna().any()
    if has_wb and wb_p99 is not None and not math.isnan(wb_p99):
        result["exceedance_hours_twb_p99"] = compute_exceedance_hours(
            dedup["wetbulb_f"], wb_p99, dt_minutes
        )
    else:
        result["exceedance_hours_twb_p99"] = float("nan")

    return result


# ── Design day ───────────────────────────────────────────────────────────────

_DESIGN_DAY_COLUMNS = ("hour", "tdb", "twb", "rh", "wind_speed_kt", "wind_dir_deg")

_COL_MAP = {
    "tdb": "temp",
    "twb": "wetbulb_f",
    "rh": "relh",
    "wind_speed_kt": "wind_speed_kt",
    "wind_dir_deg": "drct_deg",
}


def _empty_design_day() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_DESIGN_DAY_COLUMNS))


def compute_design_day(
    df: pd.DataFrame,
    timestamps: pd.Series,
    dt_minutes: float,
    metric: str = "wetbulb_f",
    min_frac: float = 0.80,
) -> pd.DataFrame:
    """Generate a 24-hour worst-day profile from the hottest contiguous block.

    Parameters
    ----------
    df : deduplicated DataFrame (must include ``timestamp`` column).
    timestamps : timestamp series aligned to *df*.
    dt_minutes : sampling interval in minutes.
    metric : column to maximise (``"wetbulb_f"`` or ``"temp"``).
    min_frac : minimum data fraction for the rolling window.

    Returns
    -------
    24-row DataFrame (hour 0–23) with columns: hour, tdb, twb, rh,
    wind_speed_kt, wind_dir_deg.  Empty DataFrame if insufficient data.
    """
    if math.isnan(dt_minutes) or dt_minutes <= 0 or len(df) == 0:
        return _empty_design_day()

    # Determine metric column — fall back to temp if wetbulb unavailable
    if metric not in df.columns or not df[metric].notna().any():
        metric = "temp"
    if metric not in df.columns or not df[metric].notna().any():
        return _empty_design_day()

    # Coverage guard
    valid_frac = df[metric].notna().sum() / max(len(df), 1)
    if valid_frac < min_frac:
        return _empty_design_day()

    # Find hottest contiguous 24h block via rolling mean
    indexed = pd.Series(df[metric].values, index=pd.DatetimeIndex(timestamps.values))
    indexed = indexed.sort_index()

    dt_round = max(1, round(dt_minutes))
    freq_str = f"{dt_round}min"
    resampled = indexed.resample(freq_str).mean()

    n_steps = int(round(24 * 60.0 / dt_round))
    if n_steps < 1:
        return _empty_design_day()
    min_periods = max(1, int(min_frac * n_steps))

    rolling_mean = resampled.rolling(window=n_steps, min_periods=min_periods).mean()
    valid_means = rolling_mean.dropna()
    if len(valid_means) == 0:
        return _empty_design_day()

    # The rolling window end is at the index of the max — block starts n_steps earlier
    block_end_ts = valid_means.idxmax()
    block_start_ts = block_end_ts - pd.Timedelta(hours=24) + pd.Timedelta(minutes=dt_round)

    # Extract raw rows in [block_start, block_start + 24h)
    ts_index = pd.DatetimeIndex(timestamps.values)
    end_ts = block_start_ts + pd.Timedelta(hours=24)
    mask = (ts_index >= block_start_ts) & (ts_index < end_ts)
    block = df.loc[mask].copy()

    if block.empty:
        return _empty_design_day()

    # Resample to hourly: hour = floor((timestamp - block_start) / 1h)
    block_ts = pd.DatetimeIndex(block["timestamp"].values)
    hour_offsets = ((block_ts - block_start_ts).total_seconds() / 3600.0).astype(int)
    block = block.copy()
    block["_hour"] = np.clip(hour_offsets, 0, 23)

    # Build output — map columns
    rows = []
    for out_col, src_col in _COL_MAP.items():
        if src_col in block.columns:
            block[out_col] = block[src_col]
        else:
            block[out_col] = float("nan")

    grouped = block.groupby("_hour", as_index=False).agg(
        {col: "mean" for col in _COL_MAP if col in block.columns}
    )
    grouped = grouped.rename(columns={"_hour": "hour"})

    # Ensure all 24 hours present
    full_hours = pd.DataFrame({"hour": range(24)})
    result = full_hours.merge(grouped, on="hour", how="left")

    # Ensure all output columns exist
    for col in _DESIGN_DAY_COLUMNS:
        if col not in result.columns:
            result[col] = float("nan")

    return result[list(_DESIGN_DAY_COLUMNS)].round(2)
