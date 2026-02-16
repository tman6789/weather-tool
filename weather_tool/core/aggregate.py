"""Yearly (and windowed) aggregation — produces the Summary table."""

from __future__ import annotations

import calendar
from datetime import date
from typing import Any

import math

import numpy as np
import pandas as pd

from weather_tool.config import FREEZE_THRESHOLD_F, PARTIAL_COVERAGE_THRESHOLD, RunConfig
from weather_tool.core.metrics import hours_above_ref
from weather_tool.core.quality import compute_quality


def cooling_degree_hours(temps: pd.Series, ref_temp: float, dt_minutes: float) -> float:
    """Integral of (temp - ref_temp) * dt_hours for all non-NaN obs above ref_temp.

    Uses a step-function approach on deduplicated rows (same convention as hours_above_ref).
    """
    if math.isnan(dt_minutes) or dt_minutes <= 0:
        return 0.0
    valid = temps.dropna()
    excess = (valid - ref_temp).clip(lower=0.0)
    return float(excess.sum() * (dt_minutes / 60.0))


def _year_slice_bounds(
    year: int,
    window_start: date,
    window_end: date,
    tz: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (slice_start, slice_end) for a given year, clamped to the analysis window."""
    jan1 = pd.Timestamp(f"{year}-01-01", tz=tz)
    dec31 = pd.Timestamp(f"{year}-12-31 23:59:59", tz=tz)

    ws = pd.Timestamp(
        f"{window_start.year}-{window_start.month:02d}-{window_start.day:02d}",
        tz=tz,
    )
    we = pd.Timestamp(
        f"{window_end.year}-{window_end.month:02d}-{window_end.day:02d} 23:59:59",
        tz=tz,
    )

    slice_start = max(jan1, ws)
    slice_end = min(dec31, we)
    return slice_start, slice_end


def _coverage(year: int, slice_start: pd.Timestamp, slice_end: pd.Timestamp) -> float:
    """Fraction of the full year covered by [slice_start, slice_end].

    slice_end is expressed as HH:MM:59 (inclusive), so add 1 second to recover
    the full day before dividing, ensuring a Jan 1 – Dec 31 window reports 1.0.
    """
    days_in_year = 366 if calendar.isleap(year) else 365
    slice_days = ((slice_end - slice_start).total_seconds() + 1.0) / 86400.0
    return min(slice_days / days_in_year, 1.0)


def build_yearly_summary(
    df: pd.DataFrame,
    cfg: RunConfig,
    global_interval: dict[str, Any],
) -> pd.DataFrame:
    """Build the per-year summary table.

    Parameters
    ----------
    df : normalized, window-filtered DataFrame (timestamp, temp, station_id, _is_dup).
    cfg : run configuration.
    global_interval : dict from ``infer_interval`` on the full dataset.

    Returns
    -------
    pd.DataFrame with the full summary schema.
    """
    dt_minutes = global_interval["dt_minutes"]
    interval_change_flag = global_interval["interval_change_flag"]

    station = cfg.station_id or "csv"

    # Determine years present
    df = df.copy()
    df["_year"] = df["timestamp"].dt.year
    years = sorted(df["_year"].unique())

    rows: list[dict[str, Any]] = []

    for yr in years:
        slice_start, slice_end = _year_slice_bounds(yr, cfg.start, cfg.end, cfg.tz)
        yr_df = df[df["_year"] == yr].copy()

        # Quality metrics
        quality = compute_quality(yr_df, slice_start, slice_end, dt_minutes, interval_change_flag)

        # Temp metrics (on non-dup rows only for uniqueness; but temp extremes use all)
        valid_temps = yr_df["temp"].dropna()
        tmax = float(valid_temps.max()) if len(valid_temps) > 0 else float("nan")
        tmin = float(valid_temps.min()) if len(valid_temps) > 0 else float("nan")

        # Hours above ref — use deduplicated rows
        dedup = yr_df[~yr_df["_is_dup"]]
        hab = hours_above_ref(dedup["temp"], cfg.ref_temp, dt_minutes)

        # Dry-bulb percentiles
        dedup_temps = dedup["temp"].dropna()
        t_p99 = float(np.percentile(dedup_temps, 99)) if len(dedup_temps) > 0 else float("nan")
        t_p996 = float(np.percentile(dedup_temps, 99.6)) if len(dedup_temps) > 0 else float("nan")

        # Cooling degree hours and freeze hours
        cdh = cooling_degree_hours(dedup["temp"], cfg.ref_temp, dt_minutes)
        freeze_hrs = hours_above_ref(-dedup["temp"], -FREEZE_THRESHOLD_F, dt_minutes)

        # Wet-bulb percentiles (only if wetbulb_f column present)
        wb_row: dict[str, Any] = {}
        if "wetbulb_f" in dedup.columns:
            wb_valid = dedup["wetbulb_f"].dropna()
            if len(wb_valid) > 0:
                wb_row["wb_p99"] = round(float(np.percentile(wb_valid, 99)), 2)
                wb_row["wb_p996"] = round(float(np.percentile(wb_valid, 99.6)), 2)
                wb_row["wb_max"] = round(float(wb_valid.max()), 2)
                wb_row["wb_mean"] = round(float(wb_valid.mean()), 2)
                wb_row["hours_wb_above_ref"] = round(
                    hours_above_ref(dedup["wetbulb_f"], cfg.ref_temp, dt_minutes), 2
                )
            else:
                wb_row["wb_p99"] = None
                wb_row["wb_p996"] = None
                wb_row["wb_max"] = None
                wb_row["wb_mean"] = None
                wb_row["hours_wb_above_ref"] = 0.0

        cov = _coverage(yr, slice_start, slice_end)
        partial = cov < PARTIAL_COVERAGE_THRESHOLD

        rows.append(
            {
                "station_id": station,
                "year": yr,
                "window_start": slice_start.date(),
                "window_end": slice_end.date(),
                "dt_minutes": round(dt_minutes, 2),
                "tmax": round(tmax, 2) if not np.isnan(tmax) else None,
                "tmin": round(tmin, 2) if not np.isnan(tmin) else None,
                "hours_above_ref": round(hab, 2),
                "t_p99": round(t_p99, 2) if not np.isnan(t_p99) else None,
                "t_p996": round(t_p996, 2) if not np.isnan(t_p996) else None,
                "cooling_degree_hours": round(cdh, 2),
                "hours_below_32": round(freeze_hrs, 2),
                "ref_temp": cfg.ref_temp,
                **wb_row,
                **quality,
                "coverage_pct": round(cov, 6),
                "partial_coverage_flag": partial,
            }
        )

    summary = pd.DataFrame(rows)
    return summary
