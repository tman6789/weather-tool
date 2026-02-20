"""Deterministic Death Day / Heat Day event finder.

Finds the top-N rolling windows of maximum combined thermal stress
(Tdb + Twb co-occurrence) over a configurable window length.

Inputs contract
---------------
Required df columns:
  - timestamp (tz-aware)
  - temp      (float, may contain NaN)

Optional df columns (silently absent → None in output):
  - wetbulb_f      enables "death_day" mode; absent → "heat_day" mode
  - relh           rh_mean_pct in output
  - wind_speed_kt  wind_mean_kt in output
  - is_calm        calm_pct in output

The caller is responsible for passing a deduplicated DataFrame
(i.e., rows where _is_dup == False have already been filtered out).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from weather_tool.config import DEATH_DAY_STRESS_EPS, ROLLING_COMPLETENESS_MIN_FRAC


def find_death_day_candidates(
    df: pd.DataFrame,
    dt_minutes: float,
    window_hours: int,
    tdb_p99: float,
    tdb_p996: float,
    wb_p99: float | None,
    wb_p996: float | None,
    top_n: int = 5,
    min_frac: float = ROLLING_COMPLETENESS_MIN_FRAC,
) -> list[dict[str, Any]]:
    """Find the top-N rolling windows with highest combined thermal stress.

    Parameters
    ----------
    df : deduplicated windowed DataFrame.  Must contain 'timestamp' and 'temp'.
    dt_minutes : inferred sampling interval in minutes.
    window_hours : length of the rolling event window in hours.
    tdb_p99 : multi-year median 99th percentile dry-bulb (baseline).
    tdb_p996 : multi-year median 99.6th percentile dry-bulb (baseline).
    wb_p99 : multi-year median 99th percentile wet-bulb, or None.
    wb_p996 : multi-year median 99.6th percentile wet-bulb, or None.
    top_n : maximum number of non-overlapping candidates to return.
    min_frac : minimum data completeness fraction for rolling window.

    Returns
    -------
    list of candidate dicts, sorted by stress_score descending (rank 1 = worst).
    Returns [] on invalid inputs or insufficient data.
    """
    # ── Guard rails ────────────────────────────────────────────────────────────
    if math.isnan(dt_minutes) or dt_minutes <= 0:
        return []
    if len(df) == 0:
        return []
    if "temp" not in df.columns or not df["temp"].notna().any():
        return []
    if math.isnan(tdb_p99) or math.isnan(tdb_p996):
        return []

    # ── Mode selection ─────────────────────────────────────────────────────────
    has_wb = (
        "wetbulb_f" in df.columns
        and df["wetbulb_f"].notna().sum() >= max(1, int(min_frac * len(df)))
        and wb_p99 is not None
        and wb_p996 is not None
    )
    mode = "death_day" if has_wb else "heat_day"
    confidence = "high" if has_wb else "low"

    # ── Resample to uniform grid ────────────────────────────────────────────────
    # Pattern mirrors core/extreme.py:compute_rolling_max lines 38-50.
    # We need the full pd.Series (not just the scalar max) to select top-N timestamps.
    dt_round = max(1, round(dt_minutes))
    freq_str = f"{dt_round}min"
    n_steps = int(round(window_hours * 60.0 / dt_round))
    if n_steps < 1:
        return []
    min_periods = max(1, int(min_frac * n_steps))

    ts_index = pd.DatetimeIndex(df["timestamp"].values)

    tdb_indexed = pd.Series(df["temp"].values, index=ts_index).sort_index()
    tdb_resampled = tdb_indexed.resample(freq_str).mean()
    tdb_roll = tdb_resampled.rolling(window=n_steps, min_periods=min_periods).mean()

    if has_wb:
        wb_indexed = pd.Series(df["wetbulb_f"].values, index=ts_index).sort_index()
        wb_resampled = wb_indexed.resample(freq_str).mean()
        wb_roll = wb_resampled.rolling(window=n_steps, min_periods=min_periods).mean()

    # ── Stress z-scores ────────────────────────────────────────────────────────
    EPS = DEATH_DAY_STRESS_EPS

    tdb_range = max(EPS, tdb_p996 - tdb_p99)
    z_tdb = (tdb_roll - tdb_p99) / tdb_range   # pd.Series

    if has_wb:
        wb_range = max(EPS, wb_p996 - wb_p99)  # type: ignore[operator]
        z_wb = (wb_roll - wb_p99) / wb_range    # type: ignore[operator]
        stress = 0.5 * z_tdb + 0.5 * z_wb
    else:
        z_wb = pd.Series(np.nan, index=tdb_roll.index, dtype=float)
        stress = z_tdb

    valid_stress = stress.dropna()
    if len(valid_stress) == 0:
        return []

    # ── De-duplicate overlapping windows ──────────────────────────────────────
    # Oversample by 3× then greedily keep non-overlapping candidates.
    # Rolling window label is the *end* timestamp (pandas convention).
    # Two windows of length W are non-overlapping iff their start times are >= W apart.
    window_td = pd.Timedelta(hours=window_hours)
    step_td = pd.Timedelta(minutes=dt_round)

    top_indices = valid_stress.nlargest(top_n * 3).index
    accepted: list[pd.Timestamp] = []

    for ts in top_indices:
        w_start = ts - window_td + step_td
        overlaps = any(
            abs(w_start - (prev - window_td + step_td)) < window_td
            for prev in accepted
        )
        if not overlaps:
            accepted.append(ts)
        if len(accepted) >= top_n:
            break

    if not accepted:
        return []

    # ── Build per-candidate stats ──────────────────────────────────────────────
    candidates: list[dict[str, Any]] = []

    for rank, ts in enumerate(accepted, 1):
        window_end_ts = ts
        window_start_ts = ts - window_td + step_td

        mask = (ts_index >= window_start_ts) & (ts_index <= window_end_ts)
        block = df.loc[mask]

        tdb_vals = block["temp"].dropna()
        wb_vals = (
            block["wetbulb_f"].dropna()
            if "wetbulb_f" in block.columns
            else pd.Series(dtype=float)
        )
        rh_vals = (
            block["relh"].dropna()
            if "relh" in block.columns
            else pd.Series(dtype=float)
        )
        wind_vals = (
            block["wind_speed_kt"].dropna()
            if "wind_speed_kt" in block.columns
            else pd.Series(dtype=float)
        )
        calm_series = (
            block["is_calm"]
            if "is_calm" in block.columns
            else pd.Series(dtype=bool)
        )

        # Pull scalar z-scores at this window's end timestamp
        z_tdb_val = float(z_tdb.get(ts, np.nan))
        z_wb_val = float(z_wb.get(ts, np.nan)) if has_wb else None
        stress_val = float(stress.get(ts, np.nan))

        candidate: dict[str, Any] = {
            "rank":         rank,
            "mode":         mode,
            "confidence":   confidence,
            "window_hours": window_hours,
            "start_ts":     window_start_ts.isoformat(),
            "end_ts":       window_end_ts.isoformat(),
            "stress_score": round(stress_val, 4) if not math.isnan(stress_val) else None,
            "z_tdb":        round(z_tdb_val, 4) if not math.isnan(z_tdb_val) else None,
            "z_wb":         round(z_wb_val, 4) if z_wb_val is not None and not math.isnan(z_wb_val) else None,
            "tdb_mean_f":   round(float(tdb_vals.mean()), 2) if len(tdb_vals) > 0 else None,
            "tdb_max_f":    round(float(tdb_vals.max()), 2) if len(tdb_vals) > 0 else None,
            "twb_mean_f":   round(float(wb_vals.mean()), 2) if len(wb_vals) > 0 else None,
            "twb_max_f":    round(float(wb_vals.max()), 2) if len(wb_vals) > 0 else None,
            "rh_mean_pct":  round(float(rh_vals.mean()), 2) if len(rh_vals) > 0 else None,
            "wind_mean_kt": round(float(wind_vals.mean()), 2) if len(wind_vals) > 0 else None,
            "calm_pct":     round(float(calm_series.sum()) / max(len(calm_series), 1), 4) if len(calm_series) > 0 else None,
        }
        candidates.append(candidate)

    return candidates
