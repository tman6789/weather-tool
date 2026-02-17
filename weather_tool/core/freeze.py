"""Freeze risk metrics (deterministic, pure functions).

All metrics use a step-function approach on deduplicated observations,
consistent with hours_above_ref and econ_tower conventions.

Freeze condition (inclusive): Tdb <= freeze_threshold_f AND Tdb not NaN
NaN temperatures are treated as non-freeze and break event continuity (conservative).
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd


def compute_freeze_hours(
    temps: pd.Series,
    freeze_threshold_f: float,
    dt_minutes: float,
) -> float:
    """Hours where Tdb <= freeze_threshold_f (step function, NaN-safe).

    Parameters
    ----------
    temps : pd.Series of dry-bulb temperatures (°F); may contain NaN.
    freeze_threshold_f : freeze threshold (°F); inclusive (<=).
    dt_minutes : inferred sampling interval in minutes.

    Returns
    -------
    float — hours where Tdb <= threshold. NaN if dt_minutes is invalid.
    """
    if math.isnan(dt_minutes) or dt_minutes <= 0:
        return float("nan")
    valid = temps.dropna()
    count = int((valid <= freeze_threshold_f).sum())
    return round(count * (dt_minutes / 60.0), 2)


def compute_total_hours_with_temp(
    temps: pd.Series,
    dt_minutes: float,
) -> float:
    """Total hours with non-NaN temperature (denominator for freeze_hours_pct).

    Parameters
    ----------
    temps : pd.Series of temperatures; may contain NaN.
    dt_minutes : inferred sampling interval in minutes.

    Returns
    -------
    float — count(not NaN) * dt_hours. NaN if dt_minutes is invalid.
    """
    if math.isnan(dt_minutes) or dt_minutes <= 0:
        return float("nan")
    return round(float(temps.notna().sum()) * (dt_minutes / 60.0), 2)


def compute_freeze_shoulder_hours(
    timestamps: pd.Series,
    temps: pd.Series,
    freeze_threshold_f: float,
    dt_minutes: float,
    shoulder_months: list[int],
) -> float:
    """Freeze hours restricted to shoulder months.

    Parameters are series (not a DataFrame) to keep this function
    column-name agnostic.

    Parameters
    ----------
    timestamps : pd.Series of tz-aware timestamps aligned to temps.
    temps : pd.Series of dry-bulb temperatures (°F); may contain NaN.
    freeze_threshold_f : freeze threshold (°F).
    dt_minutes : inferred sampling interval in minutes.
    shoulder_months : list of month numbers (1=Jan … 12=Dec).

    Returns
    -------
    float — freeze hours in shoulder months. NaN if dt_minutes is invalid.
    """
    mask = timestamps.dt.month.isin(shoulder_months)
    return compute_freeze_hours(temps[mask].reset_index(drop=True), freeze_threshold_f, dt_minutes)


def detect_freeze_events(
    timestamps: pd.Series,
    temps: pd.Series,
    freeze_threshold_f: float,
    dt_minutes: float,
    min_event_hours: float,
    gap_tolerance_mult: float,
) -> dict[str, Any]:
    """Detect contiguous freeze events with gap-tolerance rule.

    A run is broken when ANY of:
      - consecutive timestamp gap > gap_break_minutes, OR
      - the current observation is NaN (no data = no freeze; conservative), OR
      - the current observation is not freeze-condition

    NaN temperatures force a run break (conservative approach for engineering
    risk analysis — missing data should not extend freeze events).

    Parameters
    ----------
    timestamps : pd.Series of tz-aware timestamps (aligned to temps).
    temps : pd.Series of dry-bulb temperatures (°F); NaN breaks continuity.
    freeze_threshold_f : freeze threshold (°F).
    dt_minutes : inferred sampling interval in minutes.
    min_event_hours : minimum event duration to qualify (hours).
    gap_tolerance_mult : gap > (mult * dt_minutes) breaks event continuity.

    Returns
    -------
    dict with:
      freeze_event_count               — int, 0 if no qualifying events.
      freeze_event_max_duration_hours  — float, NaN if count == 0.
    """
    if math.isnan(dt_minutes) or dt_minutes <= 0:
        return {"freeze_event_count": 0, "freeze_event_max_duration_hours": float("nan")}

    gap_break_minutes = gap_tolerance_mult * dt_minutes
    dt_h = dt_minutes / 60.0

    # Use ALL observations sorted by timestamp (NaNs break continuity, not dropped)
    ts_arr = timestamps.reset_index(drop=True)
    t_arr = temps.reset_index(drop=True)

    if len(t_arr) == 0:
        return {"freeze_event_count": 0, "freeze_event_max_duration_hours": float("nan")}

    events: list[float] = []
    run_count = 0
    prev_ts = None

    for i in range(len(t_arr)):
        raw = t_arr.iloc[i]
        is_nan = pd.isna(raw)
        is_freeze = (not is_nan) and (float(raw) <= freeze_threshold_f)

        gap_breaks = False
        if prev_ts is not None:
            gap_min = (ts_arr.iloc[i] - prev_ts).total_seconds() / 60.0
            gap_breaks = gap_min > gap_break_minutes

        if is_freeze and not gap_breaks:
            run_count += 1
        else:
            # Flush current run
            if run_count > 0:
                duration = run_count * dt_h
                if duration >= min_event_hours:
                    events.append(round(duration, 2))
            run_count = 1 if is_freeze else 0

        prev_ts = ts_arr.iloc[i]

    # Flush last run
    if run_count > 0:
        duration = run_count * dt_h
        if duration >= min_event_hours:
            events.append(round(duration, 2))

    return {
        "freeze_event_count": len(events),
        "freeze_event_max_duration_hours": round(max(events), 2) if events else float("nan"),
    }


def compute_freeze_yearly(
    dedup: pd.DataFrame,
    dt_minutes: float,
    cfg: Any,
) -> dict[str, Any]:
    """Compute all freeze metrics for one year slice (deduplicated rows).

    Parameters
    ----------
    dedup : deduplicated year-slice DataFrame with 'temp' and 'timestamp' columns.
    dt_minutes : inferred sampling interval in minutes.
    cfg : RunConfig with freeze_threshold_f, freeze_min_event_hours,
          freeze_gap_tolerance_mult, freeze_shoulder_months.

    Returns
    -------
    dict with freeze metric columns ready to merge into yearly summary row.
    """
    result: dict[str, Any] = {}
    result["freeze_threshold_f"] = cfg.freeze_threshold_f
    result["freeze_min_event_hours"] = cfg.freeze_min_event_hours

    temps = dedup["temp"]
    timestamps = dedup["timestamp"]

    freeze_hrs = compute_freeze_hours(temps, cfg.freeze_threshold_f, dt_minutes)
    total_hrs = compute_total_hours_with_temp(temps, dt_minutes)
    result["freeze_hours"] = freeze_hrs
    result["total_hours_with_temp"] = total_hrs

    # Percentage: NaN when no temp data (total == 0 or NaN — no data ≠ 0% freeze)
    if not math.isnan(freeze_hrs) and not math.isnan(total_hrs) and total_hrs > 0:
        result["freeze_hours_pct"] = round(freeze_hrs / total_hrs, 6)
    else:
        result["freeze_hours_pct"] = float("nan")

    # Shoulder season (pass series — column-name agnostic)
    result["freeze_hours_shoulder"] = compute_freeze_shoulder_hours(
        timestamps, temps, cfg.freeze_threshold_f, dt_minutes, cfg.freeze_shoulder_months
    )

    # Event detection (NaN obs break event continuity — conservative)
    event_metrics = detect_freeze_events(
        timestamps, temps,
        cfg.freeze_threshold_f, dt_minutes,
        cfg.freeze_min_event_hours, cfg.freeze_gap_tolerance_mult,
    )
    result.update(event_metrics)

    return result
