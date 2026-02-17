"""Economizer and cooling tower decision metrics (deterministic, pure functions).

All metrics are screening proxies — not full psychrometric simulations.
Document assumptions clearly in any output report.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def compute_air_econ_hours(
    temps: pd.Series,
    threshold_f: float,
    dt_minutes: float,
) -> float:
    """Count hours where dry-bulb <= threshold_f (step function, NaN-safe).

    Parameters
    ----------
    temps : pd.Series of dry-bulb temperatures (°F); may contain NaN.
    threshold_f : airside economizer dry-bulb threshold (°F).
    dt_minutes : inferred sampling interval in minutes.

    Returns
    -------
    float — hours where Tdb <= threshold_f. NaN if dt_minutes is invalid.
    """
    if math.isnan(dt_minutes) or dt_minutes <= 0:
        return float("nan")
    valid = temps.dropna()
    count = int((valid <= threshold_f).sum())
    return round(count * (dt_minutes / 60.0), 2)


def compute_wec_hours(
    wb: pd.Series,
    chw_supply_f: float,
    tower_approach_f: float,
    hx_approach_f: float,
    dt_minutes: float,
) -> tuple[float, float, float, float]:
    """Compute waterside-economizer proxy hours.

    Physical temperature chain (screening-level):
      Tower delivers: CWS ≈ Twb + tower_approach
      Plate HX delivers: CHWS ≈ CWS + hx_approach
    WEC is feasible (full) when: Twb + tower_approach + hx_approach <= chw_supply

    Therefore:
      required_twb_max = chw_supply_f - tower_approach_f - hx_approach_f
      wec_hours = count(Twb <= required_twb_max AND not NaN) * dt_hours
      wec_feasible_pct = wec_hours / hours_with_wetbulb

    Parameters
    ----------
    wb : pd.Series of wet-bulb temperatures (°F); may contain NaN.
    chw_supply_f : chilled water supply temperature (°F).
    tower_approach_f : tower approach delta (°F).
    hx_approach_f : heat-exchanger approach delta (°F) for the plate HX.
    dt_minutes : inferred sampling interval in minutes.

    Returns
    -------
    tuple of (required_twb_max, wec_hours, wec_feasible_pct, hours_with_wetbulb).
    wec_hours and wec_feasible_pct are NaN when wb has no valid data or
    dt_minutes is invalid. hours_with_wetbulb is 0.0 when wb is all-NaN,
    NaN when dt_minutes is invalid.
    """
    required_twb_max = chw_supply_f - tower_approach_f - hx_approach_f
    if math.isnan(dt_minutes) or dt_minutes <= 0:
        return required_twb_max, float("nan"), float("nan"), float("nan")
    valid = wb.dropna()
    if len(valid) == 0:
        return required_twb_max, float("nan"), float("nan"), 0.0
    hours_with_wetbulb = round(float(len(valid)) * (dt_minutes / 60.0), 2)
    wec_count = int((valid <= required_twb_max).sum())
    wec_hours = round(wec_count * (dt_minutes / 60.0), 2)
    wec_feasible_pct = round(wec_hours / hours_with_wetbulb, 6) if hours_with_wetbulb > 0 else float("nan")
    return required_twb_max, wec_hours, wec_feasible_pct, hours_with_wetbulb


def compute_tower_stress_hours(
    wb: pd.Series,
    thresholds: list[float],
    dt_minutes: float,
) -> dict[str, float]:
    """Count hours where wet-bulb >= each threshold (step function, NaN-safe).

    Parameters
    ----------
    wb : pd.Series of wet-bulb temperatures (°F); may contain NaN.
    thresholds : list of wet-bulb thresholds (°F); uses >= (inclusive).
    dt_minutes : inferred sampling interval in minutes.

    Returns
    -------
    dict keyed as "tower_stress_hours_wb_gt_<int(thr)>" with float values.
    NaN if dt_minutes is invalid.
    """
    result: dict[str, float] = {}
    if math.isnan(dt_minutes) or dt_minutes <= 0:
        for thr in thresholds:
            result[f"tower_stress_hours_wb_gt_{int(thr)}"] = float("nan")
        return result
    valid = wb.dropna()
    if len(valid) == 0:
        for thr in thresholds:
            result[f"tower_stress_hours_wb_gt_{int(thr)}"] = float("nan")
        return result
    for thr in thresholds:
        count = int((valid >= thr).sum())
        result[f"tower_stress_hours_wb_gt_{int(thr)}"] = round(count * (dt_minutes / 60.0), 2)
    return result


def compute_rolling_wb_max(
    wb: pd.Series,
    timestamps: pd.Series,
    window_hours: int,
    dt_minutes: float,
    min_completeness_frac: float = 0.80,
) -> float:
    """Max rolling <window_hours>h mean wet-bulb with completeness guard.

    Resamples the deduplicated wet-bulb series onto a regular dt_minutes grid
    first, then applies an integer-step rolling window. This ensures uniform
    spacing before computing min_periods based on expected sample count.

    Parameters
    ----------
    wb : pd.Series of wet-bulb temperatures (°F); may contain NaN.
    timestamps : pd.Series of timestamps (tz-aware) aligned to wb.
    window_hours : rolling window size in hours (24 or 72).
    dt_minutes : inferred sampling interval in minutes.
    min_completeness_frac : minimum fraction of expected samples required
        for a window to count (default 0.80).

    Returns
    -------
    float — max of valid rolling means. NaN if dt_minutes invalid, wb has
    no valid data, or no window meets the completeness threshold.
    """
    if math.isnan(dt_minutes) or dt_minutes <= 0 or len(wb) == 0:
        return float("nan")

    # Build time-indexed series and resample to regular grid
    indexed = pd.Series(wb.values, index=pd.DatetimeIndex(timestamps.values))
    indexed = indexed.sort_index()

    dt_round = max(1, round(dt_minutes))
    freq_str = f"{dt_round}min"
    resampled = indexed.resample(freq_str).mean()  # NaN for gaps in grid

    n_steps = int(round(window_hours * 60.0 / dt_round))
    if n_steps < 1:
        return float("nan")
    min_periods = max(1, int(min_completeness_frac * n_steps))

    rolling_mean = resampled.rolling(window=n_steps, min_periods=min_periods).mean()
    valid_means = rolling_mean.dropna()
    if len(valid_means) == 0:
        return float("nan")
    return round(float(valid_means.max()), 2)


def compute_lwt_proxy_metrics(
    wb: pd.Series,
    tower_approach_f: float,
) -> dict[str, float]:
    """Compute leaving-water-temperature proxy percentiles.

    lwt_proxy_f = wetbulb_f + tower_approach_f

    This is a proxy only — actual condenser water LWT depends on tower
    specifications, flow rates, and heat load.

    Parameters
    ----------
    wb : pd.Series of wet-bulb temperatures (°F); may contain NaN.
    tower_approach_f : tower approach delta (°F).

    Returns
    -------
    dict with keys "lwt_proxy_p99" and "lwt_proxy_max".
    Both are NaN if wb has no valid data.
    """
    valid = wb.dropna()
    if len(valid) == 0:
        return {"lwt_proxy_p99": float("nan"), "lwt_proxy_max": float("nan")}
    lwt = valid + tower_approach_f
    return {
        "lwt_proxy_p99": round(float(np.percentile(lwt, 99)), 2),
        "lwt_proxy_max": round(float(lwt.max()), 2),
    }


def compute_econ_tower_yearly(
    dedup: pd.DataFrame,
    dt_minutes: float,
    air_econ_threshold_f: float,
    chw_supply_f: float,
    tower_approach_f: float,
    hx_approach_f: float,
    wb_stress_thresholds: list[float],
    min_completeness_frac: float = 0.80,
) -> dict[str, Any]:
    """Compute all econ/tower metrics for one year slice (deduplicated rows).

    Parameters
    ----------
    dedup : deduplicated year-slice DataFrame with 'temp' and 'timestamp'
            columns; 'wetbulb_f' column used when present.
    dt_minutes : inferred sampling interval in minutes.
    air_econ_threshold_f : dry-bulb threshold for airside econ hours (°F).
    chw_supply_f : chilled water supply temperature (°F).
    tower_approach_f : tower approach delta (°F).
    hx_approach_f : heat-exchanger approach delta (°F).
    wb_stress_thresholds : list of wet-bulb thresholds for tower stress (°F).
    min_completeness_frac : completeness fraction for rolling window.

    Returns
    -------
    dict with econ/tower metric columns ready to merge into yearly summary row.
    """
    result: dict[str, Any] = {}

    # Airside econ
    result["air_econ_threshold_f"] = air_econ_threshold_f
    result["air_econ_hours"] = compute_air_econ_hours(
        dedup["temp"], air_econ_threshold_f, dt_minutes
    )

    wb_available = "wetbulb_f" in dedup.columns
    wb = dedup["wetbulb_f"] if wb_available else pd.Series([], dtype=float)

    # WEC proxy
    required_twb_max, wec_hours, wec_feasible_pct, hours_with_wetbulb = compute_wec_hours(
        wb, chw_supply_f, tower_approach_f, hx_approach_f, dt_minutes
    )
    result["required_twb_max"] = required_twb_max
    result["wec_hours"] = wec_hours
    result["wec_feasible_pct"] = wec_feasible_pct
    result["hours_with_wetbulb"] = hours_with_wetbulb

    # Tower stress hours
    stress = compute_tower_stress_hours(wb, wb_stress_thresholds, dt_minutes)
    result.update(stress)

    # Rolling wb maxima + LWT proxy
    if wb_available:
        ts = dedup["timestamp"]
        result["wb_mean_24h_max"] = compute_rolling_wb_max(
            wb, ts, 24, dt_minutes, min_completeness_frac
        )
        result["wb_mean_72h_max"] = compute_rolling_wb_max(
            wb, ts, 72, dt_minutes, min_completeness_frac
        )
        lwt = compute_lwt_proxy_metrics(wb, tower_approach_f)
        result.update(lwt)
    else:
        result["wb_mean_24h_max"] = float("nan")
        result["wb_mean_72h_max"] = float("nan")
        result["lwt_proxy_p99"] = float("nan")
        result["lwt_proxy_max"] = float("nan")

    return result
