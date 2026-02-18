"""Window-aggregation and scoring for multi-station climate comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from weather_tool.config import (
    ECON_WB_COVERAGE_MIN_FRAC,
    FREEZE_CONFIDENCE_TEMP_AVAIL_MIN_FRAC,
    FREEZE_THRESHOLD_F,
    MISSING_DATA_WARNING_THRESHOLD,
    WETBULB_AVAIL_WARNING_THRESHOLD,
)
from weather_tool.core.compare_scores import compute_scores
from weather_tool.core.metrics import hours_above_ref

if TYPE_CHECKING:
    from weather_tool.pipeline import StationResult


def aggregate_station_window(
    summary: pd.DataFrame,
    windowed: pd.DataFrame,
    interval_info: dict[str, Any],
    ref_temps: list[float],
    station_id: str,
) -> dict[str, Any]:
    """Compute all window-aggregated metrics for one station.

    Parameters
    ----------
    summary : yearly summary DataFrame from build_yearly_summary
    windowed : normalized, window-filtered time series (with _is_dup column)
    interval_info : dict from infer_interval
    ref_temps : list of reference temperature thresholds
    station_id : identifier string

    Returns
    -------
    dict with all window-aggregated metrics for this station.
    """
    dt = interval_info["dt_minutes"]
    row: dict[str, Any] = {"station_id": station_id}

    # Guard: empty summary (no data in the analysis window) — return a sentinel
    # row so the compare DataFrame can still be assembled without crashing.
    if len(summary) == 0:
        sentinel: dict[str, Any] = {
            "station_id": station_id,
            "years_covered_count": 0,
            "dt_minutes_median": float("nan"),
            "coverage_weighted_pct": float("nan"),
            "partial_coverage_years_count": 0,
            "timestamp_missing_pct_avg": float("nan"),
            "interval_change_flag_any": False,
            "tmax_max": float("nan"),
            "tmin_min": float("nan"),
            "freeze_hours_sum": 0.0,
            "freeze_hours_shoulder_sum": 0.0,
            "total_hours_with_temp_sum": 0.0,
            "freeze_hours_pct_over_window": float("nan"),
            "freeze_event_count_sum": 0,
            "freeze_event_max_duration_hours_max": float("nan"),
            "freeze_confidence_flag": True,
            "econ_confidence_flag": True,
            "missing_data_warning": True,
        }
        for rt in ref_temps:
            rt_key = int(rt)
            sentinel[f"hours_above_ref_{rt_key}_sum"] = 0.0
            sentinel[f"cooling_degree_hours_{rt_key}_sum"] = 0.0
        return sentinel

    # Coverage / completeness
    row["years_covered_count"] = int(len(summary))
    row["dt_minutes_median"] = float(np.nanmedian(summary["dt_minutes"].astype(float)))
    row["coverage_weighted_pct"] = float(np.nanmean(summary["coverage_pct"].astype(float)))
    row["partial_coverage_years_count"] = int(summary["partial_coverage_flag"].sum())

    # Missing timestamps: NaN-safe mean (interval_unknown_flag rows have NaN missing_pct)
    valid_mp = summary["missing_pct"].dropna()
    row["timestamp_missing_pct_avg"] = float(valid_mp.mean()) if len(valid_mp) > 0 else float("nan")

    row["interval_change_flag_any"] = bool(summary["interval_change_flag"].any())

    # Wet-bulb availability (only if column present)
    if "wetbulb_availability_pct" in summary.columns:
        row["wetbulb_availability_pct_avg"] = float(
            np.nanmean(summary["wetbulb_availability_pct"].astype(float))
        )

    # Temperature extremes
    row["tmax_max"] = float(summary["tmax"].dropna().max()) if "tmax" in summary.columns and summary["tmax"].notna().any() else float("nan")
    row["tmin_min"] = float(summary["tmin"].dropna().min()) if "tmin" in summary.columns and summary["tmin"].notna().any() else float("nan")

    # Dry-bulb percentiles (median across years — new columns added in aggregate.py)
    if "t_p99" in summary.columns and summary["t_p99"].notna().any():
        row["t_p99_median"] = float(np.nanmedian(summary["t_p99"].astype(float)))
    if "t_p996" in summary.columns and summary["t_p996"].notna().any():
        row["t_p996_median"] = float(np.nanmedian(summary["t_p996"].astype(float)))

    # Hours above ref and CDH for each requested threshold — use per-year dt
    # from the summary so that mixed-interval stations get correct hour counts.
    dedup = windowed.loc[~windowed["_is_dup"]].copy()
    dedup["_year"] = dedup["timestamp"].dt.year

    dt_by_year: dict[int, float] = {}
    for _, yr_row in summary.iterrows():
        yr_dt = float(yr_row["dt_minutes"])
        dt_by_year[int(yr_row["year"])] = dt if np.isnan(yr_dt) else yr_dt

    for rt in ref_temps:
        rt_key = int(rt)
        hab_total = 0.0
        cdh_total = 0.0
        for yr, yr_dedup in dedup.groupby("_year"):
            yr_dt = dt_by_year.get(int(yr), dt)
            hab_total += hours_above_ref(yr_dedup["temp"], rt, yr_dt)
            if not (np.isnan(yr_dt) or yr_dt <= 0):
                excess = (yr_dedup["temp"].dropna() - rt).clip(lower=0.0)
                cdh_total += float(excess.sum() * (yr_dt / 60.0))
        row[f"hours_above_ref_{rt_key}_sum"] = round(hab_total, 2)
        row[f"cooling_degree_hours_{rt_key}_sum"] = round(cdh_total, 2)

    # Freeze hours — prefer new column, fall back to legacy name, then direct computation
    if "freeze_hours" in summary.columns:
        row["freeze_hours_sum"] = round(float(summary["freeze_hours"].dropna().sum()), 2)
    elif "hours_below_32" in summary.columns:
        row["freeze_hours_sum"] = round(float(summary["hours_below_32"].sum()), 2)
    else:
        row["freeze_hours_sum"] = round(
            hours_above_ref(-dedup["temp"], -FREEZE_THRESHOLD_F, dt), 2
        )

    # Shoulder freeze hours — NaN-safe sum
    if "freeze_hours_shoulder" in summary.columns:
        row["freeze_hours_shoulder_sum"] = round(
            float(summary["freeze_hours_shoulder"].dropna().sum()), 2
        )

    # Total hours with temp — sum (denominator for window-level freeze pct)
    if "total_hours_with_temp" in summary.columns:
        row["total_hours_with_temp_sum"] = round(
            float(summary["total_hours_with_temp"].dropna().sum()), 2
        )

    # Window-level freeze pct = freeze_hours_sum / total_hours_with_temp_sum
    frz_sum = row.get("freeze_hours_sum", float("nan"))
    tht_sum = row.get("total_hours_with_temp_sum", 0.0)
    if not np.isnan(frz_sum) and tht_sum > 0:
        row["freeze_hours_pct_over_window"] = round(float(frz_sum) / float(tht_sum), 6)
    else:
        row["freeze_hours_pct_over_window"] = float("nan")

    # Freeze event count — sum across years
    if "freeze_event_count" in summary.columns:
        row["freeze_event_count_sum"] = int(summary["freeze_event_count"].fillna(0).sum())

    # Freeze event max duration — max of yearly maxima
    if "freeze_event_max_duration_hours" in summary.columns:
        valid_ev = summary["freeze_event_max_duration_hours"].dropna()
        row["freeze_event_max_duration_hours_max"] = (
            float(valid_ev.max()) if len(valid_ev) > 0 else float("nan")
        )

    # Freeze confidence flag — uses actual window hours from summary date range
    frz_mpa = row.get("timestamp_missing_pct_avg", float("nan"))
    frz_tht = row.get("total_hours_with_temp_sum", float("nan"))
    low_temp_avail = False
    if (
        "window_start" in summary.columns
        and "window_end" in summary.columns
        and len(summary) > 0
    ):
        win_start = pd.Timestamp(str(summary["window_start"].min()))
        win_end = pd.Timestamp(str(summary["window_end"].max())) + pd.Timedelta(days=1)
        window_hours = (win_end - win_start).total_seconds() / 3600.0
        if window_hours > 0 and not np.isnan(frz_tht):
            low_temp_avail = (frz_tht / window_hours) < FREEZE_CONFIDENCE_TEMP_AVAIL_MIN_FRAC
    row["freeze_confidence_flag"] = bool(
        (not np.isnan(frz_mpa) and frz_mpa > MISSING_DATA_WARNING_THRESHOLD)
        or low_temp_avail
    )

    # Wet-bulb extremes
    if "wb_p99" in summary.columns and summary["wb_p99"].notna().any():
        row["wb_p99_median"] = float(np.nanmedian(summary["wb_p99"].dropna().astype(float)))
    if "wb_p996" in summary.columns and summary["wb_p996"].notna().any():
        row["wb_p996_median"] = float(np.nanmedian(summary["wb_p996"].dropna().astype(float)))

    # ── Economizer / tower aggregation ────────────────────────────────────────

    # Airside econ — sum across years
    if "air_econ_hours" in summary.columns:
        row["air_econ_hours_sum"] = round(float(summary["air_econ_hours"].sum()), 2)

    # WEC hours — NaN-safe sum (NaN if all years had no wetbulb)
    if "wec_hours" in summary.columns:
        valid_wec = summary["wec_hours"].dropna()
        row["wec_hours_sum"] = round(float(valid_wec.sum()), 2) if len(valid_wec) > 0 else float("nan")

    # Hours with wetbulb — sum (0-safe; 0.0 means no wetbulb data at all)
    if "hours_with_wetbulb" in summary.columns:
        hwb_vals = summary["hours_with_wetbulb"].fillna(0.0)
        row["hours_with_wetbulb_sum"] = round(float(hwb_vals.sum()), 2)

    # WEC feasibility over full window = wec_hours_sum / hours_with_wetbulb_sum
    wec_sum = row.get("wec_hours_sum", float("nan"))
    hwb_sum = row.get("hours_with_wetbulb_sum", 0.0)
    if not np.isnan(wec_sum) and hwb_sum > 0:
        row["wec_feasible_pct_over_window"] = round(float(wec_sum) / float(hwb_sum), 6)
    else:
        row["wec_feasible_pct_over_window"] = float("nan")

    # Tower stress hours per threshold — sum across years
    stress_cols = [c for c in summary.columns if c.startswith("tower_stress_hours_wb_gt_")]
    for sc in stress_cols:
        valid_sc = summary[sc].dropna()
        row[f"{sc}_sum"] = round(float(valid_sc.sum()), 2) if len(valid_sc) > 0 else float("nan")

    # Rolling wb maxima — max of yearly maxima
    for max_col in ("wb_mean_24h_max", "wb_mean_72h_max"):
        if max_col in summary.columns:
            valid_v = summary[max_col].dropna()
            row[f"{max_col}_max"] = float(valid_v.max()) if len(valid_v) > 0 else float("nan")

    # LWT proxy — median across years
    if "lwt_proxy_p99" in summary.columns:
        valid_lwt = summary["lwt_proxy_p99"].dropna()
        row["lwt_proxy_p99_median"] = (
            float(np.nanmedian(valid_lwt)) if len(valid_lwt) > 0 else float("nan")
        )

    # Econ confidence flag — fires when any of:
    #   1. timestamp_missing_pct_avg > MISSING_DATA_WARNING_THRESHOLD (2%)
    #   2. wetbulb_availability_pct_avg < WETBULB_AVAIL_WARNING_THRESHOLD (70%)
    #   3. hours_with_wetbulb_sum / max_possible_hours < ECON_WB_COVERAGE_MIN_FRAC (90%)
    mpa_econ = row.get("timestamp_missing_pct_avg", float("nan"))
    wba_econ = row.get("wetbulb_availability_pct_avg", float("nan"))
    cov_econ = row.get("coverage_weighted_pct", float("nan"))
    hwb_econ = row.get("hours_with_wetbulb_sum", float("nan"))
    # Compute max possible hours from actual window dates (leap-year safe),
    # mirroring the freeze confidence block above.
    low_wb_coverage = False
    if (
        "window_start" in summary.columns
        and "window_end" in summary.columns
        and len(summary) > 0
    ):
        econ_win_start = pd.Timestamp(str(summary["window_start"].min()))
        econ_win_end = pd.Timestamp(str(summary["window_end"].max())) + pd.Timedelta(days=1)
        max_possible_hours = (econ_win_end - econ_win_start).total_seconds() / 3600.0
        if max_possible_hours > 0 and not np.isnan(cov_econ) and not np.isnan(hwb_econ):
            adjusted_hours = max_possible_hours * cov_econ
            low_wb_coverage = (hwb_econ / adjusted_hours) < ECON_WB_COVERAGE_MIN_FRAC
    row["econ_confidence_flag"] = bool(
        (not np.isnan(mpa_econ) and mpa_econ > MISSING_DATA_WARNING_THRESHOLD)
        or (not np.isnan(wba_econ) and wba_econ < WETBULB_AVAIL_WARNING_THRESHOLD)
        or low_wb_coverage
    )

    # Wind summary aggregation
    if "wind_speed_mean_kt" in summary.columns and summary["wind_speed_mean_kt"].notna().any():
        row["wind_speed_mean_kt"] = round(float(summary["wind_speed_mean_kt"].mean()), 2)
    if "wind_speed_max_kt" in summary.columns and summary["wind_speed_max_kt"].notna().any():
        row["wind_speed_max_kt"] = round(float(summary["wind_speed_max_kt"].max()), 2)
    if "wind_calm_pct" in summary.columns and summary["wind_calm_pct"].notna().any():
        row["wind_calm_pct"] = round(float(summary["wind_calm_pct"].mean()), 2)

    # Quality warning flag
    mpa = row.get("timestamp_missing_pct_avg", float("nan"))
    wba = row.get("wetbulb_availability_pct_avg", float("nan"))
    row["missing_data_warning"] = bool(
        (not np.isnan(mpa) and mpa > MISSING_DATA_WARNING_THRESHOLD)
        or (not np.isnan(wba) and wba < WETBULB_AVAIL_WARNING_THRESHOLD)
    )

    return row


def build_compare_summary(
    station_results: list[StationResult],
    ref_temps: list[float],
) -> pd.DataFrame:
    """Aggregate all stations and compute scores.

    Parameters
    ----------
    station_results : list of StationResult from run_station_pipeline
    ref_temps : list of reference temperatures for hours_above_ref columns

    Returns
    -------
    pd.DataFrame — one row per station with all window-aggregated metrics and scores.
    """
    rows = []
    for r in station_results:
        row = aggregate_station_window(
            summary=r.summary,
            windowed=r.windowed,
            interval_info=r.interval_info,
            ref_temps=ref_temps,
            station_id=r.cfg.station_id or "csv",
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    df = compute_scores(df, ref_temps)
    return df
