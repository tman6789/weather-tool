"""Decision AI packet builder.

Assembles authoritative JSON packets from already-computed StationResult data.
All logic is pure (no file I/O).

Inputs contract
---------------
Required summary columns:      t_p99, t_p996, tmax, tmin (via _safe_* helpers)
Optional summary columns:      wb_p99, wb_p996, wb_max, wb_mean,
                                wetbulb_availability_pct,
                                air_econ_hours, wec_hours, hours_with_wetbulb,
                                tower_stress_hours_wb_gt_75/78/80,
                                tdb_mean_24h_max, tdb_mean_72h_max,
                                wb_mean_24h_max, wb_mean_72h_max,
                                lwt_proxy_p99, freeze_hours, freeze_hours_shoulder,
                                freeze_event_count, freeze_event_max_duration_hours,
                                missing_pct

Required windowed df columns:  timestamp, temp, _is_dup
Optional windowed df columns:  wetbulb_f, relh, wind_speed_kt, is_calm
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import pandas as pd

from weather_tool.config import (
    MISSING_DATA_WARNING_THRESHOLD,
    WETBULB_AVAIL_WARNING_THRESHOLD,
)
from weather_tool.insights.death_day import find_death_day_candidates
from weather_tool.insights.rules import PROFILES, Recommendation, evaluate_station_flags

if TYPE_CHECKING:
    from weather_tool.config import RunConfig
    from weather_tool.pipeline import StationResult


def _safe_median(summary: pd.DataFrame, col: str) -> float | None:
    if col in summary.columns and summary[col].notna().any():
        return round(float(summary[col].dropna().median()), 2)
    return None


def _safe_max(summary: pd.DataFrame, col: str) -> float | None:
    if col in summary.columns and summary[col].notna().any():
        return round(float(summary[col].dropna().max()), 2)
    return None


def _safe_min(summary: pd.DataFrame, col: str) -> float | None:
    if col in summary.columns and summary[col].notna().any():
        return round(float(summary[col].dropna().min()), 2)
    return None


def _safe_sum(summary: pd.DataFrame, col: str) -> float | None:
    if col in summary.columns and summary[col].notna().any():
        return round(float(summary[col].dropna().sum()), 2)
    return None


def build_station_packet(
    cfg: RunConfig,
    result: StationResult,
    profile_name: str,
    window_hours: int,
    top_n: int,
) -> dict[str, Any]:
    """Build an authoritative station decision packet.

    Parameters
    ----------
    cfg : RunConfig for this station.
    result : StationResult with summary and windowed DataFrames already computed.
    profile_name : one of the PROFILES keys.
    window_hours : rolling window length for Death Day search.
    top_n : number of Death Day candidates to find.

    Returns
    -------
    dict with keys: meta, quality, design_conditions, operational_efficiency,
    freeze_risk, death_day, risk_flags, recommendations.
    """
    summary = result.summary

    # ── Design conditions ──────────────────────────────────────────────────────
    design_conditions: dict[str, Any] = {
        "tdb_p99":          _safe_median(summary, "t_p99"),
        "tdb_p996":         _safe_median(summary, "t_p996"),
        "tdb_max":          _safe_max(summary, "tmax"),
        "wb_p99":           _safe_median(summary, "wb_p99"),
        "wb_p996":          _safe_median(summary, "wb_p996"),
        "wb_max":           _safe_max(summary, "wb_max"),
        "wb_mean":          _safe_median(summary, "wb_mean"),
        "tdb_mean_24h_max": _safe_max(summary, "tdb_mean_24h_max"),
        "tdb_mean_72h_max": _safe_max(summary, "tdb_mean_72h_max"),
        "wb_mean_24h_max":  _safe_max(summary, "wb_mean_24h_max"),
        "wb_mean_72h_max":  _safe_max(summary, "wb_mean_72h_max"),
        "lwt_proxy_p99":    _safe_median(summary, "lwt_proxy_p99"),
        # Provenance
        "baseline_years_count": int(len(summary)),
        "baseline_method":      "median_over_years",
    }

    # ── Operational efficiency ─────────────────────────────────────────────────
    wec_sum = _safe_sum(summary, "wec_hours")
    hwb_sum = _safe_sum(summary, "hours_with_wetbulb")
    if wec_sum is not None and hwb_sum is not None and hwb_sum > 0:
        wec_feasible_pct: float | None = round(wec_sum / hwb_sum, 4)
    else:
        wec_feasible_pct = None

    operational_efficiency: dict[str, Any] = {
        "air_econ_hours_sum":              _safe_sum(summary, "air_econ_hours"),
        "wec_hours_sum":                   wec_sum,
        "hours_with_wetbulb_sum":          hwb_sum,
        "wec_feasible_pct_over_window":    wec_feasible_pct,
        "tower_stress_hours_wb_gt_75_sum": _safe_sum(summary, "tower_stress_hours_wb_gt_75"),
        "tower_stress_hours_wb_gt_78_sum": _safe_sum(summary, "tower_stress_hours_wb_gt_78"),
        "tower_stress_hours_wb_gt_80_sum": _safe_sum(summary, "tower_stress_hours_wb_gt_80"),
    }

    # ── Freeze risk ────────────────────────────────────────────────────────────
    freeze_event_count: int = 0
    if "freeze_event_count" in summary.columns:
        freeze_event_count = int(summary["freeze_event_count"].fillna(0).sum())

    freeze_risk: dict[str, Any] = {
        "freeze_hours_sum":                    _safe_sum(summary, "freeze_hours"),
        "freeze_hours_shoulder_sum":           _safe_sum(summary, "freeze_hours_shoulder"),
        "freeze_event_count_sum":              freeze_event_count,
        "freeze_event_max_duration_hours_max": _safe_max(summary, "freeze_event_max_duration_hours"),
        "tmin_min":                            _safe_min(summary, "tmin"),
    }

    # ── Quality flags (computed inline) ───────────────────────────────────────
    mpa: float = 0.0
    if "missing_pct" in summary.columns and summary["missing_pct"].notna().any():
        mpa = float(summary["missing_pct"].dropna().mean())

    wba: float = 100.0
    if "wetbulb_availability_pct" in summary.columns and summary["wetbulb_availability_pct"].notna().any():
        wba = float(summary["wetbulb_availability_pct"].dropna().mean())

    missing_data_warning = (mpa > MISSING_DATA_WARNING_THRESHOLD) or (wba < WETBULB_AVAIL_WARNING_THRESHOLD)

    quality: dict[str, Any] = {
        "missing_data_warning":      missing_data_warning,
        "wetbulb_availability_pct":  round(wba, 2),
        "missing_pct_avg":           round(mpa, 4),
        "interval_change_flag":      bool(result.interval_info.get("interval_change_flag", False)),
    }

    # ── Death Day candidates ───────────────────────────────────────────────────
    dedup = result.windowed.loc[~result.windowed["_is_dup"]].copy()

    tdb_p99_val  = design_conditions.get("tdb_p99")  or float("nan")
    tdb_p996_val = design_conditions.get("tdb_p996") or float("nan")
    wb_p99_val   = design_conditions.get("wb_p99")
    wb_p996_val  = design_conditions.get("wb_p996")

    candidates = find_death_day_candidates(
        df=dedup,
        dt_minutes=result.interval_info["dt_minutes"],
        window_hours=window_hours,
        tdb_p99=tdb_p99_val,
        tdb_p996=tdb_p996_val,
        wb_p99=wb_p99_val,
        wb_p996=wb_p996_val,
        top_n=top_n,
    )

    death_day_mode = candidates[0]["mode"] if candidates else "heat_day"
    death_day_block: dict[str, Any] = {
        "mode":         death_day_mode,
        "window_hours": window_hours,
        "candidates":   candidates,
    }

    # ── Risk flags + recommendations ───────────────────────────────────────────
    profile = PROFILES[profile_name]
    rules_packet = {
        "design_conditions":      design_conditions,
        "operational_efficiency": operational_efficiency,
        "freeze_risk":            freeze_risk,
        "quality":                quality,
    }
    flags, rec_objects = evaluate_station_flags(rules_packet, profile)

    # Serialize Recommendation dataclasses to dicts for JSON portability
    recommendations = [asdict(r) for r in rec_objects]

    # ── Assemble packet ────────────────────────────────────────────────────────
    return {
        "meta": {
            "station_id":    cfg.station_id or "csv",
            "window_start":  str(cfg.start),
            "window_end":    str(cfg.end),
            "profile":       profile_name,
            "dt_minutes":    result.interval_info["dt_minutes"],
            "years_covered": int(len(summary)),
            "ref_temp":      cfg.ref_temp,
        },
        "quality":                quality,
        "design_conditions":      design_conditions,
        "operational_efficiency": operational_efficiency,
        "freeze_risk":            freeze_risk,
        "death_day":              death_day_block,
        "risk_flags":             flags,
        "recommendations":        recommendations,
    }


# ── Compare packet ─────────────────────────────────────────────────────────────

def _extract_ranking(
    compare_df: pd.DataFrame,
    col: str,
    stations: list[str],
) -> list[dict[str, Any]]:
    """Return a list of {station_id, value, rank} sorted descending by col."""
    if col not in compare_df.columns:
        return []
    rows = compare_df[["station_id", col]].dropna(subset=[col])
    rows = rows.sort_values(col, ascending=False).reset_index(drop=True)
    return [
        {"station_id": str(row["station_id"]), "value": row[col], "rank": i + 1}
        for i, (_, row) in enumerate(rows.iterrows())
    ]


def _station_with_extreme(
    compare_df: pd.DataFrame,
    col: str,
    highest: bool = True,
) -> dict[str, Any] | None:
    if col not in compare_df.columns or compare_df[col].isna().all():
        return None
    idx = compare_df[col].idxmax() if highest else compare_df[col].idxmin()
    row = compare_df.loc[idx]
    return {"station_id": str(row["station_id"]), "value": row[col]}


def _aggregate_flags(station_packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect all high-severity flags across all station packets, deduplicated by flag_id."""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for pkt in station_packets:
        sid = pkt.get("meta", {}).get("station_id", "?")
        for f in pkt.get("risk_flags", []):
            if f["severity"] == "high":
                key = f"{f['flag_id']}_{sid}"
                if key not in seen:
                    seen.add(key)
                    result.append({**f, "station_id": sid})
    return result


def build_compare_packet(
    cfg_start: str,
    cfg_end: str,
    compare_df: pd.DataFrame,
    station_packets: list[dict[str, Any]],
    stations: list[str],
    full: bool = False,
) -> dict[str, Any]:
    """Build a compare-level decision packet.

    Parameters
    ----------
    cfg_start, cfg_end : analysis window start/end strings.
    compare_df : DataFrame from build_compare_summary (one row per station).
    station_packets : list of station packets from build_station_packet.
    stations : ordered list of station IDs.
    full : if True, embed full per-station packets; otherwise use lean summaries.

    Returns
    -------
    dict with keys: meta, station_summaries, station_packets (if full),
    rankings, cross_station_extremes, aggregated_flags.
    """
    # Lean per-station summaries (always included)
    station_summaries = [
        {
            "station_id":       p["meta"]["station_id"],
            "tdb_p996":         p["design_conditions"].get("tdb_p996"),
            "wb_p996":          p["design_conditions"].get("wb_p996"),
            "air_econ_hours_sum": p["operational_efficiency"].get("air_econ_hours_sum"),
            "freeze_hours_sum": p["freeze_risk"].get("freeze_hours_sum"),
            "death_day_rank1":  p["death_day"]["candidates"][0] if p["death_day"]["candidates"] else None,
            "risk_flags":       p["risk_flags"],
        }
        for p in station_packets
    ]

    score_cols = ["overall_score", "heat_score", "moisture_score", "freeze_score", "data_quality_score"]
    rankings = {col: _extract_ranking(compare_df, col, stations) for col in score_cols}

    cross_station_extremes: dict[str, Any] = {
        "hottest_station":   _station_with_extreme(compare_df, "tmax_max", highest=True),
        "coldest_station":   _station_with_extreme(compare_df, "tmin_min", highest=False),
        "highest_wb_p996":   _station_with_extreme(compare_df, "wb_p996_median", highest=True),
        "most_econ_hours":   _station_with_extreme(compare_df, "air_econ_hours_sum", highest=True),
        "most_freeze_hours": _station_with_extreme(compare_df, "freeze_hours_sum", highest=True),
    }

    packet: dict[str, Any] = {
        "meta": {
            "window_start":   cfg_start,
            "window_end":     cfg_end,
            "stations":       stations,
            "station_count":  len(stations),
        },
        "station_summaries":     station_summaries,
        "station_packets":       {p["meta"]["station_id"]: p for p in station_packets} if full else None,
        "rankings":              rankings,
        "cross_station_extremes": cross_station_extremes,
        "aggregated_flags":      _aggregate_flags(station_packets),
    }
    return packet
