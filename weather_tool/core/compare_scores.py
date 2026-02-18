"""Scoring logic for multi-station climate comparison."""

from __future__ import annotations

import pandas as pd

from weather_tool.config import (
    FREEZE_SCORE_WEIGHTS,
    HEAT_SCORE_WEIGHTS,
    SCORE_WEIGHTS,
)


def _minmax_norm(series: pd.Series, invert: bool = False) -> pd.Series:
    """Min-max normalize *series* to [0, 100]. Returns 50 for all-equal series."""
    lo = float(series.min())
    hi = float(series.max())
    if hi == lo:
        return pd.Series(50.0, index=series.index, dtype=float)
    result = (series - lo) / (hi - lo) * 100.0
    return (100.0 - result) if invert else result


def compute_scores(df: pd.DataFrame, ref_temps: list[float]) -> pd.DataFrame:
    """Add heat/moisture/freeze/quality/overall score columns to a copy of *df*.

    Scores are min-max normalized across stations to [0, 100].
    Higher = more of that characteristic (more heat, more moisture, more freeze risk, better quality).
    """
    out = df.copy()

    # Primary ref temp = max threshold (highest threshold captures hottest stations)
    primary_rt = int(max(ref_temps))
    primary_col = f"hours_above_ref_{primary_rt}_sum"

    # ── heat_score ────────────────────────────────────────────────────────────
    h_hours_raw = out[primary_col].fillna(0.0) if primary_col in out.columns else pd.Series(0.0, index=out.index)
    h_t99_raw = out["t_p99_median"].fillna(out["t_p99_median"].mean()) if "t_p99_median" in out.columns else pd.Series(0.0, index=out.index)
    h_hours = _minmax_norm(h_hours_raw)
    h_t99 = _minmax_norm(h_t99_raw)
    out["heat_score"] = (
        HEAT_SCORE_WEIGHTS["hours_above_ref"] * h_hours
        + HEAT_SCORE_WEIGHTS["t_p99"] * h_t99
    ).round(1)

    # ── moisture_score ────────────────────────────────────────────────────────
    if "wb_p99_median" in out.columns and out["wb_p99_median"].notna().any():
        wb_filled = out["wb_p99_median"].fillna(float(out["wb_p99_median"].mean()))
        out["moisture_score"] = _minmax_norm(wb_filled).round(1)
    else:
        out["moisture_score"] = float("nan")

    # ── freeze_score ──────────────────────────────────────────────────────────
    f_hrs_raw = out["freeze_hours_sum"].fillna(0.0) if "freeze_hours_sum" in out.columns else pd.Series(0.0, index=out.index)
    f_tmin_raw = out["tmin_min"].fillna(float(out["tmin_min"].mean())) if "tmin_min" in out.columns else pd.Series(0.0, index=out.index)
    f_hrs = _minmax_norm(f_hrs_raw)
    f_tmin = _minmax_norm(f_tmin_raw, invert=True)  # lower tmin = more freeze risk = higher score
    out["freeze_score"] = (
        FREEZE_SCORE_WEIGHTS["freeze_hours"] * f_hrs
        + FREEZE_SCORE_WEIGHTS["tmin_min"] * f_tmin
    ).round(1)

    # ── data_quality_score ────────────────────────────────────────────────────
    q_missing_raw = out["timestamp_missing_pct_avg"].fillna(0.0) if "timestamp_missing_pct_avg" in out.columns else pd.Series(0.0, index=out.index)
    q_cov_raw = out["coverage_weighted_pct"].fillna(1.0) if "coverage_weighted_pct" in out.columns else pd.Series(1.0, index=out.index)
    q_icf_raw = out["interval_change_flag_any"].astype(float) if "interval_change_flag_any" in out.columns else pd.Series(0.0, index=out.index)

    q_missing = _minmax_norm(q_missing_raw, invert=True)
    q_cov = _minmax_norm(q_cov_raw)
    q_icf = _minmax_norm(q_icf_raw, invert=True)

    if "wetbulb_availability_pct_avg" in out.columns:
        q_wb = _minmax_norm(out["wetbulb_availability_pct_avg"].fillna(50.0))
        out["data_quality_score"] = (
            0.40 * q_missing + 0.20 * q_cov + 0.30 * q_wb + 0.10 * q_icf
        ).round(1)
    else:
        # No wet-bulb availability data — use neutral 50 for that component
        out["data_quality_score"] = (
            0.40 * q_missing + 0.20 * q_cov + pd.Series(50.0 * 0.30, index=out.index) + 0.10 * q_icf
        ).round(1)

    # ── overall_score ─────────────────────────────────────────────────────────
    w = SCORE_WEIGHTS
    if out["moisture_score"].notna().all():
        out["overall_score"] = (
            w["heat"] * out["heat_score"]
            + w["moisture"] * out["moisture_score"]
            + w["freeze"] * out["freeze_score"]
            + w["quality"] * out["data_quality_score"]
        ).round(1)
    else:
        # Redistribute moisture weight: heat 0.50, freeze 0.35, quality 0.15
        out["overall_score"] = (
            0.50 * out["heat_score"]
            + 0.35 * out["freeze_score"]
            + 0.15 * out["data_quality_score"]
        ).round(1)

    # Stations with no data (years_covered_count == 0) should not receive scores.
    if "years_covered_count" in out.columns:
        no_data = out["years_covered_count"] == 0
        score_cols = ["heat_score", "moisture_score", "freeze_score",
                      "data_quality_score", "overall_score"]
        for sc in score_cols:
            if sc in out.columns:
                out.loc[no_data, sc] = float("nan")

    return out
