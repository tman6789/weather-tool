"""Deterministic insights: rankings, trends, and markdown report generation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from weather_tool.config import RunConfig


def top_n(summary: pd.DataFrame, col: str, n: int = 3, ascending: bool = False) -> pd.DataFrame:
    """Return top-N rows by *col*."""
    return summary.nlargest(n, col) if not ascending else summary.nsmallest(n, col)


def trend_hours_above(summary: pd.DataFrame) -> dict[str, Any]:
    """Simple linear regression of hours_above_ref vs year.

    Returns dict with slope, intercept, r_value, p_value, direction.
    """
    valid = summary.dropna(subset=["hours_above_ref"])
    if len(valid) < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r_value": float("nan"),
            "p_value": float("nan"),
            "direction": "insufficient data",
        }
    x = valid["year"].astype(float).values
    y = valid["hours_above_ref"].astype(float).values
    result = stats.linregress(x, y)
    direction = "increasing" if result.slope > 0 else ("decreasing" if result.slope < 0 else "flat")
    return {
        "slope": round(float(result.slope), 4),
        "intercept": round(float(result.intercept), 2),
        "r_value": round(float(result.rvalue), 4),
        "p_value": round(float(result.pvalue), 6),
        "direction": direction,
    }


def trend_wb_p99(summary: pd.DataFrame) -> dict[str, Any]:
    """Simple linear regression of wb_p99 vs year.

    Returns dict with slope, intercept, r_value, p_value, direction.
    """
    if "wb_p99" not in summary.columns:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r_value": float("nan"),
            "p_value": float("nan"),
            "direction": "no data",
        }
    valid = summary.dropna(subset=["wb_p99"])
    if len(valid) < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r_value": float("nan"),
            "p_value": float("nan"),
            "direction": "insufficient data",
        }
    x = valid["year"].astype(float).values
    y = valid["wb_p99"].astype(float).values
    result = stats.linregress(x, y)
    direction = "increasing" if result.slope > 0 else ("decreasing" if result.slope < 0 else "flat")
    return {
        "slope": round(float(result.slope), 4),
        "intercept": round(float(result.intercept), 2),
        "r_value": round(float(result.rvalue), 4),
        "p_value": round(float(result.pvalue), 6),
        "direction": direction,
    }


def _fmt_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Format a DataFrame as a markdown table (up to max_rows)."""
    subset = df.head(max_rows)
    return subset.to_markdown(index=False)


def generate_insights_md(
    summary: pd.DataFrame,
    cfg: RunConfig,
    interval_info: dict[str, Any],
) -> str:
    """Build the full insights markdown report (deterministic only)."""
    lines: list[str] = []
    lines.append(f"# Weather Insights Report")
    lines.append(f"")
    lines.append(f"**Station:** {cfg.station_id or 'CSV'}  ")
    lines.append(f"**Window:** {cfg.start} to {cfg.end}  ")
    lines.append(f"**Reference temp:** {cfg.ref_temp}  ")
    lines.append(f"**Units:** {cfg.units}  ")
    lines.append(f"**Inferred dt:** {interval_info.get('dt_minutes', '?')} minutes  ")
    lines.append(f"")

    # ── Top 3 hottest years ──
    lines.append("## Top 3 Hottest Years (by Tmax)")
    lines.append("")
    hot = top_n(summary, "tmax", 3)
    if not hot.empty:
        for _, r in hot.iterrows():
            lines.append(f"- **{int(r['year'])}**: Tmax = {r['tmax']}")
    else:
        lines.append("_No data._")
    lines.append("")

    # ── Top 3 years by hours_above_ref ──
    lines.append(f"## Top 3 Years by Hours Above {cfg.ref_temp}")
    lines.append("")
    hab = top_n(summary, "hours_above_ref", 3)
    if not hab.empty:
        for _, r in hab.iterrows():
            lines.append(f"- **{int(r['year'])}**: {r['hours_above_ref']} hours")
    else:
        lines.append("_No data._")
    lines.append("")

    # ── Trend ──
    trend = trend_hours_above(summary)
    lines.append("## Hours-Above-Ref Trend")
    lines.append("")
    lines.append(f"- Direction: **{trend['direction']}**")
    lines.append(f"- Slope: {trend['slope']} hours/year")
    lines.append(f"- R-value: {trend['r_value']}")
    lines.append(f"- p-value: {trend['p_value']}")
    lines.append("")

    # ── Moisture / Tower Stress ──
    if "wb_p99" in summary.columns and summary["wb_p99"].notna().any():
        lines.append("## Moisture / Tower Stress")
        lines.append("")

        wb_top = top_n(summary, "wb_p99", 3)
        lines.append("**Top 3 years by wb_p99 (99th percentile wet-bulb, °F):**")
        for _, r in wb_top.iterrows():
            lines.append(f"- **{int(r['year'])}**: wb_p99 = {r['wb_p99']} °F")
        lines.append("")

        wb_trend = trend_wb_p99(summary)
        lines.append(f"**wb_p99 trend:** {wb_trend['direction']}  ")
        lines.append(f"- Slope: {wb_trend['slope']} °F/year")
        lines.append(f"- R-value: {wb_trend['r_value']}")
        lines.append(f"- p-value: {wb_trend['p_value']}")
        lines.append("")

        if "wetbulb_availability_pct" in summary.columns:
            low_wb = summary[summary["wetbulb_availability_pct"] < 80.0]
            if not low_wb.empty:
                lines.append("**Low wet-bulb availability years (<80%) — data quality warning:**")
                for _, r in low_wb.iterrows():
                    lines.append(
                        f"- {int(r['year'])}: {r['wetbulb_availability_pct']:.1f}% availability"
                    )
                lines.append("")

    # ── Economizer / Tower Decision ──
    lines.append("## Economizer / Tower Decision")
    lines.append("")
    lines.append(
        f"> *Proxy metrics only. Air econ threshold: {cfg.air_econ_threshold_f} °F. "
        f"WEC proxy: Twb ≤ {cfg.chw_supply_f - cfg.tower_approach_f - cfg.hx_approach_f:.1f} °F "
        f"(CHW supply {cfg.chw_supply_f} − tower approach {cfg.tower_approach_f} − "
        f"HX approach {cfg.hx_approach_f} °F). "
        f"LWT proxy = Twb + tower approach (not a full simulation).*"
    )
    lines.append("")

    if "air_econ_hours" in summary.columns:
        air_total = float(summary["air_econ_hours"].sum())
        top_ae = summary.nlargest(3, "air_econ_hours")
        lines.append(f"**Airside econ (Tdb ≤ {cfg.air_econ_threshold_f} °F):** {air_total:.0f} hrs total")
        for _, r in top_ae.iterrows():
            lines.append(f"- **{int(r['year'])}**: {r['air_econ_hours']:.0f} hrs")
        lines.append("")

    if "wec_hours" in summary.columns and summary["wec_hours"].notna().any():
        req_twb = float(summary["required_twb_max"].iloc[0])
        wec_total = float(summary["wec_hours"].dropna().sum())
        top_wec = summary.dropna(subset=["wec_hours"]).nlargest(3, "wec_hours")
        lines.append(f"**WEC proxy (Twb ≤ {req_twb:.1f} °F):** {wec_total:.0f} hrs total")
        for _, r in top_wec.iterrows():
            feas = r.get("wec_feasibility_pct")
            pct = f" ({feas*100:.1f}%)" if pd.notna(feas) else ""
            lines.append(f"- **{int(r['year'])}**: {r['wec_hours']:.0f} hrs{pct}")
        lines.append("")
    elif "wec_hours" in summary.columns:
        lines.append("**WEC proxy:** _wet-bulb data unavailable — metric not computed._")
        lines.append("")

    stress_cols = sorted([c for c in summary.columns if c.startswith("tower_stress_hours_wb_gt_")])
    if stress_cols and summary[stress_cols[0]].notna().any():
        lines.append("**Tower stress (Twb ≥ threshold):**")
        for sc in stress_cols:
            thr = sc.replace("tower_stress_hours_wb_gt_", "")
            total = float(summary[sc].dropna().sum())
            lines.append(f"- Twb ≥ {thr} °F: {total:.0f} hrs total")
        lines.append("")

    if "wb_mean_72h_max" in summary.columns and summary["wb_mean_72h_max"].notna().any():
        max72 = float(summary["wb_mean_72h_max"].dropna().max())
        worst_yr = int(summary.loc[summary["wb_mean_72h_max"].idxmax(), "year"])
        lines.append(f"**Worst 72h sustained Twb:** {max72:.1f} °F (year {worst_yr})")
        lines.append("")

    if "lwt_proxy_p99" in summary.columns and summary["lwt_proxy_p99"].notna().any():
        lwt_p99 = float(summary["lwt_proxy_p99"].dropna().max())
        lwt_max_val = (
            float(summary["lwt_proxy_max"].dropna().max())
            if "lwt_proxy_max" in summary.columns and summary["lwt_proxy_max"].notna().any()
            else float("nan")
        )
        lwt_max_str = f"{lwt_max_val:.1f} °F" if not pd.isna(lwt_max_val) else "N/A"
        lines.append(
            f"**LWT proxy (Twb + approach):** p99 = {lwt_p99:.1f} °F, max = {lwt_max_str}"
            f"  *(proxy — actual LWT depends on tower specs)*"
        )
        lines.append("")

    # ── Data quality notes ──
    lines.append("## Data Quality Notes")
    lines.append("")

    high_missing = summary[summary["missing_pct"] > 0.05]
    if not high_missing.empty:
        lines.append("**Years with >5% missing data:**")
        for _, r in high_missing.iterrows():
            lines.append(f"- {int(r['year'])}: {r['missing_pct']*100:.1f}% missing")
        lines.append("")

    dup_years = summary[summary["duplicate_count"] > 0]
    if not dup_years.empty:
        lines.append("**Years with duplicate timestamps:**")
        for _, r in dup_years.iterrows():
            lines.append(f"- {int(r['year'])}: {int(r['duplicate_count'])} duplicates")
        lines.append("")

    if interval_info.get("interval_change_flag"):
        lines.append("**Interval change detected:** sampling interval is not uniform across the dataset.")
        lines.append("")

    nan_years = summary[summary["nan_temp_count"] > 0]
    if not nan_years.empty:
        lines.append("**Years with NaN temperatures:**")
        for _, r in nan_years.iterrows():
            lines.append(f"- {int(r['year'])}: {int(r['nan_temp_count'])} NaN readings")
        lines.append("")

    partial = summary[summary["partial_coverage_flag"]]
    if not partial.empty:
        lines.append("**Partial-coverage years:**")
        for _, r in partial.iterrows():
            lines.append(
                f"- {int(r['year'])}: {r['coverage_pct']*100:.1f}% of year covered "
                f"({r['window_start']} to {r['window_end']})"
            )
        lines.append("")

    # ── Summary table preview ──
    lines.append("## Summary Table (up to 10 rows)")
    lines.append("")
    lines.append(_fmt_table(summary))
    lines.append("")

    return "\n".join(lines)
