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
