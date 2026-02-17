"""Deterministic multi-station comparison markdown report."""

from __future__ import annotations

from datetime import date

import pandas as pd

from weather_tool.config import MISSING_DATA_WARNING_THRESHOLD, WETBULB_AVAIL_WARNING_THRESHOLD


def generate_compare_report_md(
    compare_df: pd.DataFrame,
    stations: list[str],
    window_start: date,
    window_end: date,
    fields: list[str],
    ref_temps: list[float],
) -> str:
    """Generate a deterministic markdown comparison report.

    All statements are derived from *compare_df* — no speculation.
    """
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        f"# Climate Comparison: {', '.join(stations)}",
        "",
        f"**Window:** {window_start} – {window_end}  ",
        f"**Fields:** {', '.join(fields)}  ",
        f"**Reference temps:** {', '.join(str(int(t) if t == int(t) else t) for t in ref_temps)} °F  ",
        "",
    ]

    if len(compare_df) < 3:
        lines += ["*Note: Rankings are weak with fewer than 3 stations.*", ""]

    # ── Data quality summary ──────────────────────────────────────────────────
    lines += ["## Data Quality Summary", ""]
    for _, row in compare_df.iterrows():
        warn_parts: list[str] = []
        mpa = row.get("timestamp_missing_pct_avg")
        if mpa is not None and not pd.isna(mpa) and float(mpa) > MISSING_DATA_WARNING_THRESHOLD:
            warn_parts.append(f"timestamp missing {float(mpa) * 100:.1f}%")
        if row.get("interval_change_flag_any", False):
            warn_parts.append("interval change detected")
        wba = row.get("wetbulb_availability_pct_avg")
        if wba is not None and not pd.isna(wba) and float(wba) < WETBULB_AVAIL_WARNING_THRESHOLD:
            warn_parts.append(f"wet-bulb availability {float(wba):.0f}%")
        status = " | ".join(warn_parts) if warn_parts else "OK"
        lines.append(f"- **{row['station_id']}**: {status}")
    lines.append("")

    # ── Ranking tables ────────────────────────────────────────────────────────
    score_specs = [
        ("overall_score", "Overall Score"),
        ("heat_score", "Heat Score (higher = more heat exposure)"),
        ("moisture_score", "Moisture / Tower Stress Score (higher = more stress)"),
        ("freeze_score", "Freeze Risk Score (higher = more freeze risk)"),
        ("data_quality_score", "Data Quality Score (higher = better quality)"),
    ]
    for score_col, label in score_specs:
        if score_col not in compare_df.columns:
            continue
        if compare_df[score_col].isna().all():
            continue
        lines += [f"## Rankings: {label}", ""]
        ranked = (
            compare_df[["station_id", score_col]]
            .sort_values(score_col, ascending=False)
            .reset_index(drop=True)
        )
        for i, (_, r) in enumerate(ranked.iterrows(), 1):
            val = r[score_col]
            val_str = f"{float(val):.1f}" if not pd.isna(val) else "N/A"
            lines.append(f"{i}. **{r['station_id']}** — {val_str}")
        lines.append("")

    # ── Key differences ───────────────────────────────────────────────────────
    lines += ["## Key Differences", ""]

    # Most heat exposure (prefer 85°F threshold, else use highest available)
    hrs_cols = sorted(
        [c for c in compare_df.columns if c.startswith("hours_above_ref_") and c.endswith("_sum")]
    )
    heat_col = next((c for c in hrs_cols if "_85_" in c), hrs_cols[-1] if hrs_cols else None)
    if heat_col and compare_df[heat_col].notna().any():
        top_idx = compare_df[heat_col].idxmax()
        bot_idx = compare_df[heat_col].idxmin()
        top = compare_df.loc[top_idx]
        bot = compare_df.loc[bot_idx]
        if top["station_id"] != bot["station_id"] and float(bot[heat_col]) > 0:
            ratio = float(top[heat_col]) / float(bot[heat_col])
            threshold = heat_col.split("_")[3]
            lines.append(
                f"- **{top['station_id']}** has {ratio:.1f}× hours >{threshold}°F"
                f" vs **{bot['station_id']}**"
                f" ({int(top[heat_col])} vs {int(bot[heat_col])} hrs)"
            )

    # Highest wet-bulb stress
    if "wb_p99_median" in compare_df.columns and compare_df["wb_p99_median"].notna().any():
        wb_top_idx = compare_df["wb_p99_median"].idxmax()
        wb_top = compare_df.loc[wb_top_idx]
        lines.append(
            f"- **{wb_top['station_id']}** has highest wet-bulb p99"
            f" ({float(wb_top['wb_p99_median']):.1f}°F) → greatest cooling-tower stress"
        )

    # Highest freeze exposure
    if "freeze_hours_sum" in compare_df.columns and compare_df["freeze_hours_sum"].notna().any():
        frz_top_idx = compare_df["freeze_hours_sum"].idxmax()
        frz_top = compare_df.loc[frz_top_idx]
        lines.append(
            f"- **{frz_top['station_id']}** has highest freeze exposure"
            f" ({float(frz_top['freeze_hours_sum']):.0f} hrs < 32°F)"
        )

    # Coldest tmin
    if "tmin_min" in compare_df.columns and compare_df["tmin_min"].notna().any():
        cold_idx = compare_df["tmin_min"].idxmin()
        cold = compare_df.loc[cold_idx]
        lines.append(
            f"- **{cold['station_id']}** recorded the lowest dry-bulb minimum"
            f" ({float(cold['tmin_min']):.1f}°F)"
        )

    lines.append("")

    # ── Economizer / Tower Rankings ───────────────────────────────────────────
    econ_specs: list[tuple[str, str]] = []
    if "air_econ_hours_sum" in compare_df.columns:
        econ_specs.append(("air_econ_hours_sum", "Airside Economizer Hours (descending = more econ potential)"))
    if "wec_feasible_pct_over_window" in compare_df.columns:
        econ_specs.append(("wec_feasible_pct_over_window", "WEC Feasibility % over Window (descending = more feasible)"))
    if "wec_hours_sum" in compare_df.columns:
        econ_specs.append(("wec_hours_sum", "Waterside Econ Proxy Hours (descending = more WEC potential)"))
    for sc in sorted([c for c in compare_df.columns if c.startswith("tower_stress_hours_wb_gt_") and c.endswith("_sum")]):
        thr = sc.replace("tower_stress_hours_wb_gt_", "").replace("_sum", "")
        econ_specs.append((sc, f"Tower Stress Twb ≥ {thr} °F (descending = more stress)"))
    if "wb_mean_72h_max_max" in compare_df.columns:
        econ_specs.append(("wb_mean_72h_max_max", "Worst 72h Sustained Twb °F (descending)"))
    if "lwt_proxy_p99_median" in compare_df.columns:
        econ_specs.append(("lwt_proxy_p99_median", "LWT Proxy p99 °F (descending = higher condenser water temp)"))

    if econ_specs:
        lines += ["## Economizer / Tower Rankings", ""]
        for col, label in econ_specs:
            if compare_df[col].isna().all():
                continue
            lines += [f"### {label}", ""]
            ranked = (
                compare_df[["station_id", col]]
                .sort_values(col, ascending=False)
                .reset_index(drop=True)
            )
            for i, (_, r) in enumerate(ranked.iterrows(), 1):
                val = r[col]
                val_str = f"{float(val):.1f}" if not pd.isna(val) else "N/A (wetbulb unavailable)"
                lines.append(f"{i}. **{r['station_id']}** — {val_str}")
            lines.append("")

        if "econ_confidence_flag" in compare_df.columns and compare_df["econ_confidence_flag"].any():
            lines += ["**Low-confidence econ/tower metrics (missing data or low wet-bulb availability):**", ""]
            for _, r in compare_df[compare_df["econ_confidence_flag"]].iterrows():
                lines.append(f"- **{r['station_id']}**")
            lines.append("")

    # ── Freeze Risk Rankings ──────────────────────────────────────────────────
    freeze_rank_specs: list[tuple[str, str]] = []
    if "freeze_hours_sum" in compare_df.columns:
        freeze_rank_specs.append(
            ("freeze_hours_sum", "Freeze Hours (descending = more freeze exposure)")
        )
    if "freeze_hours_shoulder_sum" in compare_df.columns:
        freeze_rank_specs.append(
            ("freeze_hours_shoulder_sum", "Shoulder-Season Freeze Hours (descending)")
        )
    if "freeze_event_max_duration_hours_max" in compare_df.columns:
        freeze_rank_specs.append(
            ("freeze_event_max_duration_hours_max", "Longest Freeze Event (h, descending)")
        )

    if freeze_rank_specs:
        lines += ["## Freeze Risk Rankings", ""]
        for col, label in freeze_rank_specs:
            if compare_df[col].isna().all():
                continue
            lines += [f"### {label}", ""]
            ranked = (
                compare_df[["station_id", col]]
                .sort_values(col, ascending=False)
                .reset_index(drop=True)
            )
            for i, (_, r) in enumerate(ranked.iterrows(), 1):
                val = r[col]
                val_str = f"{float(val):.1f}" if not pd.isna(val) else "N/A"
                lines.append(f"{i}. **{r['station_id']}** — {val_str}")
            lines.append("")

        if "freeze_confidence_flag" in compare_df.columns and compare_df["freeze_confidence_flag"].any():
            lines += ["**Low-confidence freeze metrics (missing data):**", ""]
            for _, r in compare_df[compare_df["freeze_confidence_flag"]].iterrows():
                lines.append(f"- **{r['station_id']}**")
            lines.append("")

    # ── Full comparison table ─────────────────────────────────────────────────
    lines += ["## Comparison Table", ""]
    # Show a curated subset of columns for readability
    econ_display = [
        c for c in compare_df.columns
        if c in (
            "air_econ_hours_sum",
            "wec_feasible_pct_over_window",
            "wec_hours_sum",
            "hours_with_wetbulb_sum",
            "wb_mean_72h_max_max",
            "lwt_proxy_p99_median",
        )
    ]
    freeze_display = [
        c for c in compare_df.columns
        if c in (
            "freeze_hours_sum",
            "freeze_hours_shoulder_sum",
            "freeze_event_count_sum",
            "freeze_event_max_duration_hours_max",
        )
    ]
    display_cols = [c for c in [
        "station_id", "years_covered_count", "tmax_max", "tmin_min",
        "t_p99_median", "wb_p99_median",
        *[c for c in compare_df.columns if c.startswith("hours_above_ref_") and c.endswith("_sum")],
        *freeze_display,
        *econ_display,
        "heat_score", "moisture_score", "freeze_score", "data_quality_score", "overall_score",
        "coverage_weighted_pct", "timestamp_missing_pct_avg", "missing_data_warning",
    ] if c in compare_df.columns]
    lines.append(compare_df[display_cols].to_markdown(index=False))
    lines.append("")

    return "\n".join(lines)
