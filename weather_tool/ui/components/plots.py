"""Chart components for the Streamlit UI.

All functions take pre-loaded DataFrames and render directly via st.*
No weather math — display only.
"""

from __future__ import annotations

import streamlit as st

# Plotly config applied to every chart: no pan/zoom, no modebar, fixed height.
_PLOTLY_CONFIG = {"staticPlot": True, "displayModeBar": False}
_CHART_HEIGHT  = 380

# Tower Stress threshold columns and display labels (ordered low → high)
_TOWER_STRESS_COLS: dict[str, str] = {
    "tower_stress_hours_wb_gt_75": "WB > 75°F",
    "tower_stress_hours_wb_gt_78": "WB > 78°F",
    "tower_stress_hours_wb_gt_80": "WB > 80°F",
}


def exceedance_hours_bar_chart(
    summary_data: "dict[str, object]",  # dict[str, pd.DataFrame | None]
    chart_mode: str,
    norm_mode: str,
) -> None:
    """Render a bar chart of exceedance/stress hours per station.

    Parameters
    ----------
    summary_data : {station_id: summary_df} — per-year summary DataFrames.
    chart_mode   : "Tower Stress" or "Economizer".
    norm_mode    : "per_year" or "aggregate".
    """
    import pandas as pd
    import plotly.express as px

    # ── Column selection by chart_mode ────────────────────────────────────────
    if chart_mode == "Tower Stress":
        # Preferred: three tiered stress columns (melted into one multi-series chart)
        # Fallback: single wet-bulb exceedance column
        primary_cols  = list(_TOWER_STRESS_COLS.keys())
        fallback_cols = ["exceedance_hours_twb_p99"]
    else:  # Economizer
        primary_cols  = ["air_econ_hours"]
        fallback_cols = ["hours_above_ref", "exceedance_hours_tdb_p99"]

    # ── Build combined DataFrame ───────────────────────────────────────────────
    frames: list[pd.DataFrame] = []
    used_fallback = False
    missing_reason: str | None = None

    for sid, df in summary_data.items():
        if df is None or df.empty:
            continue

        avail_primary = [c for c in primary_cols if c in df.columns]
        if avail_primary:
            cols = avail_primary
        else:
            avail_fallback = [c for c in fallback_cols if c in df.columns]
            if avail_fallback:
                cols = [avail_fallback[0]]
                used_fallback = True
            else:
                missing_reason = (
                    f"No {chart_mode} columns found. "
                    "Re-run with the appropriate IEM fields enabled."
                )
                continue

        tmp = df[["year"] + cols].copy()
        tmp["station"] = sid
        frames.append(tmp)

    if not frames:
        reason = missing_reason or f"No summary data available for chart mode '{chart_mode}'."
        st.info(reason)
        return

    if used_fallback:
        fallback_name = [c for c in frames[0].columns if c not in ("year", "station")]
        fallback_name = fallback_name[0] if fallback_name else "fallback"
        st.caption(f"⚠️ Primary {chart_mode} columns not found — showing `{fallback_name}` instead.")

    combined = pd.concat(frames, ignore_index=True)

    # ── Tower Stress: ONE multi-series chart (melt thresholds as color) ────────
    avail_stress = {k: v for k, v in _TOWER_STRESS_COLS.items() if k in combined.columns}
    if chart_mode == "Tower Stress" and avail_stress:
        long_df = (
            combined[["year", "station"] + list(avail_stress.keys())]
            .rename(columns=avail_stress)
            .melt(id_vars=["year", "station"], var_name="Threshold", value_name="Hours")
        )
        long_df["Hours"] = long_df["Hours"].fillna(0)

        if norm_mode == "per_year":
            multi_station = combined["station"].nunique() > 1
            fig = px.bar(
                long_df,
                x="year", y="Hours", color="Threshold",
                facet_col="station" if multi_station else None,
                barmode="group",
                labels={"year": "Year"},
                height=_CHART_HEIGHT,
            )
        else:
            agg_df = (
                long_df.groupby(["station", "Threshold"], sort=False)["Hours"]
                .sum()
                .reset_index()
            )
            fig = px.bar(
                agg_df,
                x="station", y="Hours", color="Threshold",
                barmode="group",
                labels={"station": "Station"},
                height=_CHART_HEIGHT,
            )

        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CONFIG)
        return  # single chart rendered — done

    # ── Economizer (or Tower Stress single-column fallback): one chart per col ─
    value_cols = [c for c in combined.columns if c not in ("year", "station")]

    for col in value_cols:
        label = (
            col.replace("exceedance_hours_", "")
               .replace("_p99", " p99")
               .replace("_", " ")
               .upper()
        )
        st.caption(f"**{label}**")

        if norm_mode == "per_year":
            fig = px.bar(
                combined[["year", "station", col]].fillna(0),
                x="year", y=col, color="station",
                barmode="group",
                labels={col: label, "year": "Year", "station": "Station"},
                height=_CHART_HEIGHT,
            )
        else:
            agg_df = combined.groupby("station")[col].sum().reset_index()
            agg_df.columns = ["Station", label]
            fig = px.bar(agg_df, x="Station", y=label, height=_CHART_HEIGHT)

        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CONFIG)
