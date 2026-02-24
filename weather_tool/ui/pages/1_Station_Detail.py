"""Station Detail page — coverage badge, annual table, freeze risk, and Decision AI.

Reads from session_state["run_dir"]. All sections degrade gracefully when
Decision AI artifacts are absent.

Tabs:
    Summary        — coverage badge, annual table (DB/WB toggle), footer rollups,
                     annual chart, CSV download
    Freeze Risk    — freeze KPI cards, per-year table, freeze charts
    Decision AI    — design condition cards, exec summary, risk flags, Death Day,
                     recommendations, raw JSON evidence
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from weather_tool.ui import backend
from weather_tool.ui.components import tables as tbl_components

st.title("🏭 Station Detail")

# ── Guard ──────────────────────────────────────────────────────────────────────

if "run_dir" not in st.session_state:
    st.info("No run loaded. Use **⚙️ Run Setup** to run the pipeline or load an existing folder.")
    st.stop()

run_dir: str = st.session_state["run_dir"]
station_ids: list[str] = st.session_state.get("station_ids", [])

if not station_ids:
    st.warning("No stations found in this run folder.")
    st.stop()

# ── Station selector ───────────────────────────────────────────────────────────

selected = st.selectbox("Station", station_ids, key="station_detail_selector")

# ── Load artifacts (all cached) ────────────────────────────────────────────────

packets        = backend.load_packets(run_dir)
exec_summaries = backend.load_exec_summaries(run_dir)

pkt: dict  = packets.get(selected, {})
md: str    = exec_summaries.get(selected, "")
summary_df = backend.load_summary_csv(run_dir, selected)

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_summary, tab_freeze, tab_decision = st.tabs(
    ["📊 Summary", "🥶 Freeze Risk", "🤖 Decision AI"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Summary
# ═══════════════════════════════════════════════════════════════════════════════

with tab_summary:
    if summary_df is None or summary_df.empty:
        st.info("No summary data found for this station. Run the pipeline first.")
    else:
        # ── Coverage badge ──────────────────────────────────────────────────────
        if "missing_pct" in summary_df.columns:
            valid_miss = summary_df["missing_pct"].dropna()
            if not valid_miss.empty:
                worst_idx  = valid_miss.idxmax()
                worst_year = int(summary_df.loc[worst_idx, "year"])
                worst_miss = float(valid_miss.max()) * 100
                avg_miss   = float(valid_miss.mean()) * 100
                n_years    = len(summary_df)
                yr_range   = f"{int(summary_df['year'].min())} \u2013 {int(summary_df['year'].max())}"
                color = "green" if avg_miss < 2 else ("orange" if avg_miss < 5 else "red")
                st.markdown(
                    f"**Data coverage** \u2014 :{color}[avg missing: {avg_miss:.1f}%] | "
                    f"worst year: {worst_year} ({worst_miss:.1f}%) | "
                    f"{n_years} years ({yr_range})"
                )

        # ── WB column preference (chart + extremes rollup) ──────────────────────
        wb_col   = "wb_6h_rollmax_max_f"
        wb_label = "WB Max 6h (\u00b0F)"
        if wb_col not in summary_df.columns or summary_df[wb_col].isna().all():
            wb_col   = "wb_max"
            wb_label = "WB Max (\u00b0F)"

        # ── Ref temp type selector ──────────────────────────────────────────────
        ref_type = st.radio(
            "Hrs Above Ref basis", ["DB", "WB"], horizontal=True, key="summary_ref_type"
        )
        hrs_col = "hours_above_ref" if ref_type == "DB" else "hours_wb_above_ref"

        # ── Annual table ────────────────────────────────────────────────────────
        def _col(name: str) -> pd.Series | None:
            return summary_df[name] if name in summary_df.columns else None

        annual_data: dict = {"Year": summary_df["year"].astype(int)}

        COVERAGE_COLS = [
            ("expected_hours", "Exp Hrs"),
            ("observed_hours", "Obs Hrs"),
            ("missing_hours",  "Missing Hrs"),
        ]
        for src, lbl in COVERAGE_COLS:
            annual_data[lbl] = summary_df[src] if src in summary_df.columns else None

        annual_data["Missing %"] = (
            (summary_df["missing_pct"] * 100).round(1)
            if "missing_pct" in summary_df.columns else None
        )
        annual_data["NaN Temp"] = (
            summary_df["nan_temp_count"]
            if "nan_temp_count" in summary_df.columns else None
        )

        if _col("tmax") is not None:
            annual_data["Max DB (\u00b0F)"] = _col("tmax")
        if _col("tmin") is not None:
            annual_data["Min DB (\u00b0F)"] = _col("tmin")
        if _col("ref_temp") is not None:
            annual_data["Ref Temp (\u00b0F)"] = _col("ref_temp")

        hrs_series = _col(hrs_col)
        if hrs_series is not None:
            annual_data["Hrs Above Ref"] = hrs_series

        thwt = _col("total_hours_with_temp")
        if hrs_series is not None and thwt is not None:
            pct_above = (hrs_series / thwt.replace(0, float("nan")) * 100).round(1)
            annual_data["% Above"] = pct_above

        annual_df = pd.DataFrame(annual_data)
        st.dataframe(annual_df, use_container_width=True)

        # ── Footer rollups ──────────────────────────────────────────────────────
        st.subheader("Rollups")

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            if "expected_hours" in summary_df.columns:
                st.metric("Total Exp Hours", f"{int(summary_df['expected_hours'].sum()):,}")
        with rc2:
            if "observed_hours" in summary_df.columns:
                st.metric("Total Obs Hours", f"{int(summary_df['observed_hours'].sum()):,}")
        with rc3:
            if "missing_hours" in summary_df.columns:
                st.metric("Total Missing Hrs", f"{int(summary_df['missing_hours'].sum()):,}")
        with rc4:
            if "missing_hours" in summary_df.columns and "expected_hours" in summary_df.columns:
                denom = int(summary_df["expected_hours"].sum())
                if denom > 0:
                    pct = summary_df["missing_hours"].sum() / denom * 100
                    st.metric("Overall % Missing", f"{pct:.1f}%")
                else:
                    st.metric("Overall % Missing", "N/A")
            else:
                st.metric("Overall % Missing", "N/A")

        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            if "tmax" in summary_df.columns:
                st.metric("DB Max (\u00b0F)", f"{summary_df['tmax'].max():.1f}")
        with ex2:
            if "tmin" in summary_df.columns:
                st.metric("DB Min (\u00b0F)", f"{summary_df['tmin'].min():.1f}")
        with ex3:
            if wb_col in summary_df.columns and not summary_df[wb_col].isna().all():
                st.metric(wb_label, f"{summary_df[wb_col].max():.1f}")
            else:
                st.metric("WB Max", "N/A")

        # ── CSV download ────────────────────────────────────────────────────────
        st.download_button(
            "\u2b07 Download annual table",
            data=annual_df.to_csv(index=False),
            file_name=f"{selected}_annual.csv",
            mime="text/csv",
        )

        st.divider()

        # ── Annual chart ────────────────────────────────────────────────────────
        import plotly.express as px

        chart_rows = []
        for _, row in summary_df.iterrows():
            yr = int(row["year"])
            if "tmax" in summary_df.columns and pd.notna(row.get("tmax")):
                chart_rows.append({"Year": yr, "Value": row["tmax"], "Metric": "DB Max (\u00b0F)"})
            if wb_col in summary_df.columns and pd.notna(row.get(wb_col)):
                chart_rows.append({"Year": yr, "Value": row[wb_col], "Metric": wb_label})

        if chart_rows:
            chart_long = pd.DataFrame(chart_rows)
            fig = px.line(
                chart_long,
                x="Year", y="Value", color="Metric",
                height=380,
                labels={"Value": "\u00b0F"},
            )
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title_text="")
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"staticPlot": True, "displayModeBar": False},
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Freeze Risk
# ═══════════════════════════════════════════════════════════════════════════════

with tab_freeze:
    _FREEZE_COLS = [
        "freeze_hours", "freeze_event_count",
        "freeze_event_max_duration_hours", "freeze_hours_shoulder",
    ]
    if summary_df is None or summary_df.empty or \
            not any(c in summary_df.columns for c in _FREEZE_COLS):
        st.info("Freeze data not available in this run folder.")
    else:
        # ── KPI cards ───────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            if "freeze_hours" in summary_df.columns:
                st.metric("Total Freeze Hours", f"{summary_df['freeze_hours'].sum():,.1f}")
        with k2:
            if "freeze_event_count" in summary_df.columns:
                st.metric("Total Events", f"{int(summary_df['freeze_event_count'].sum())}")
        with k3:
            if "freeze_event_max_duration_hours" in summary_df.columns:
                st.metric(
                    "Longest Event (h)",
                    f"{summary_df['freeze_event_max_duration_hours'].max():.1f}",
                )
        with k4:
            if "freeze_hours_shoulder" in summary_df.columns:
                st.metric("Shoulder Hrs", f"{summary_df['freeze_hours_shoulder'].sum():,.1f}")

        st.divider()

        # ── Per-year freeze table ───────────────────────────────────────────────
        FREEZE_LABEL_MAP = {
            "year":                          "Year",
            "freeze_hours":                  "Freeze Hrs",
            "freeze_event_count":            "Events",
            "freeze_event_max_duration_hours": "Longest Event (h)",
            "freeze_hours_shoulder":         "Shoulder Hrs",
        }
        freeze_cols = [c for c in FREEZE_LABEL_MAP if c in summary_df.columns]
        freeze_df   = summary_df[freeze_cols].rename(columns=FREEZE_LABEL_MAP)
        st.dataframe(freeze_df, use_container_width=True)

        st.divider()

        # ── Freeze charts ───────────────────────────────────────────────────────
        import plotly.express as px

        if "freeze_hours" in summary_df.columns:
            fig_fh = px.bar(
                summary_df, x="year", y="freeze_hours",
                labels={"freeze_hours": "Freeze Hours", "year": "Year"},
                height=280,
            )
            fig_fh.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(
                fig_fh,
                use_container_width=True,
                config={"staticPlot": True, "displayModeBar": False},
            )

        if "freeze_event_count" in summary_df.columns:
            fig_fe = px.bar(
                summary_df, x="year", y="freeze_event_count",
                labels={"freeze_event_count": "Freeze Events", "year": "Year"},
                height=280,
            )
            fig_fe.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(
                fig_fe,
                use_container_width=True,
                config={"staticPlot": True, "displayModeBar": False},
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Decision AI
# ═══════════════════════════════════════════════════════════════════════════════

with tab_decision:
    if not pkt:
        st.info(
            "No Decision AI data. Re-run with a Decision Profile other than 'None'."
        )
    else:
        # ── Design condition cards ─────────────────────────────────────────────
        design = pkt.get("design_conditions", {})
        if design:
            key_metrics = {
                "Tdb p99 (\u00b0F)":   design.get("tdb_p99"),
                "Tdb p99.6 (\u00b0F)": design.get("tdb_p996"),
                "WB p99 (\u00b0F)":    design.get("wb_p99"),
                "WB p99.6 (\u00b0F)":  design.get("wb_p996"),
                "72h Tdb max (\u00b0F)": design.get("tdb_mean_72h_max"),
                "72h WB max (\u00b0F)":  design.get("wb_mean_72h_max"),
            }
            cols = st.columns(3)
            for i, (label, val) in enumerate(key_metrics.items()):
                display = (
                    f"{val:.1f}"
                    if isinstance(val, (int, float)) and val is not None
                    else "N/A"
                )
                cols[i % 3].metric(label, display)

        st.divider()

        # ── Exec summary markdown ──────────────────────────────────────────────
        st.subheader("Executive Engineering Summary")
        if md:
            st.markdown(md)
        else:
            st.info(
                "No exec summary found. Re-run with a Decision Profile other than 'None' "
                "to generate the Decision AI packet and summary."
            )

        st.divider()

        # ── Freeze normalization toggle ────────────────────────────────────────
        # Widget key "freeze_norm_mode" is intentionally shared with Compare page
        # so the user's selection persists when navigating between pages.
        norm_mode = st.radio(
            "Freeze threshold mode",
            ["per_year", "aggregate"],
            format_func=lambda x: "Per year (default)" if x == "per_year" else "Aggregate (full window)",
            horizontal=True,
            key="freeze_norm_mode",
        )

        # ── Risk Flags ─────────────────────────────────────────────────────────
        st.subheader("Risk Flags")
        flag_df = backend.flags_table({selected: pkt}, normalization_mode=norm_mode)
        tbl_components.render_flags_table(flag_df, {selected: pkt})

        st.divider()

        # ── Death Day Candidates ───────────────────────────────────────────────
        st.subheader("Death Day Candidates")
        death_day  = pkt.get("death_day", {})
        candidates: list[dict] = death_day.get("candidates", [])

        if death_day:
            mode_dd  = death_day.get("mode", "unknown")
            window_h = death_day.get("window_hours", "?")
            st.caption(f"Mode: **{mode_dd}** | Window: **{window_h}h**")

            if mode_dd == "heat_day":
                st.warning(
                    "Wet-bulb data unavailable \u2014 candidates ranked by dry-bulb stress only "
                    "(low confidence). Re-run with `dwpf` or `relh` fields for full Death Day analysis."
                )

        if candidates:
            st.dataframe(pd.DataFrame(candidates), use_container_width=True)
        else:
            st.info(
                "No Death Day candidates found "
                "(no heatwave-level windows detected in the record)."
            )

        # ── Recommendations ────────────────────────────────────────────────────
        recs: list[dict] = pkt.get("recommendations", [])
        if recs:
            st.divider()
            st.subheader("Recommendations")
            for i, r in enumerate(recs, 1):
                sev  = r.get("severity", "low")
                icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(sev, "•")
                with st.expander(
                    f"{i}. {icon} {r.get('title', '')} *({sev})*",
                    expanded=(sev == "high"),
                ):
                    if r.get("rationale"):
                        st.write(r["rationale"])
                    evidence = r.get("evidence", [])
                    if evidence:
                        st.dataframe(pd.DataFrame(evidence), use_container_width=True)

        st.divider()

        # ── Raw evidence expanders ─────────────────────────────────────────────
        with st.expander("Full station packet JSON", expanded=False):
            st.json(pkt)

        quality_info = pkt.get("quality", {})
        if quality_info:
            with st.expander("Quality report", expanded=False):
                st.json(quality_info)

        oe = pkt.get("operational_efficiency", {})
        if oe:
            with st.expander("Operational efficiency JSON", expanded=False):
                st.json(oe)
