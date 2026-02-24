"""Compare page — multi-station rankings, exceedance plot, risk flags, Death Day.

Reads from session_state["run_dir"]. Degrades gracefully when Decision AI
artifacts are absent (non-Decision-AI compare runs show compare_summary.csv).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from weather_tool.ui import backend
from weather_tool.ui.components import plots as plot_components
from weather_tool.ui.components import tables as tbl_components

st.title("📊 Compare Stations")

# ── Guard ──────────────────────────────────────────────────────────────────────

if "run_dir" not in st.session_state:
    st.info("No run loaded. Use **⚙️ Run Setup** to run the pipeline or load an existing folder.")
    st.stop()

run_dir: str = st.session_state["run_dir"]
station_ids: list[str] = st.session_state.get("station_ids", [])

# ── Load artifacts (all cached) ────────────────────────────────────────────────

packets = backend.load_packets(run_dir)
compare_pkt = backend.load_compare_packet(run_dir)
compare_csv = backend.load_compare_summary_csv(run_dir)

# ── Toggles ────────────────────────────────────────────────────────────────────

col_t1, col_t2 = st.columns(2)

with col_t1:
    chart_mode = st.selectbox(
        "Chart mode",
        ["Tower Stress", "Economizer"],
        key="compare_chart_mode",
        help="Tower Stress: WB hours above cooling tower thresholds (75 / 78 / 80 °F). "
             "Economizer: dry-bulb free-cooling opportunity.",
    )

with col_t2:
    # Widget key "freeze_norm_mode" is intentionally shared with Station Detail
    # so the user's selection persists when navigating between pages.
    norm_mode = st.radio(
        "Freeze threshold mode",
        ["per_year", "aggregate"],
        format_func=lambda x: "Per year (default)" if x == "per_year" else "Aggregate (full window)",
        horizontal=True,
        key="freeze_norm_mode",
        help="Controls freeze normalization in the risk flags table",
    )

st.divider()

# ── Rankings ───────────────────────────────────────────────────────────────────

st.header("Rankings")

SCORE_LABEL_MAP = {
    "overall_score":      "Overall",
    "heat_score":         "Heat",
    "moisture_score":     "Moisture",
    "freeze_score":       "Freeze",
    "data_quality_score": "Data Quality",
}

if compare_pkt and compare_pkt.get("rankings"):
    rankings = compare_pkt["rankings"]
    score_frames = {}
    for key, lbl in SCORE_LABEL_MAP.items():
        rows = rankings.get(key, [])
        if rows:
            score_frames[lbl] = (
                pd.DataFrame(rows)[["station_id", "value"]]
                .rename(columns={"value": lbl})
                .set_index("station_id")
            )
    if score_frames:
        rank_df = pd.concat(score_frames.values(), axis=1).reset_index()
        os_rank = {r["station_id"]: r["rank"] for r in rankings.get("overall_score", [])}
        rank_df.insert(0, "Rank", rank_df["station_id"].map(os_rank))
        rank_df = rank_df.sort_values("Rank").rename(columns={"station_id": "Station"})
        st.dataframe(rank_df, use_container_width=True)
    else:
        st.json(rankings)

elif compare_csv is not None:
    st.caption("(Decision AI not enabled — showing compare_summary.csv)")
    st.dataframe(compare_csv, use_container_width=True)

elif packets and len(packets) > 1:
    # Build a quick side-by-side key metrics table from individual packets
    rows = []
    for sid, pkt in packets.items():
        dc = pkt.get("design_conditions", {})
        oe = pkt.get("operational_efficiency", {})
        fr = pkt.get("freeze_risk", {})
        rows.append(
            {
                "station_id": sid,
                "tdb_p996": dc.get("tdb_p996"),
                "wb_p996": dc.get("wb_p996"),
                "air_econ_hours_sum": oe.get("air_econ_hours_sum"),
                "wec_feasible_pct": oe.get("wec_feasible_pct_over_window"),
                "freeze_hours_sum": fr.get("freeze_hours_sum"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

else:
    st.info("No compare data available. Run with multiple stations to enable comparison.")

st.divider()

# ── Exceedance Hours Plot ──────────────────────────────────────────────────────

st.header("Exceedance Hours Plot")

# Load per-station summary CSVs (cached)
sids_for_plot = station_ids or list(packets.keys())
summary_data: dict[str, object] = {
    sid: backend.load_summary_csv(run_dir, sid) for sid in sids_for_plot
}

plot_components.exceedance_hours_bar_chart(summary_data, chart_mode, norm_mode)

with st.expander("🔍 Debug: plot data", expanded=False):
    for sid, df in summary_data.items():
        if df is not None and not df.empty:
            st.caption(f"**{sid}**")
            st.dataframe(df, use_container_width=True)

st.divider()

# ── Risk Flags ────────────────────────────────────────────────────────────────

st.header("Risk Flags")

if packets:
    flag_df = backend.flags_table(packets, normalization_mode=norm_mode)
    tbl_components.render_flags_table(flag_df, packets)
elif compare_pkt and compare_pkt.get("aggregated_flags"):
    # Non-Decision-AI per-station packets absent, but compare packet has
    # aggregated high-severity flags — show a minimal scalar table.
    agg_rows = [
        {
            "station": f.get("station_id", ""),
            "flag_id": f.get("flag_id", ""),
            "severity": f.get("severity", ""),
            "confidence": "",
            "evidence_summary": "",
            "freeze_hours_norm": None,
            "freeze_hours_display": "",
            "normalization_mode": "",
            "notes": "",
        }
        for f in compare_pkt["aggregated_flags"]
    ]
    tbl_components.render_flags_table(pd.DataFrame(agg_rows), {})
else:
    st.info("Re-run with **--decision-ai** to see risk flags.")

st.divider()

# ── Death Day Rank-1 Candidates ───────────────────────────────────────────────

st.header("Worst Death Day Windows (Rank 1 per Station)")

dd_rows: list[dict] = []
for sid, pkt in packets.items():
    candidates = pkt.get("death_day", {}).get("candidates", [])
    if candidates:
        c = candidates[0]
        dd_rows.append(
            {
                "station": sid,
                "start_ts": c.get("start_ts"),
                "end_ts": c.get("end_ts"),
                "stress_score": c.get("stress_score"),
                "tdb_mean_f": c.get("tdb_mean_f"),
                "tdb_max_f": c.get("tdb_max_f"),
                "twb_mean_f": c.get("twb_mean_f"),
                "twb_max_f": c.get("twb_max_f"),
                "mode": c.get("mode"),
                "confidence": c.get("confidence"),
            }
        )

if dd_rows:
    st.dataframe(pd.DataFrame(dd_rows), use_container_width=True)
elif packets:
    st.info("No Death Day candidates found (insufficient data or no heatwave events).")
else:
    st.info("Re-run with **--decision-ai** to see Death Day analysis.")

# ── Cross-station extremes ─────────────────────────────────────────────────────

if compare_pkt and compare_pkt.get("cross_station_extremes"):
    st.divider()
    st.header("Cross-Station Extremes")
    extremes = compare_pkt["cross_station_extremes"]
    ext_rows = [{"metric": k, "value": v} for k, v in extremes.items()]
    st.dataframe(pd.DataFrame(ext_rows), use_container_width=True)
