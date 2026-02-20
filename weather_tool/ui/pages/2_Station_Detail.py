"""Station Detail page — tabbed view of design conditions, exec summary, and raw packet.

Reads from session_state["run_dir"]. All sections degrade gracefully when
Decision AI artifacts are absent.

Tabs:
    Summary          — key metric cards + per-year summary CSV table
    Executive Summary — exec MD, freeze toggle, risk flags, Death Day, recommendations
    Raw / Evidence   — full packet JSON + quality JSON
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

packets = backend.load_packets(run_dir)
exec_summaries = backend.load_exec_summaries(run_dir)

pkt: dict = packets.get(selected, {})
md: str = exec_summaries.get(selected, "")
summary_df = backend.load_summary_csv(run_dir, selected)

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_summary, tab_exec, tab_raw = st.tabs(
    ["📊 Summary", "📝 Executive Summary", "🔬 Raw / Evidence"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Summary
# ═══════════════════════════════════════════════════════════════════════════════

with tab_summary:
    # ── Key metric cards ───────────────────────────────────────────────────────
    design = pkt.get("design_conditions", {})
    if design:
        key_metrics = {
            "Tdb p99 (°F)": design.get("tdb_p99"),
            "Tdb p99.6 (°F)": design.get("tdb_p996"),
            "WB p99 (°F)": design.get("wb_p99"),
            "WB p99.6 (°F)": design.get("wb_p996"),
            "72h Tdb max (°F)": design.get("tdb_mean_72h_max"),
            "72h WB max (°F)": design.get("wb_mean_72h_max"),
        }
        cols = st.columns(3)
        for i, (label, val) in enumerate(key_metrics.items()):
            display = (
                f"{val:.1f}"
                if isinstance(val, (int, float)) and val is not None
                else "N/A"
            )
            cols[i % 3].metric(label, display)
    else:
        st.info("No design condition data. Re-run with **--decision-ai**.")

    st.divider()

    # ── Per-year summary CSV ───────────────────────────────────────────────────
    st.subheader("Per-Year Summary")
    if summary_df is not None and not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("No summary CSV found for this station in the run folder.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Executive Summary
# ═══════════════════════════════════════════════════════════════════════════════

with tab_exec:
    # ── Exec summary markdown ──────────────────────────────────────────────────
    st.subheader("Executive Engineering Summary")
    if md:
        st.markdown(md)
    else:
        st.info(
            "No exec summary found. Re-run with **--decision-ai** to generate "
            "the Decision AI packet and summary."
        )

    st.divider()

    # ── Freeze normalization toggle ────────────────────────────────────────────
    # Widget key "freeze_norm_mode" is intentionally shared with the Compare page
    # so the user's selection persists when navigating between pages.
    norm_mode = st.radio(
        "Freeze threshold mode",
        ["per_year", "aggregate"],
        format_func=lambda x: "Per year (default)" if x == "per_year" else "Aggregate (full window)",
        horizontal=True,
        key="freeze_norm_mode",
    )

    # ── Risk Flags ─────────────────────────────────────────────────────────────
    st.subheader("Risk Flags")
    if pkt:
        flag_df = backend.flags_table({selected: pkt}, normalization_mode=norm_mode)
        tbl_components.render_flags_table(flag_df, {selected: pkt})
    else:
        st.info("Re-run with **--decision-ai** to evaluate risk flags.")

    st.divider()

    # ── Death Day Candidates ───────────────────────────────────────────────────
    st.subheader("Death Day Candidates")
    death_day = pkt.get("death_day", {})
    candidates: list[dict] = death_day.get("candidates", [])

    if death_day:
        mode = death_day.get("mode", "unknown")
        window_h = death_day.get("window_hours", "?")
        st.caption(f"Mode: **{mode}** | Window: **{window_h}h**")

        if mode == "heat_day":
            st.warning(
                "Wet-bulb data unavailable — candidates ranked by dry-bulb stress only "
                "(low confidence). Re-run with `dwpf` or `relh` fields for full Death Day analysis."
            )

    if candidates:
        st.dataframe(pd.DataFrame(candidates), use_container_width=True)
    elif pkt:
        st.info("No Death Day candidates found (no heatwave-level windows detected in the record).")
    else:
        st.info("Re-run with **--decision-ai** to compute Death Day candidates.")

    # ── Recommendations ────────────────────────────────────────────────────────
    recs: list[dict] = pkt.get("recommendations", [])
    if recs:
        st.divider()
        st.subheader("Recommendations")
        for i, r in enumerate(recs, 1):
            sev = r.get("severity", "low")
            icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(sev, "•")
            with st.expander(
                f"{i}. {icon} {r.get('title', '')} *({sev})*",
                expanded=(sev == "high"),
            ):
                if r.get("rationale"):
                    st.write(r["rationale"])
                evidence = r.get("evidence", [])
                if evidence:
                    ev_df = pd.DataFrame(evidence)
                    st.dataframe(ev_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Raw / Evidence
# ═══════════════════════════════════════════════════════════════════════════════

with tab_raw:
    if pkt:
        with st.expander("Full station packet JSON", expanded=False):
            st.json(pkt)

        quality = pkt.get("quality", {})
        if quality:
            with st.expander("Quality report", expanded=False):
                st.json(quality)

        oe = pkt.get("operational_efficiency", {})
        if oe:
            with st.expander("Operational efficiency JSON", expanded=False):
                st.json(oe)
    else:
        st.info(
            "No station packet found. Re-run with **--decision-ai** to generate the packet."
        )
