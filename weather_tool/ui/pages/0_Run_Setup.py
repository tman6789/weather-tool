"""Run Setup page — pipeline controls and folder loader.

All Run/Load controls live here. On success, writes session_state keys
and busts the loader cache. Other pages read from session_state only.
"""

from __future__ import annotations

import datetime
import uuid
from datetime import date
from pathlib import Path

import streamlit as st

from weather_tool.ui import backend

st.title("⚙️ Run Setup")
st.caption("Configure and launch the pipeline, or load an existing output folder.")


def _display(sid: str) -> str:
    """Show 'ORD' instead of 'KORD' for 4-char US ICAO codes (display only)."""
    return sid[1:] if len(sid) == 4 and sid.startswith("K") else sid


# ── Mode selector ──────────────────────────────────────────────────────────────

run_mode = st.radio(
    "Mode",
    ["Run pipeline", "Load existing folder"],
    index=0,
    horizontal=True,
    key="run_mode_radio",
)

st.divider()

# ── RUN PIPELINE ───────────────────────────────────────────────────────────────

if run_mode == "Run pipeline":
    st.subheader("Pipeline configuration")

    station_ids_str = st.text_input(
        "Station IDs (comma-separated)",
        value="KORD",
        help="ICAO station codes, e.g. KORD,KPHX,KIAD",
    )

    _end_yr = date.today().year - 1
    time_range = st.radio(
        "Time range",
        ["Past 5 years", "Past 10 years", "Past 20 years", "Custom"],
        horizontal=True,
        index=1,   # default: Past 10
        key="time_range_radio",
    )
    _n_map = {"Past 5 years": 5, "Past 10 years": 10, "Past 20 years": 20}
    if time_range == "Custom":
        col_y1, col_y2 = st.columns(2)
        year_start = col_y1.number_input("Start year", min_value=1990, max_value=2030, value=2018)
        year_end   = col_y2.number_input("End year",   min_value=1990, max_value=2030, value=_end_yr)
    else:
        year_end   = _end_yr
        year_start = _end_yr - _n_map[time_range] + 1

    ref_temp = st.number_input(
        "Ref temp (°F)",
        value=65.0,
        step=1.0,
        help="Reference temperature for hours-above-ref metric",
    )

    profile = st.selectbox(
        "Decision profile",
        options=["None", "datacenter", "economizer_first", "freeze_sensitive"],
        index=1,   # default: datacenter
        help="Rules engine profile for risk flag evaluation. 'None' disables Decision AI.",
    )

    wind_rose = st.toggle(
        "Wind analysis",
        value=False,
        help="Compute wind roses and co-occurrence events (sknt + drct added automatically; slow)",
    )

    st.divider()

    if st.button("▶ Run", type="primary", use_container_width=False):
        sids = [s.strip().upper() for s in station_ids_str.split(",") if s.strip()]
        if not sids:
            st.error("Enter at least one station ID.")
        elif int(year_start) > int(year_end):
            st.error("Start year must be ≤ end year.")
        else:
            start_date = date(int(year_start), 1, 1)
            end_date   = date(int(year_end), 12, 31)
            decision_ai = profile != "None"
            _ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            _uid = uuid.uuid4().hex[:6]
            outdir = Path("outputs") / "ui_runs" / f"{_ts}_{_uid}"
            try:
                with st.spinner(f"Running pipeline for {', '.join(sids)}..."):
                    out = backend.run_pipeline_and_save(
                        station_ids=sids,
                        start=start_date,
                        end=end_date,
                        ref_temp=float(ref_temp),
                        profile=str(profile),
                        decision_ai=decision_ai,
                        wind_rose=bool(wind_rose),
                        outdir=outdir,
                    )
                # Bust cached loaders before writing session_state so stale data
                # from a previous run into the same folder is not returned.
                st.cache_data.clear()
                st.session_state["run_dir"] = out
                st.session_state["station_ids"] = sids
                st.session_state["is_compare"] = len(sids) > 1
                st.session_state["run_meta"] = {
                    "start": str(start_date),
                    "end": str(end_date),
                    "profile": profile,
                    "decision_ai": decision_ai,
                    "wind_rose": wind_rose,
                }
                st.success(f"Run complete → {out}")
                st.caption("Navigate to a page using the left-hand menu.")
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")

# ── LOAD EXISTING FOLDER ───────────────────────────────────────────────────────

else:
    st.subheader("Load existing run folder")

    folder_path_input = st.text_input(
        "Run folder path",
        value="outputs",
        help="Path to a single-station output folder or a compare_* subfolder",
    )

    if st.button("Load", type="primary", use_container_width=False):
        rd = Path(folder_path_input)
        if not rd.exists():
            st.error(f"Folder not found: {rd}")
        else:
            # Bust cache before loading a potentially different directory
            st.cache_data.clear()

            # Derive station_ids via fallback chain:
            #   compare_packet → compare_metadata → compare_summary.csv → station packets
            compare_pkt = backend.load_compare_packet(str(rd))
            compare_meta = backend.load_compare_metadata(str(rd))
            compare_csv = backend.load_compare_summary_csv(str(rd))
            pkts = backend.load_packets(str(rd))

            if compare_pkt and compare_pkt.get("meta", {}).get("stations"):
                sids = compare_pkt["meta"]["stations"]
            elif compare_meta and compare_meta.get("stations"):
                sids = compare_meta["stations"]
            elif compare_csv is not None and "station_id" in compare_csv.columns:
                sids = compare_csv["station_id"].unique().tolist()
            elif pkts:
                sids = list(pkts.keys())
            else:
                sids = []

            is_compare = (compare_pkt or compare_meta or compare_csv) is not None

            st.session_state["run_dir"] = str(rd)
            st.session_state["station_ids"] = sids
            st.session_state["is_compare"] = is_compare
            st.session_state["run_meta"] = {}

            if sids:
                st.success(f"Loaded {len(sids)} station(s) from {rd.name}")
                st.caption("Navigate to a page using the left-hand menu.")
            else:
                st.warning(
                    f"No station data found in {rd}. "
                    "Check that the folder contains summary CSV or station packet files."
                )

# ── Active run status ──────────────────────────────────────────────────────────

if "run_dir" in st.session_state:
    st.divider()
    st.subheader("Current run")
    run_dir = st.session_state["run_dir"]
    sids_display = st.session_state.get("station_ids", [])
    meta = st.session_state.get("run_meta", {})

    st.code(run_dir, language=None)
    if sids_display:
        st.write(f"**Stations:** {', '.join(_display(s) for s in sids_display)}")
    if meta.get("start"):
        st.write(
            f"**Window:** {meta['start']} → {meta['end']} | "
            f"**Profile:** {meta.get('profile', '—')}"
        )
