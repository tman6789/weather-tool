"""Wind page — wind rose images, hours tables, co-occurrence event stats.

Reads from session_state["run_dir"]. Skips gracefully if no wind artifacts exist.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from weather_tool.ui import backend

st.title("🌬️ Wind Analysis")

# ── Guard ──────────────────────────────────────────────────────────────────────

if "run_dir" not in st.session_state:
    st.info("No run loaded. Use the sidebar to run the pipeline or load an existing folder.")
    st.stop()

run_dir: str = st.session_state["run_dir"]

# ── Load wind artifacts (cached) ───────────────────────────────────────────────

wind_artifacts = backend.load_wind_artifacts(run_dir)

if not wind_artifacts:
    st.info(
        "No wind outputs found in this run folder. "
        "Enable **Wind analysis** on the ⚙️ Run Setup page, or pass **--wind-rose** "
        "on the CLI (sknt + drct are added automatically by the UI)."
    )
    st.stop()

# ── Station + slice selectors ──────────────────────────────────────────────────

station_ids_with_wind = sorted(wind_artifacts.keys())
selected_station = st.selectbox("Station", station_ids_with_wind, key="wind_station_selector")

station_data = wind_artifacts.get(selected_station, {})
slices = station_data.get("slices", {})

if not slices:
    st.warning(f"No wind rose data found for {selected_station}.")
    st.stop()

selected_slice = st.selectbox(
    "Season / Slice",
    sorted(slices.keys()),
    key="wind_slice_selector",
)

slice_data = slices[selected_slice]
csv_path_str: str | None = slice_data.get("csv_path")
png_path_str: str | None = slice_data.get("png_path")
events_path_str: str | None = station_data.get("events_path")

st.divider()

# ── Wind Rose Image ────────────────────────────────────────────────────────────

st.subheader(f"Wind Rose — {selected_station} / {selected_slice}")

if png_path_str and Path(png_path_str).exists():
    st.image(png_path_str, use_container_width=True)
else:
    st.info(
        "Wind rose PNG not available. "
        "Install matplotlib to generate PNGs: `pip install 'weather-tool[viz]'`"
    )

# ── Wind Rose Hours Table ──────────────────────────────────────────────────────

st.subheader("Wind Rose — Hours by Direction and Speed")

if csv_path_str and Path(csv_path_str).exists():
    try:
        # The CSV has leading comment lines starting with "#" (metadata headers).
        # Skip them and parse the remaining CSV with direction bins as index.
        lines: list[str] = []
        meta_lines: list[str] = []
        with open(csv_path_str, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    meta_lines.append(line.lstrip("# ").strip())
                else:
                    lines.append(line)

        if meta_lines:
            with st.expander("Rose metadata"):
                for ml in meta_lines:
                    st.caption(ml)

        rose_df = pd.read_csv(io.StringIO("".join(lines)), index_col=0)
        st.dataframe(rose_df, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not parse wind rose CSV: {exc}")
else:
    st.info("Wind rose CSV not found.")

# ── Co-occurrence Event Stats ──────────────────────────────────────────────────

st.divider()
st.subheader("Wind Co-occurrence Events")

if events_path_str and Path(events_path_str).exists():
    try:
        events: dict = json.loads(Path(events_path_str).read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Could not load wind events JSON: {exc}")
        events = {}

    if not events:
        st.info("No co-occurrence events recorded.")
    else:
        # Filter to events for the selected station and slice
        matching = {k: v for k, v in events.items() if selected_slice in k}
        if not matching:
            matching = events   # fall back: show all

        for event_key, ev in matching.items():
            with st.expander(f"**{event_key}**", expanded=False):
                # Scalar summary fields
                scalar = {
                    k: v for k, v in ev.items()
                    if not isinstance(v, (dict, list)) and v is not None
                }
                if scalar:
                    st.dataframe(
                        pd.DataFrame([scalar]),
                        use_container_width=True,
                    )

                # Sector percentage bar chart
                sector_pct: dict = ev.get("sector_pct", {})
                if sector_pct:
                    sp_df = (
                        pd.Series(sector_pct, name="pct")
                        .reset_index()
                        .rename(columns={"index": "sector"})
                        .set_index("sector")
                    )
                    st.caption("Sector distribution (%) during events")
                    st.bar_chart(sp_df)

                # Sector deltas vs annual baseline
                sector_deltas: dict = ev.get("sector_deltas", {})
                if sector_deltas:
                    sd_df = (
                        pd.Series(sector_deltas, name="delta_pct")
                        .reset_index()
                        .rename(columns={"index": "sector"})
                    )
                    st.caption("Sector deltas vs annual baseline (positive = over-represented during events)")
                    st.dataframe(sd_df, use_container_width=True)
else:
    st.info(
        "No co-occurrence event data found. "
        "Re-run with `--wind-rose` and `--wind-event-thresholds` to enable event analysis."
    )
