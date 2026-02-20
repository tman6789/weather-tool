"""Exports page — download station packets, exec summaries, and ZIP bundles.

Reads from session_state["run_dir"]. All download buttons use in-memory buffers;
no extra files are written.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import streamlit as st

from weather_tool.ui import backend

st.title("⬇️ Exports")

# ── Guard ──────────────────────────────────────────────────────────────────────

if "run_dir" not in st.session_state:
    st.info("No run loaded. Use the sidebar to run the pipeline or load an existing folder.")
    st.stop()

run_dir: str = st.session_state["run_dir"]
station_ids: list[str] = st.session_state.get("station_ids", [])

# ── Load artifacts (all cached) ────────────────────────────────────────────────

packets = backend.load_packets(run_dir)
exec_summaries = backend.load_exec_summaries(run_dir)

# ── Per-station downloads ──────────────────────────────────────────────────────

st.header("Per-Station Downloads")

if not station_ids:
    st.warning("No stations loaded.")
else:
    if len(station_ids) == 1:
        selected = station_ids[0]
    else:
        selected = st.selectbox("Station", station_ids, key="export_station_selector")

    pkt = packets.get(selected)
    md = exec_summaries.get(selected, "")

    col1, col2 = st.columns(2)

    with col1:
        if pkt:
            pkt_json = json.dumps(pkt, indent=2, default=str)
            st.download_button(
                label=f"📄 {selected} — Station Packet JSON",
                data=pkt_json,
                file_name=f"station_packet_{selected}.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("No station packet. Re-run with **--decision-ai**.")

    with col2:
        if md:
            st.download_button(
                label=f"📝 {selected} — Exec Summary (Markdown)",
                data=md,
                file_name=f"exec_summary_{selected}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.info("No exec summary. Re-run with **--decision-ai**.")

st.divider()

# ── Compare packet download ────────────────────────────────────────────────────

compare_pkt = backend.load_compare_packet(run_dir)
if compare_pkt:
    st.header("Compare Packet")
    st.download_button(
        label="📊 Compare Packet JSON",
        data=json.dumps(compare_pkt, indent=2, default=str),
        file_name="compare_packet.json",
        mime="application/json",
        use_container_width=False,
    )
    st.divider()

# ── ZIP bundle ────────────────────────────────────────────────────────────────

st.header("Download Run Folder Bundle (ZIP)")

st.caption(
    "Bundles all JSON, Markdown, CSV, and PNG files from the run folder. "
    "Large Parquet files are excluded to keep the download size reasonable."
)

_INCLUDE_SUFFIXES = {".json", ".md", ".csv", ".png"}

if st.button("🗜️ Build ZIP", use_container_width=False):
    rd = Path(run_dir)
    files_to_zip = [f for f in rd.rglob("*") if f.is_file() and f.suffix in _INCLUDE_SUFFIXES]

    if not files_to_zip:
        st.warning("No eligible files found to bundle.")
    else:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files_to_zip:
                # Archive path relative to the parent of run_dir so the folder
                # name is preserved inside the ZIP (e.g., outputs/summary_KORD_....csv)
                try:
                    arc_name = f.relative_to(rd.parent)
                except ValueError:
                    arc_name = f.name
                zf.write(f, arc_name)
        buf.seek(0)

        zip_name = f"{rd.name}.zip"
        st.download_button(
            label=f"⬇️ Download {zip_name} ({len(files_to_zip)} files)",
            data=buf,
            file_name=zip_name,
            mime="application/zip",
            use_container_width=False,
        )
        st.caption(f"Included {len(files_to_zip)} files (JSON/MD/CSV/PNG). Parquet excluded.")
