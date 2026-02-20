"""Streamlit entrypoint for weather-tool dashboard.

Run with:
    streamlit run weather_tool/ui/app.py

All pipeline configuration and Run/Load controls are on the ⚙️ Run Setup page.
This home page and the sidebar show the current run status only.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="weather-tool",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar — run info only ────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌤️ weather-tool")
    st.caption("Deterministic weather analysis dashboard")
    st.divider()

    if "run_dir" in st.session_state:
        run_dir = st.session_state["run_dir"]
        sids_display = st.session_state.get("station_ids", [])
        meta = st.session_state.get("run_meta", {})

        st.caption("**Active run**")
        st.caption(f"📂 `{Path(run_dir).name}`")
        if sids_display:
            st.caption(f"Stations: {', '.join(sids_display)}")
        if meta.get("start"):
            st.caption(
                f"{meta['start']} → {meta['end']} | "
                f"profile: {meta.get('profile', '—')}"
            )
    else:
        st.info("No run loaded. Go to **⚙️ Run Setup** to start.")

# ── Main area (home page) ──────────────────────────────────────────────────────

st.title("weather-tool Dashboard")

if "run_dir" not in st.session_state:
    st.info("Go to **⚙️ Run Setup** to run the pipeline or load an existing folder.")
    st.markdown("""
### Quick start

```bash
# Load the sample KORD run (no network needed)
# → Run Setup → Load existing folder → outputs/
```

**Pages:**
- **⚙️ Run Setup** — run the pipeline or load an existing output folder
- **📊 Compare** — station rankings, risk flags, worst Death Day windows
- **🏭 Station Detail** — exec summary, design conditions, risk flags, Death Day
- **🌬️ Wind** — wind rose images, co-occurrence event tables
- **⬇️ Exports** — download station packet JSON, exec summary, or ZIP bundle
""")
else:
    run_dir = st.session_state["run_dir"]
    sids = st.session_state.get("station_ids", [])
    is_compare = st.session_state.get("is_compare", False)

    cols = st.columns(3)
    cols[0].metric("Stations loaded", len(sids))
    cols[1].metric("Run folder", Path(run_dir).name)
    cols[2].metric("Mode", "Compare" if is_compare else "Single station")

    st.caption("Navigate using the page list on the left.")
