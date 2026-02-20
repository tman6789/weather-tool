"""Chart components for the Streamlit UI.

All functions take pre-loaded DataFrames and render directly via st.*
No weather math — display only.
"""

from __future__ import annotations

import streamlit as st


def exceedance_hours_bar_chart(
    summary_data: "dict[str, object]",  # dict[str, pd.DataFrame | None]
    metric_axis: str,
    norm_mode: str,
) -> None:
    """Render a bar chart of exceedance hours per station.

    Parameters
    ----------
    summary_data : {station_id: summary_df} — per-year summary DataFrames.
    metric_axis  : "Tdb", "Twb", or "Both".
    norm_mode    : "per_year" or "aggregate".
    """
    import pandas as pd

    # Build candidate column lists based on metric_axis
    tdb_col = "exceedance_hours_tdb_p99"
    twb_col = "exceedance_hours_twb_p99"
    fallback_col = "hours_above_ref"

    frames: list[pd.DataFrame] = []
    used_fallback = False
    missing_reason: str | None = None

    for sid, df in summary_data.items():
        if df is None or df.empty:
            continue

        # Determine which columns to use
        cols: list[str] = []
        if metric_axis in ("Tdb", "Both") and tdb_col in df.columns:
            cols.append(tdb_col)
        if metric_axis in ("Twb", "Both") and twb_col in df.columns:
            cols.append(twb_col)

        if not cols:
            # Fallback to hours_above_ref if it exists
            if fallback_col in df.columns:
                cols = [fallback_col]
                used_fallback = True
            else:
                missing_reason = (
                    "No exceedance hour columns or hours_above_ref found. "
                    "Re-run with `tmpf` (and `dwpf`/`relh` for Twb) fields."
                )
                continue

        tmp = df[["year"] + cols].copy()
        tmp["station"] = sid
        frames.append(tmp)

    if not frames:
        reason = missing_reason or (
            f"No summary data available for metric axis '{metric_axis}'."
        )
        st.info(reason)
        return

    if used_fallback:
        st.caption(
            f"⚠️ `exceedance_hours_*` not available — showing `{fallback_col}` instead."
        )

    combined = pd.concat(frames, ignore_index=True)
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
            # Wide frame: index = year, columns = station_ids
            # st.bar_chart renders each column as a separate bar group
            pivot = combined.pivot_table(
                index="year", columns="station", values=col, aggfunc="first"
            )
            pivot.columns.name = None
            st.bar_chart(pivot)
        else:
            # Aggregate: sum over all years, one bar per station
            agg = combined.groupby("station")[col].sum().rename(col)
            st.bar_chart(agg.to_frame())
