"""Table rendering helpers for the Streamlit UI.

No weather math — display only.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from weather_tool.ui.components.formatting import SEV_ICON


def render_flags_table(df: pd.DataFrame, packets: dict) -> None:
    """Render a risk flags DataFrame with severity highlighting and evidence expanders.

    Parameters
    ----------
    df      : Output of backend.flags_table() — strictly scalar columns.
              Columns: station, flag_id, severity, confidence, evidence_summary,
                       freeze_hours_display, normalization_mode, notes.
    packets : Raw packets dict {station_id: packet} used for evidence drill-down.
              Pass {} when packets are unavailable (e.g., aggregated_flags fallback).
    """
    if df.empty:
        st.success("No risk flags raised.")
        return

    # Build display frame — drop the numeric freeze_hours_norm (internal use only)
    display_cols = [c for c in df.columns if c != "freeze_hours_norm"]
    display_df = df[display_cols].copy()

    # Add severity icon prefix to severity column
    if "severity" in display_df.columns:
        display_df["severity"] = display_df["severity"].apply(
            lambda s: f"{SEV_ICON.get(str(s).lower(), '')} {s}" if s else ""
        )

    def _highlight(row: pd.Series) -> list[str]:
        raw_sev = str(row.get("severity", "")).lower()
        # Icon prefix means "high" is now "🔴 high" etc. — strip the icon
        sev = raw_sev.split()[-1] if raw_sev else ""
        color = {
            "high": "background-color:#ffd7d7",
            "medium": "background-color:#fff3cd",
        }.get(sev, "")
        return [color] * len(row)

    try:
        st.dataframe(
            display_df.style.apply(_highlight, axis=1),
            use_container_width=True,
        )
    except Exception:
        st.dataframe(display_df, use_container_width=True)

    # Evidence drill-down expanders — only when raw packets are available
    if not packets:
        return

    for _, row in df.iterrows():
        sid = str(row.get("station", ""))
        flag_id = str(row.get("flag_id", ""))
        pkt = packets.get(sid, {})
        for flag in pkt.get("risk_flags", []):
            if flag.get("flag_id") == flag_id:
                evidence = flag.get("evidence") or []
                if evidence:
                    with st.expander(
                        f"Evidence — {sid} / {flag_id}", expanded=False
                    ):
                        try:
                            st.dataframe(
                                pd.DataFrame(evidence), use_container_width=True
                            )
                        except Exception as exc:
                            st.write(evidence)
                            st.caption(f"(could not render as table: {exc})")
                break
