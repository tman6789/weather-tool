"""Deterministic Executive Engineering Summary renderer.

Converts station and compare decision packets into human-readable markdown.
All logic is pure (no file I/O, no network calls, no LLM).

Section layout — station packet
--------------------------------
1. Executive Engineering Summary (3–6 bullets from risk_flags)
2. Design Stress Profile (design_conditions table + death day candidates)
3. Efficiency Outlook (operational_efficiency table)
4. Infrastructure Resilience (highest-severity flag + freeze risk summary)
5. Risk Mitigations (numbered list from recommendations)

Section layout — compare packet
--------------------------------
1. Multi-Station Summary (cross-station extremes table)
2. Station Rankings (all 5 score columns)
3. Per-Station Flags (one line per high-severity flag per station)
4. Cross-Station Efficiency Comparison
5. Deduplicated Recommendations
"""

from __future__ import annotations

import math
from typing import Any


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _val(v: Any, fmt: str = ".1f", suffix: str = "") -> str:
    """Format a single metric value; return 'N/A' when absent or NaN."""
    if v is None:
        return "N/A"
    try:
        if math.isnan(float(v)):
            return "N/A"
    except (TypeError, ValueError):
        return str(v)
    try:
        return f"{float(v):{fmt}}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def _row(label: str, value: str) -> str:
    return f"| {label} | {value} |"


def _table(rows: list[tuple[str, str]]) -> str:
    lines = ["| Metric | Value |", "| --- | --- |"]
    lines.extend(_row(label, val) for label, val in rows)
    return "\n".join(lines)


# ── Station exec summary ───────────────────────────────────────────────────────

def render_exec_summary_station(packet: dict[str, Any]) -> str:
    """Render a deterministic markdown executive summary for a single station packet."""
    meta = packet.get("meta", {})
    dc   = packet.get("design_conditions", {})
    oe   = packet.get("operational_efficiency", {})
    fr   = packet.get("freeze_risk", {})
    dd   = packet.get("death_day", {})
    flags = packet.get("risk_flags", [])
    recs  = packet.get("recommendations", [])

    station_id = meta.get("station_id", "Unknown")
    profile    = meta.get("profile", "datacenter")
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(f"# Executive Engineering Summary — {station_id}")
    lines.append(f"**Profile:** {profile}  |  "
                 f"**Window:** {meta.get('window_start', 'N/A')} → {meta.get('window_end', 'N/A')}  |  "
                 f"**Years covered:** {meta.get('years_covered', 'N/A')}")
    lines.append("")

    # ── Section 1: Executive bullet summary ───────────────────────────────────
    lines.append("## 1. Executive Engineering Summary")
    high_flags  = [f for f in flags if f.get("severity") == "high"]
    medium_flags = [f for f in flags if f.get("severity") == "medium"]
    bullet_flags = high_flags + (medium_flags if len(high_flags) < 3 else [])
    bullet_flags = bullet_flags[:6]

    if bullet_flags:
        for f in bullet_flags:
            sev   = f.get("severity", "").upper()
            fid   = f.get("flag_id", "")
            notes = f.get("notes", "")
            ev    = f.get("evidence", [{}])
            ev0   = ev[0] if ev else {}
            metric_str = ""
            if ev0:
                metric_str = (
                    f" [{ev0.get('metric', '')} = "
                    f"{_val(ev0.get('value'), '.1f')} "
                    f"vs threshold {_val(ev0.get('threshold'), '.1f')}]"
                )
            lines.append(f"- **[{sev}] {fid}**{metric_str}: {notes}")
    else:
        lines.append("- No critical flags detected for this station.")
    lines.append("")

    # ── Section 2: Design Stress Profile ──────────────────────────────────────
    lines.append("## 2. Design Stress Profile")
    design_rows = [
        ("TDB p99 (°F)",          _val(dc.get("tdb_p99"))),
        ("TDB p99.6 (°F)",        _val(dc.get("tdb_p996"))),
        ("TDB max observed (°F)", _val(dc.get("tdb_max"))),
        ("WB p99 (°F)",           _val(dc.get("wb_p99"))),
        ("WB p99.6 (°F)",         _val(dc.get("wb_p996"))),
        ("WB max observed (°F)",  _val(dc.get("wb_max"))),
        ("WB mean (°F)",          _val(dc.get("wb_mean"))),
        ("TDB 24h rolling max (°F)", _val(dc.get("tdb_mean_24h_max"))),
        ("TDB 72h rolling max (°F)", _val(dc.get("tdb_mean_72h_max"))),
        ("WB 24h rolling max (°F)", _val(dc.get("wb_mean_24h_max"))),
        ("WB 72h rolling max (°F)", _val(dc.get("wb_mean_72h_max"))),
        ("LWT proxy p99 (°F)",    _val(dc.get("lwt_proxy_p99"))),
        ("Baseline years",        _val(dc.get("baseline_years_count"), ".0f")),
        ("Baseline method",       str(dc.get("baseline_method", "N/A"))),
    ]
    lines.append(_table(design_rows))
    lines.append("")

    # Death Day candidates
    candidates = dd.get("candidates", [])
    dd_mode    = dd.get("mode", "heat_day")
    lines.append(f"**Death Day Candidates** (mode: `{dd_mode}`, window: {dd.get('window_hours', 24)}h)")
    if dd_mode == "heat_day":
        lines.append("> *Wet-bulb data unavailable — candidates ranked by dry-bulb stress only "
                     "(low confidence).*")
    if candidates:
        cand_header = "| Rank | Start | End | Stress | z_TDB | z_WB | TDB mean | WB mean |"
        cand_sep    = "| --- | --- | --- | --- | --- | --- | --- | --- |"
        lines.append(cand_header)
        lines.append(cand_sep)
        for c in candidates:
            lines.append(
                f"| {c.get('rank', '?')} "
                f"| {c.get('start_ts', 'N/A')} "
                f"| {c.get('end_ts', 'N/A')} "
                f"| {_val(c.get('stress_score'))} "
                f"| {_val(c.get('z_tdb'))} "
                f"| {_val(c.get('z_wb'))} "
                f"| {_val(c.get('tdb_mean_f'))} "
                f"| {_val(c.get('twb_mean_f'))} |"
            )
    else:
        lines.append("*No candidates identified.*")
    lines.append("")

    # ── Section 3: Efficiency Outlook ─────────────────────────────────────────
    lines.append("## 3. Efficiency Outlook")
    eff_rows = [
        ("Air-side econ hours (sum)",       _val(oe.get("air_econ_hours_sum"), ".0f")),
        ("WEC feasible hours (sum)",        _val(oe.get("wec_hours_sum"), ".0f")),
        ("WEC feasible % over window",      _val(oe.get("wec_feasible_pct_over_window"), ".1%")),
        ("Tower stress hrs WB>75°F (sum)",  _val(oe.get("tower_stress_hours_wb_gt_75_sum"), ".0f")),
        ("Tower stress hrs WB>78°F (sum)",  _val(oe.get("tower_stress_hours_wb_gt_78_sum"), ".0f")),
        ("Tower stress hrs WB>80°F (sum)",  _val(oe.get("tower_stress_hours_wb_gt_80_sum"), ".0f")),
    ]
    lines.append(_table(eff_rows))
    lines.append("")

    # ── Section 4: Infrastructure Resilience ──────────────────────────────────
    lines.append("## 4. Infrastructure Resilience")
    if high_flags:
        worst = high_flags[0]
        ev    = worst.get("evidence", [{}])
        ev0   = ev[0] if ev else {}
        lines.append(f"**Highest-priority flag:** `{worst.get('flag_id', '')}` "
                     f"(severity: {worst.get('severity', '')})")
        if ev0:
            lines.append(f"- Metric `{ev0.get('metric', '')}` = "
                         f"{_val(ev0.get('value'), '.2f')} "
                         f"(threshold: {_val(ev0.get('threshold'), '.2f')})")
        if worst.get("notes"):
            lines.append(f"- {worst['notes']}")
    else:
        lines.append("No high-severity flags detected.")
    lines.append("")

    freeze_rows = [
        ("Freeze hours (sum)",                _val(fr.get("freeze_hours_sum"), ".0f")),
        ("Shoulder freeze hours (sum)",        _val(fr.get("freeze_hours_shoulder_sum"), ".0f")),
        ("Freeze events (count)",              _val(fr.get("freeze_event_count_sum"), ".0f")),
        ("Longest freeze event (h)",           _val(fr.get("freeze_event_max_duration_hours_max"), ".1f")),
        ("Record low temp (°F)",               _val(fr.get("tmin_min"), ".1f")),
    ]
    lines.append("**Freeze Risk Summary**")
    lines.append(_table(freeze_rows))
    lines.append("")

    # ── Section 5: Risk Mitigations ───────────────────────────────────────────
    lines.append("## 5. Risk Mitigations")
    if recs:
        for i, r in enumerate(recs, 1):
            title    = r.get("title", "")
            severity = r.get("severity", "")
            rationale = r.get("rationale", "")
            evidence  = r.get("evidence", [])
            lines.append(f"{i}. **{title}** *({severity})*")
            if rationale:
                lines.append(f"   {rationale}")
            for ev in evidence:
                lines.append(
                    f"   Evidence: `{ev.get('metric', '')}` = "
                    f"{_val(ev.get('value'), '.2f')} "
                    f"(threshold: {_val(ev.get('threshold'), '.2f')})"
                )
    else:
        lines.append("No specific mitigations required at this time.")
    lines.append("")

    return "\n".join(lines)


# ── Compare exec summary ───────────────────────────────────────────────────────

def render_exec_summary_compare(packet: dict[str, Any]) -> str:
    """Render a deterministic markdown executive summary for a compare packet."""
    meta       = packet.get("meta", {})
    summaries  = packet.get("station_summaries", [])
    rankings   = packet.get("rankings", {})
    cse        = packet.get("cross_station_extremes", {})
    agg_flags  = packet.get("aggregated_flags", [])

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    stations_str = ", ".join(str(s) for s in meta.get("stations", []))
    lines.append("# Executive Engineering Summary — Multi-Station Comparison")
    lines.append(f"**Stations:** {stations_str}  |  "
                 f"**Window:** {meta.get('window_start', 'N/A')} → {meta.get('window_end', 'N/A')}")
    lines.append("")

    # ── Section 1: Cross-station extremes ─────────────────────────────────────
    lines.append("## 1. Multi-Station Summary")
    def _extreme_str(d: dict | None) -> str:
        if not d:
            return "N/A"
        return f"{d.get('station_id', '?')} ({_val(d.get('value'))})"

    extreme_rows = [
        ("Hottest station (TDB max)",    _extreme_str(cse.get("hottest_station"))),
        ("Coldest station (Tmin min)",   _extreme_str(cse.get("coldest_station"))),
        ("Highest WB p99.6",             _extreme_str(cse.get("highest_wb_p996"))),
        ("Most air econ hours",          _extreme_str(cse.get("most_econ_hours"))),
        ("Most freeze hours",            _extreme_str(cse.get("most_freeze_hours"))),
    ]
    lines.append(_table(extreme_rows))
    lines.append("")

    # ── Section 2: Rankings ───────────────────────────────────────────────────
    lines.append("## 2. Station Rankings")
    score_cols = ["overall_score", "heat_score", "moisture_score", "freeze_score", "data_quality_score"]
    # Build header from all stations in overall_score ranking
    os_ranking = rankings.get("overall_score", [])
    ranked_ids = [r["station_id"] for r in os_ranking]
    if ranked_ids:
        hdr_cells  = " | ".join(f"**{sid}**" for sid in ranked_ids)
        rank_header = f"| Score | {hdr_cells} |"
        rank_sep    = "| --- |" + " --- |" * len(ranked_ids)
        lines.append(rank_header)
        lines.append(rank_sep)
        for col in score_cols:
            col_ranking = rankings.get(col, [])
            # Build a map: station_id → value
            col_map = {r["station_id"]: r["value"] for r in col_ranking}
            cells = " | ".join(_val(col_map.get(sid)) for sid in ranked_ids)
            lines.append(f"| {col} | {cells} |")
    else:
        lines.append("*No ranking data available.*")
    lines.append("")

    # ── Section 3: Per-station flags ──────────────────────────────────────────
    lines.append("## 3. Per-Station High-Severity Flags")
    if agg_flags:
        for f in agg_flags:
            sid   = f.get("station_id", "?")
            fid   = f.get("flag_id", "")
            notes = f.get("notes", "")
            ev    = f.get("evidence", [{}])
            ev0   = ev[0] if ev else {}
            metric_str = ""
            if ev0:
                metric_str = (
                    f" [{ev0.get('metric', '')} = {_val(ev0.get('value'), '.1f')} "
                    f"vs {_val(ev0.get('threshold'), '.1f')}]"
                )
            lines.append(f"- **{sid}** `{fid}`{metric_str}: {notes}")
    else:
        lines.append("*No high-severity flags across any station.*")
    lines.append("")

    # ── Section 4: Cross-station efficiency ───────────────────────────────────
    lines.append("## 4. Cross-Station Efficiency Comparison")
    if summaries:
        eff_header = "| Station | TDB p99.6 | WB p99.6 | Air Econ Hrs | Freeze Hrs |"
        eff_sep    = "| --- | --- | --- | --- | --- |"
        lines.append(eff_header)
        lines.append(eff_sep)
        for ss in summaries:
            lines.append(
                f"| {ss.get('station_id', '?')} "
                f"| {_val(ss.get('tdb_p996'))} "
                f"| {_val(ss.get('wb_p996'))} "
                f"| {_val(ss.get('air_econ_hours_sum'), '.0f')} "
                f"| {_val(ss.get('freeze_hours_sum'), '.0f')} |"
            )
    else:
        lines.append("*No station summaries available.*")
    lines.append("")

    # ── Section 5: Cross-Station Recommendations ──────────────────────────────
    lines.append("## 5. Cross-Station Recommendations")
    station_pkts = packet.get("station_packets")  # dict sid→packet (full mode) or None (lean)
    seen_rec_ids: set[str] = set()
    rendered_recs: list[dict] = []

    if station_pkts:
        # Full mode: collect and deduplicate actual Recommendation dicts by rec_id
        for _sid, spkt in station_pkts.items():
            for r in spkt.get("recommendations", []):
                rid = r.get("rec_id", "")
                if rid not in seen_rec_ids:
                    seen_rec_ids.add(rid)
                    rendered_recs.append(r)

    if rendered_recs:
        lines.append("Prioritised actions (deduplicated across stations):")
        for i, r in enumerate(rendered_recs, 1):
            lines.append(f"{i}. **{r.get('title', '')}** *({r.get('severity', '')})*")
            if r.get("rationale"):
                lines.append(f"   {r['rationale']}")
            for ev in r.get("evidence", []):
                lines.append(
                    f"   Evidence: `{ev.get('metric', '')}` = "
                    f"{_val(ev.get('value'), '.2f')} "
                    f"(threshold: {_val(ev.get('threshold'), '.2f')})"
                )
    elif agg_flags:
        lines.append("Address the following high-severity flags across stations:")
        for f in agg_flags:
            fid = f.get("flag_id", "")
            sid = f.get("station_id", "?")
            if fid not in seen_rec_ids:
                seen_rec_ids.add(fid)
                notes = f.get("notes", "")
                lines.append(f"- **{fid}** (first seen at {sid}): {notes}")
    else:
        lines.append("*No cross-station mitigations required at this time.*")
    lines.append("")

    return "\n".join(lines)
