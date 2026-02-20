"""UI service layer — orchestrates pipeline calls and artifact loading.

Pages import from here only. No page should import pipeline internals directly.

Public API
----------
build_run_config_from_sidebar(...)  -> RunConfig
run_pipeline_and_save(...)          -> str  (output directory path)
load_packets(run_dir)               -> dict[station_id, packet_dict]
load_exec_summaries(run_dir)        -> dict[station_id, md_str]
load_wind_artifacts(run_dir)        -> dict[station_id, artifact_handles]
load_compare_packet(run_dir)        -> dict | None
load_compare_summary_csv(run_dir)   -> pd.DataFrame | None
load_compare_metadata(run_dir)      -> dict | None
load_summary_csv(run_dir, sid)      -> pd.DataFrame | None
flags_table(packets, norm_mode)     -> pd.DataFrame
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st


# ── Private helpers ────────────────────────────────────────────────────────────


def _sid_from_packet_stem(stem: str) -> str:
    """station_packet_{station_id}_{start}_{end} → station_id (fallback parser)."""
    parts = stem.split("_")
    return parts[2] if len(parts) >= 3 else stem


# ── Config builder ─────────────────────────────────────────────────────────────


def build_run_config_from_sidebar(
    station_id: str,
    start: date,
    end: date,
    ref_temp: float,
    fields: list[str],
    profile: str,
    decision_ai: bool,
    wind_rose: bool,
    outdir: Path,
) -> Any:
    """Build and validate a RunConfig for a single station."""
    from weather_tool.config import RunConfig

    cfg = RunConfig(
        mode="iem",
        station_id=station_id.strip().upper(),
        start=start,
        end=end,
        ref_temp=ref_temp,
        fields=fields,
        decision_ai=decision_ai,
        decision_profile=profile,
        wind_rose=wind_rose,
        outdir=outdir,
    )
    cfg.validate()
    return cfg


# ── Pipeline runner (NOT cached — has file-write side effects) ─────────────────


def run_pipeline_and_save(
    station_ids: list[str],
    start: date,
    end: date,
    ref_temp: float,
    fields: list[str],
    profile: str,
    decision_ai: bool,
    wind_rose: bool,
    outdir: Path,
) -> str:
    """Run pipeline for one or more stations, save all artifacts, return output dir.

    For a single station, returns str(outdir).
    For multiple stations, runs a full compare and returns str(compare_dir).
    Mirrors cli.py behaviour exactly (parquet, insights MD, compare report, etc.).
    """
    import datetime as _dt

    import pandas as pd

    from weather_tool.insights.deterministic import generate_insights_md
    from weather_tool.pipeline import run_station_pipeline
    from weather_tool.storage.io import (
        save_exec_summary_md,
        save_insights_md,
        save_metadata_json,
        save_quality_json,
        save_raw_parquet,
        save_station_packet_json,
        save_summary_csv,
        save_wind_event_json,
        save_wind_rose_csv,
        save_wind_rose_png,
    )

    station_results = []
    for sid in station_ids:
        cfg = build_run_config_from_sidebar(
            sid, start, end, ref_temp, fields, profile, decision_ai, wind_rose, outdir
        )
        result = run_station_pipeline(cfg, echo=False)

        # ── Save all artifacts (CLI parity) ───────────────────────────────
        save_summary_csv(result.summary, cfg)
        save_raw_parquet(result.windowed, cfg)
        save_quality_json(result.quality_report, cfg)
        save_metadata_json(result.metadata, cfg)

        insights_md = generate_insights_md(result.summary, cfg, result.interval_info)
        save_insights_md(insights_md, cfg)

        if result.decision:
            save_station_packet_json(result.decision["packet"], cfg)
            save_exec_summary_md(result.decision["exec_summary_md"], cfg, llm=False)

        if result.wind_results:
            wind_dir = outdir / "wind"
            for slice_name, sdata in result.wind_results.get("slices", {}).items():
                save_wind_rose_csv(
                    sdata["rose_hours"], sdata["rose_meta"],
                    wind_dir, cfg.file_tag, slice_name,
                )
                save_wind_rose_png(
                    sdata["rose_hours"], sdata["rose_meta"],
                    cfg.wind_speed_bins, cfg.wind_speed_units,
                    wind_dir, cfg.file_tag, slice_name,
                    f"{sid} — {slice_name}",
                )
            if result.wind_results.get("events"):
                save_wind_event_json(result.wind_results["events"], wind_dir, cfg.file_tag)

        station_results.append(result)

    # ── Single-station: done ───────────────────────────────────────────────
    if len(station_ids) == 1:
        return str(outdir)

    # ── Multi-station compare: mirror cli.py compare exactly ──────────────
    from weather_tool.core.compare import build_compare_summary
    from weather_tool.insights.compare_report import generate_compare_report_md
    from weather_tool.storage.io import _compare_dir, save_compare_outputs

    compare_df = build_compare_summary(station_results, [ref_temp])

    report_md = generate_compare_report_md(
        compare_df=compare_df,
        stations=station_ids,
        window_start=start,
        window_end=end,
        fields=fields,
        ref_temps=[ref_temp],
    )

    yearly_concat = pd.concat(
        [r.summary.assign(station_id=r.cfg.station_id) for r in station_results],
        ignore_index=True,
    )

    metadata: dict[str, Any] = {
        "stations": station_ids,
        "window_start": str(start),
        "window_end": str(end),
        "fields": fields,
        "ref_temps": [ref_temp],
        "run_timestamp": str(_dt.datetime.now(tz=_dt.timezone.utc).isoformat()),
        "per_station_dt": {
            r.cfg.station_id: r.interval_info["dt_minutes"]
            for r in station_results
        },
    }

    cdir = _compare_dir(outdir, str(start), str(end), station_ids)
    save_compare_outputs(compare_df, station_results, yearly_concat, report_md, metadata, cdir)

    if decision_ai:
        from weather_tool.insights.exec_summary import render_exec_summary_compare
        from weather_tool.insights.packet import build_compare_packet
        from weather_tool.storage.io import (
            save_compare_exec_summary_md,
            save_compare_packet_json,
        )

        station_packets = [r.decision["packet"] for r in station_results if r.decision]
        if station_packets:
            compare_packet = build_compare_packet(
                cfg_start=str(start),
                cfg_end=str(end),
                compare_df=compare_df,
                station_packets=station_packets,
                stations=station_ids,
            )
            tag = cdir.name
            save_compare_packet_json(compare_packet, cdir, tag)
            save_compare_exec_summary_md(render_exec_summary_compare(compare_packet), cdir, tag)

    return str(cdir)


# ── Cached loaders (pure disk reads) ──────────────────────────────────────────
# All take run_dir: str (hashable for st.cache_data).
# Callers must call st.cache_data.clear() before updating session_state after
# a Run/Load click so stale data cannot be returned on a repeated run into the
# same folder.


@st.cache_data(show_spinner=False)
def load_packets(run_dir: str) -> dict[str, dict]:
    """Recursively find and load all station_packet_*.json under run_dir.

    Works for both single-station (outputs/) and compare runs where Decision AI
    was enabled (compare_dir/stations/{sid}/station_packet_*.json).
    Returns {} if none found (non-Decision-AI compare runs).
    """
    results: dict[str, dict] = {}
    for p in Path(run_dir).rglob("station_packet_*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sid = data.get("meta", {}).get("station_id") or _sid_from_packet_stem(p.stem)
            results[sid] = data
        except Exception:
            pass
    return results


@st.cache_data(show_spinner=False)
def load_exec_summaries(run_dir: str) -> dict[str, str]:
    """Find and load exec_summary_*.md (non-LLM). Infers station_id from filename.

    Filename pattern: exec_summary_{station_id}_{start}_{end}.md
    """
    results: dict[str, str] = {}
    for p in Path(run_dir).rglob("exec_summary_*.md"):
        if "llm" in p.stem:
            continue
        parts = p.stem.split("_")
        sid = parts[2] if len(parts) >= 3 else p.stem
        results[sid] = p.read_text(encoding="utf-8")
    return results


@st.cache_data(show_spinner=False)
def load_wind_artifacts(run_dir: str) -> dict[str, dict]:
    """Find all wind artifacts under run_dir.

    Returns:
        dict[station_id, {
            "slices": {slice_name: {"csv_path": str, "png_path": str | None}},
            "events_path": str | None,
        }]

    CSV stem pattern: wind_rose_{slice}_{station_id}_{start}_{end}
    """
    artifacts: dict[str, dict] = {}
    for csv_path in Path(run_dir).rglob("wind_rose_*.csv"):
        stem = csv_path.stem                     # wind_rose_{slice}_{tag}
        parts = stem.split("_", 3)               # ['wind', 'rose', slice, rest]
        if len(parts) < 4:
            continue
        slice_name = parts[2]
        sid = parts[3].split("_")[0]             # first token of {station_id}_{start}_{end}
        png = csv_path.with_suffix(".png")
        artifacts.setdefault(sid, {"slices": {}, "events_path": None})
        artifacts[sid]["slices"][slice_name] = {
            "csv_path": str(csv_path),
            "png_path": str(png) if png.exists() else None,
        }
    for ev in Path(run_dir).rglob("wind_events_*.json"):
        sid = ev.stem.replace("wind_events_", "").split("_")[0]
        artifacts.setdefault(sid, {"slices": {}, "events_path": None})
        artifacts[sid]["events_path"] = str(ev)
    return artifacts


@st.cache_data(show_spinner=False)
def load_compare_packet(run_dir: str) -> dict | None:
    """Load compare_packet_*.json if present. Returns None for non-Decision-AI runs."""
    for p in Path(run_dir).rglob("compare_packet_*.json"):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


@st.cache_data(show_spinner=False)
def load_compare_summary_csv(run_dir: str) -> Any:
    """Load compare_summary.csv if present (always written in compare runs)."""
    import pandas as pd

    p = Path(run_dir) / "compare_summary.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_compare_metadata(run_dir: str) -> dict | None:
    """Load compare_metadata.json if present (always written in compare runs)."""
    p = Path(run_dir) / "compare_metadata.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_summary_csv(run_dir: str, station_id: str) -> Any:
    """Find and load summary_{station_id}_*.csv under run_dir.

    Returns pd.DataFrame or None if not found.
    """
    import pandas as pd

    for p in Path(run_dir).rglob(f"summary_{station_id}_*.csv"):
        return pd.read_csv(p)
    return None


def flags_table(
    packets: dict,
    normalization_mode: str = "per_year",
) -> Any:
    """Build a strictly-scalar DataFrame of risk flags suitable for st.dataframe.

    The raw ``evidence`` field in each flag is ``list[dict]`` which Streamlit
    renders as ``[object Object]``.  This function flattens it to a plain string
    and adds explicit freeze normalization columns so callers can sort/filter
    numerically without rendering issues.

    Parameters
    ----------
    packets          : {station_id: packet_dict} from load_packets().
    normalization_mode : "per_year" | "aggregate".
                         Controls how freeze_hours_norm / freeze_hours_display
                         are calculated for freeze-related flags only.

    Returns
    -------
    pd.DataFrame with columns:
        station, flag_id, severity, confidence, evidence_summary,
        freeze_hours_norm (float | None), freeze_hours_display (str),
        normalization_mode (str), notes.
    """
    import pandas as pd

    from weather_tool.ui.components.formatting import format_freeze_hours, safe_fmt_float

    rows = []
    for sid, pkt in packets.items():
        years_covered: int = pkt.get("meta", {}).get("years_covered", 0) or 0
        freeze_sum = pkt.get("freeze_risk", {}).get("freeze_hours_sum")

        for flag in pkt.get("risk_flags", []):
            evidence = flag.get("evidence") or []
            ev_parts: list[str] = []
            for ev in evidence[:3]:
                if not isinstance(ev, dict):
                    continue
                m = ev.get("metric", "")
                v = ev.get("value")
                t = ev.get("threshold")
                part = f"{m} = {safe_fmt_float(v)}"
                if t is not None:
                    part += f" (thr: {safe_fmt_float(t)})"
                ev_parts.append(part)

            flag_id: str = flag.get("flag_id", "")
            is_freeze = flag_id.lower().startswith("freeze")

            freeze_norm: float | None = None
            freeze_display = ""
            if is_freeze and freeze_sum is not None:
                if normalization_mode == "per_year" and years_covered > 0:
                    freeze_norm = freeze_sum / years_covered
                else:
                    freeze_norm = freeze_sum
                freeze_display = format_freeze_hours(
                    freeze_sum, years_covered, normalization_mode
                )

            rows.append(
                {
                    "station": sid,
                    "flag_id": flag_id,
                    "severity": flag.get("severity", ""),
                    "confidence": flag.get("confidence", ""),
                    "evidence_summary": "; ".join(ev_parts),
                    "freeze_hours_norm": freeze_norm,
                    "freeze_hours_display": freeze_display,
                    "normalization_mode": normalization_mode if is_freeze else "",
                    "notes": flag.get("notes", ""),
                }
            )

    return pd.DataFrame(rows) if rows else pd.DataFrame()
