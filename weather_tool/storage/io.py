"""File I/O — export summary CSV, clean parquet, quality JSON, metadata JSON, insights MD."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from weather_tool.config import RunConfig

if TYPE_CHECKING:
    from weather_tool.pipeline import StationResult


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_summary_csv(summary: pd.DataFrame, cfg: RunConfig) -> Path:
    _ensure_dir(cfg.outdir)
    p = cfg.outdir / f"summary_{cfg.file_tag}.csv"
    summary.to_csv(p, index=False)
    return p


def save_raw_parquet(df: pd.DataFrame, cfg: RunConfig) -> Path:
    _ensure_dir(cfg.outdir)
    p = cfg.outdir / f"raw_clean_{cfg.file_tag}.parquet"
    # Drop internal columns before saving
    export = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    export.to_parquet(p, index=False, engine="pyarrow")
    return p


def save_quality_json(quality_data: dict[str, Any], cfg: RunConfig) -> Path:
    _ensure_dir(cfg.outdir)
    p = cfg.outdir / f"quality_report_{cfg.file_tag}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(quality_data, f, indent=2, default=str)
    return p


def save_metadata_json(metadata: dict[str, Any], cfg: RunConfig) -> Path:
    _ensure_dir(cfg.outdir)
    p = cfg.outdir / f"metadata_{cfg.file_tag}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return p


def save_insights_md(content: str, cfg: RunConfig) -> Path:
    _ensure_dir(cfg.outdir)
    p = cfg.outdir / f"insights_{cfg.file_tag}.md"
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return p


# ── Compare output helpers ────────────────────────────────────────────────────


def _compare_dir(outdir: Path, start: str, end: str, stations: list[str]) -> Path:
    """Return (and create) the compare output directory."""
    n = len(stations)
    tag = f"compare_{start}_{end}_{n}stations"
    p = outdir / tag
    _ensure_dir(p)
    return p


def save_compare_outputs(
    compare_df: pd.DataFrame,
    station_results: list[StationResult],
    yearly_concat: pd.DataFrame,
    report_md: str,
    metadata: dict[str, Any],
    compare_dir: Path,
) -> dict[str, Path]:
    """Write all compare outputs into *compare_dir* and per-station subfolders.

    Returns a dict mapping output type → Path.
    """
    paths: dict[str, Path] = {}

    # compare_summary.csv
    p = compare_dir / "compare_summary.csv"
    compare_df.to_csv(p, index=False)
    paths["summary"] = p

    # compare_yearly.parquet
    p2 = compare_dir / "compare_yearly.parquet"
    yearly_export = yearly_concat.drop(
        columns=[c for c in yearly_concat.columns if c.startswith("_")], errors="ignore"
    )
    yearly_export.to_parquet(p2, index=False, engine="pyarrow")
    paths["yearly"] = p2

    # compare_report.md
    p3 = compare_dir / "compare_report.md"
    p3.write_text(report_md, encoding="utf-8")
    paths["report"] = p3

    # compare_metadata.json
    p4 = compare_dir / "compare_metadata.json"
    with open(p4, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    paths["metadata"] = p4

    # Per-station subfolders — summary CSV + quality + metadata JSONs
    stations_dir = compare_dir / "stations"
    for r in station_results:
        sid = r.cfg.station_id or "csv"
        sdir = stations_dir / sid
        _ensure_dir(sdir)
        sid_cfg = dataclasses.replace(r.cfg, outdir=sdir)
        save_summary_csv(r.summary, sid_cfg)
        save_quality_json(r.quality_report, sid_cfg)
        save_metadata_json(r.metadata, sid_cfg)

    return paths


# ── Wind output helpers ──────────────────────────────────────────────────────


def save_wind_rose_csv(
    rose_hours: pd.DataFrame,
    rose_meta: dict[str, Any],
    outdir: Path,
    file_tag: str,
    slice_name: str,
) -> Path:
    """Save wind rose hours matrix as CSV with metadata header comments."""
    _ensure_dir(outdir)
    p = outdir / f"wind_rose_{slice_name}_{file_tag}.csv"
    with open(p, "w", encoding="utf-8") as f:
        for k, v in rose_meta.items():
            f.write(f"# {k}: {v}\n")
        rose_hours.to_csv(f)
    return p


def save_wind_rose_png(
    rose_hours: pd.DataFrame,
    rose_meta: dict[str, Any],
    speed_edges: list[float],
    speed_units: str,
    outdir: Path,
    file_tag: str,
    slice_name: str,
    title: str,
) -> Path | None:
    """Save wind rose PNG. Returns None if matplotlib is not installed."""
    _ensure_dir(outdir)
    p = outdir / f"wind_rose_{slice_name}_{file_tag}.png"
    try:
        from weather_tool.core.wind_plot import plot_wind_rose
        plot_wind_rose(rose_hours, rose_meta, speed_edges, speed_units, title, p)
        return p
    except ImportError:
        return None


def save_wind_event_json(
    event_stats: dict[str, Any],
    outdir: Path,
    file_tag: str,
) -> Path:
    """Save wind co-occurrence event stats as JSON."""
    _ensure_dir(outdir)
    p = outdir / f"wind_events_{file_tag}.json"
    # Filter out non-serializable items (DataFrames, Series)
    clean = {}
    for k, v in event_stats.items():
        if isinstance(v, (pd.DataFrame, pd.Series)):
            continue
        clean[k] = v
    with open(p, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, default=str)
    return p
