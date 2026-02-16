"""Single-station analysis pipeline — reusable core, no file I/O."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from weather_tool.config import IEM_UNITS, RunConfig


@dataclass
class StationResult:
    """All pipeline outputs for one station, without any file I/O."""

    summary: pd.DataFrame       # yearly summary (one row per year)
    windowed: pd.DataFrame      # normalized, window-filtered time series
    interval_info: dict[str, Any]
    quality_report: dict[str, Any]
    metadata: dict[str, Any]
    cfg: RunConfig


def run_station_pipeline(cfg: RunConfig, echo: bool = False) -> StationResult:
    """Run the full single-station pipeline and return a StationResult.

    Parameters
    ----------
    cfg : validated RunConfig
    echo : if True, emit progress lines via typer.echo

    Returns
    -------
    StationResult — no files are written.
    """
    from weather_tool.connectors.csv_connector import load_csv
    from weather_tool.connectors.iem_connector import load_iem
    from weather_tool.core.aggregate import build_yearly_summary
    from weather_tool.core.metrics import compute_wetbulb_f, infer_interval
    from weather_tool.core.normalize import deduplicated, filter_window, normalize_timestamps

    def _echo(msg: str) -> None:
        if echo:
            import typer
            typer.echo(msg)

    # 1. Load raw data
    _echo(f"  Loading data ({cfg.mode} mode)...")
    if cfg.mode == "csv":
        raw = load_csv(cfg)
    else:
        raw = load_iem(cfg)
    _echo(f"  Loaded {len(raw)} raw records.")

    # 2. Normalize + window
    _echo("  Normalising timestamps & filtering window...")
    normed = normalize_timestamps(raw, tz=cfg.tz)
    windowed = filter_window(normed, cfg.start, cfg.end, tz=cfg.tz)
    _echo(f"  {len(windowed)} records in analysis window.")

    # Compute wet-bulb if the required columns are present
    if any(c in windowed.columns for c in ("tmpf", "relh", "dwpf")):
        windowed = windowed.copy()
        windowed["wetbulb_f"] = compute_wetbulb_f(windowed)
        wb_avail = windowed["wetbulb_f"].notna().sum()
        _echo(f"  Wet-bulb computed: {wb_avail} non-NaN values.")

    # 3. Interval inference (on deduplicated data)
    _echo("  Inferring sampling interval...")
    dedup = deduplicated(windowed)
    interval_info = infer_interval(dedup["timestamp"])
    _echo(f"  dt_minutes = {interval_info['dt_minutes']}")
    if interval_info["interval_change_flag"]:
        _echo("  WARNING: interval changes detected in data.")

    # 4. Yearly summary
    _echo("  Computing yearly summary...")
    summary = build_yearly_summary(windowed, cfg, interval_info)
    _echo(f"  {len(summary)} year(s) in summary.")

    # 5. Build quality report + metadata
    _base_quality_cols = [
        "year", "missing_pct", "duplicate_count", "nan_temp_count",
        "interval_change_flag", "interval_unknown_flag", "partial_coverage_flag", "coverage_pct",
    ]
    _extra_quality_cols = [
        c for c in summary.columns
        if c.startswith("nan_count_") or c.startswith("field_missing_pct_")
        or c == "wetbulb_availability_pct"
    ]
    _quality_cols = [c for c in _base_quality_cols + _extra_quality_cols if c in summary.columns]

    quality_report: dict[str, Any] = {
        "station_id": cfg.station_id or "csv",
        "window": f"{cfg.start} to {cfg.end}",
        "total_raw_records": len(raw),
        "total_windowed_records": len(windowed),
        "interval": interval_info,
        "per_year": summary[_quality_cols].to_dict(orient="records"),
    }

    _units_meta: dict[str, Any] = {"temp": str(cfg.units)}
    if cfg.mode == "iem":
        _units_meta = {k: v for k, v in IEM_UNITS.items() if k in (cfg.fields + ["wetbulb_f"])}

    metadata: dict[str, Any] = {
        "station_id": cfg.station_id or "csv",
        "source": cfg.mode,
        "fields": cfg.fields,
        "units": _units_meta,
        "ref_temp": cfg.ref_temp,
        "tz": cfg.tz,
        "window_start": str(cfg.start),
        "window_end": str(cfg.end),
        "dt_inference": {
            "dt_minutes": interval_info["dt_minutes"],
            "p10": interval_info["p10"],
            "p90": interval_info["p90"],
            "interval_change_flag": interval_info["interval_change_flag"],
            "unique_diff_counts": interval_info["unique_diff_counts"],
        },
    }

    return StationResult(
        summary=summary,
        windowed=windowed,
        interval_info=interval_info,
        quality_report=quality_report,
        metadata=metadata,
        cfg=cfg,
    )
