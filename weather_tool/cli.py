"""Typer CLI entry-point for weather-tool."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import typer

from weather_tool.config import Mode, RunConfig, Units

app = typer.Typer(name="weather-tool", help="Deterministic historical weather analysis tool.")


@app.command()
def run(
    mode: Mode = typer.Option("csv", help="Data source mode: csv or iem"),
    input: Optional[Path] = typer.Option(None, "--input", help="Path to CSV file (csv mode)"),
    timestamp_col: str = typer.Option("timestamp", help="Name of the timestamp column"),
    temp_col: str = typer.Option("temp", help="Name of the temperature column"),
    station_id: Optional[str] = typer.Option(None, "--station-id", help="Station identifier"),
    start: str = typer.Option(..., help="Analysis start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="Analysis end date YYYY-MM-DD"),
    ref_temp: float = typer.Option(..., "--ref-temp", help="Reference temperature threshold"),
    units: Units = typer.Option("agnostic", help="Units: agnostic, F, C, K, or auto"),
    tz: str = typer.Option("UTC", "--tz", help="IANA timezone for timestamps"),
    outdir: Path = typer.Option(Path("outputs"), help="Output directory"),
    llm: bool = typer.Option(False, "--llm", help="Generate optional LLM narrative"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Run a weather analysis pipeline."""
    cfg = RunConfig(
        mode=mode,
        input_path=input,
        timestamp_col=timestamp_col,
        temp_col=temp_col,
        station_id=station_id,
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        ref_temp=ref_temp,
        units=units,
        tz=tz,
        outdir=outdir,
        use_llm=llm,
        verbose=verbose,
    )
    cfg.validate()

    _execute(cfg)


def _execute(cfg: RunConfig) -> None:
    """Orchestrate the full pipeline."""
    import pandas as pd

    from weather_tool.connectors.csv_connector import load_csv
    from weather_tool.connectors.iem_connector import load_iem
    from weather_tool.core.aggregate import build_yearly_summary
    from weather_tool.core.metrics import infer_interval
    from weather_tool.core.normalize import deduplicated, filter_window, normalize_timestamps
    from weather_tool.insights.deterministic import generate_insights_md, trend_hours_above
    from weather_tool.insights.llm import generate_llm_narrative
    from weather_tool.storage.io import (
        save_insights_md,
        save_metadata_json,
        save_quality_json,
        save_raw_parquet,
        save_summary_csv,
    )

    # 1. Load raw data
    typer.echo(f"[1/6] Loading data ({cfg.mode} mode)...")
    if cfg.mode == "csv":
        raw = load_csv(cfg)
    else:
        raw = load_iem(cfg)

    typer.echo(f"       Loaded {len(raw)} raw records.")

    # 2. Normalize + window
    typer.echo("[2/6] Normalising timestamps & filtering window...")
    normed = normalize_timestamps(raw, tz=cfg.tz)
    windowed = filter_window(normed, cfg.start, cfg.end, tz=cfg.tz)
    typer.echo(f"       {len(windowed)} records in analysis window.")

    # 3. Interval inference (on deduplicated data)
    typer.echo("[3/6] Inferring sampling interval...")
    dedup = deduplicated(windowed)
    interval_info = infer_interval(dedup["timestamp"])
    typer.echo(f"       dt_minutes = {interval_info['dt_minutes']}")
    if interval_info["interval_change_flag"]:
        typer.echo("       WARNING: interval changes detected in data.")

    # 4. Yearly summary
    typer.echo("[4/6] Computing yearly summary...")
    summary = build_yearly_summary(windowed, cfg, interval_info)
    typer.echo(f"       {len(summary)} year(s) in summary.")

    if cfg.verbose:
        typer.echo(summary.to_string(index=False))

    # 5. Build quality report + metadata
    quality_report = {
        "station_id": cfg.station_id or "csv",
        "window": f"{cfg.start} to {cfg.end}",
        "total_raw_records": len(raw),
        "total_windowed_records": len(windowed),
        "interval": interval_info,
        "per_year": summary[
            ["year", "missing_pct", "duplicate_count", "nan_temp_count",
             "interval_change_flag", "partial_coverage_flag", "coverage_pct"]
        ].to_dict(orient="records"),
    }
    metadata = {
        "station_id": cfg.station_id or "csv",
        "source": cfg.mode,
        "units": cfg.units,
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

    # 6. Insights
    typer.echo("[5/6] Generating insights report...")
    insights_md = generate_insights_md(summary, cfg, interval_info)

    llm_narrative = None
    if cfg.use_llm:
        typer.echo("       Generating LLM narrative...")
        llm_narrative = generate_llm_narrative(summary, insights_md, quality_report)
        if llm_narrative:
            insights_md += "\n\n## LLM Narrative\n\n" + llm_narrative
        else:
            typer.echo("       LLM narrative skipped (no API key or openai not installed).")

    # 7. Save outputs
    typer.echo("[6/6] Saving outputs...")
    p1 = save_summary_csv(summary, cfg)
    p2 = save_raw_parquet(windowed, cfg)
    p3 = save_quality_json(quality_report, cfg)
    p4 = save_metadata_json(metadata, cfg)
    p5 = save_insights_md(insights_md, cfg)

    typer.echo(f"  Summary CSV:     {p1}")
    typer.echo(f"  Clean Parquet:   {p2}")
    typer.echo(f"  Quality JSON:    {p3}")
    typer.echo(f"  Metadata JSON:   {p4}")
    typer.echo(f"  Insights MD:     {p5}")
    typer.echo("Done.")


if __name__ == "__main__":
    app()
