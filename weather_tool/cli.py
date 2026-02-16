"""Typer CLI entry-point for weather-tool."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import typer

from weather_tool.config import Mode, RunConfig, Units

app = typer.Typer(name="weather-tool", help="Deterministic historical weather analysis tool.")


@app.callback()
def _callback() -> None:
    """Deterministic historical weather analysis tool."""


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
    fields: str = typer.Option("tmpf", "--fields", help="Comma-separated IEM fields to request (e.g. tmpf,dwpf,relh,sknt,drct,gust)"),
) -> None:
    """Run a weather analysis pipeline."""
    fields_list = [f.strip() for f in fields.split(",") if f.strip()]
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
        fields=fields_list,
    )
    cfg.validate()

    _execute(cfg)


def _execute(cfg: RunConfig) -> None:
    """Orchestrate the full pipeline and save outputs."""
    from weather_tool.insights.deterministic import generate_insights_md
    from weather_tool.insights.llm import generate_llm_narrative
    from weather_tool.pipeline import run_station_pipeline
    from weather_tool.storage.io import (
        save_insights_md,
        save_metadata_json,
        save_quality_json,
        save_raw_parquet,
        save_summary_csv,
    )

    typer.echo(f"[1/6] Loading data ({cfg.mode} mode)...")
    result = run_station_pipeline(cfg, echo=False)

    # Echo the pipeline steps that run_station_pipeline handled silently
    typer.echo(f"       Loaded {result.quality_report['total_raw_records']} raw records.")
    typer.echo("[2/6] Normalising timestamps & filtering window...")
    typer.echo(f"       {result.quality_report['total_windowed_records']} records in analysis window.")
    if "wetbulb_f" in result.windowed.columns:
        wb_avail = result.windowed["wetbulb_f"].notna().sum()
        typer.echo(f"       Wet-bulb computed: {wb_avail} non-NaN values.")
    typer.echo("[3/6] Inferring sampling interval...")
    typer.echo(f"       dt_minutes = {result.interval_info['dt_minutes']}")
    if result.interval_info["interval_change_flag"]:
        typer.echo("       WARNING: interval changes detected in data.")
    typer.echo("[4/6] Computing yearly summary...")
    typer.echo(f"       {len(result.summary)} year(s) in summary.")

    if cfg.verbose:
        typer.echo(result.summary.to_string(index=False))

    typer.echo("[5/6] Generating insights report...")
    insights_md = generate_insights_md(result.summary, cfg, result.interval_info)

    llm_narrative = None
    if cfg.use_llm:
        typer.echo("       Generating LLM narrative...")
        llm_narrative = generate_llm_narrative(result.summary, insights_md, result.quality_report)
        if llm_narrative:
            insights_md += "\n\n## LLM Narrative\n\n" + llm_narrative
        else:
            typer.echo("       LLM narrative skipped (no API key or openai not installed).")

    typer.echo("[6/6] Saving outputs...")
    p1 = save_summary_csv(result.summary, cfg)
    p2 = save_raw_parquet(result.windowed, cfg)
    p3 = save_quality_json(result.quality_report, cfg)
    p4 = save_metadata_json(result.metadata, cfg)
    p5 = save_insights_md(insights_md, cfg)

    typer.echo(f"  Summary CSV:     {p1}")
    typer.echo(f"  Clean Parquet:   {p2}")
    typer.echo(f"  Quality JSON:    {p3}")
    typer.echo(f"  Metadata JSON:   {p4}")
    typer.echo(f"  Insights MD:     {p5}")
    typer.echo("Done.")


@app.command()
def compare(
    stations: list[str] = typer.Argument(..., help="Station IDs to compare (e.g. KIAD KDEN KPHX)"),
    start: str = typer.Option(..., help="Analysis start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="Analysis end date YYYY-MM-DD"),
    ref_temps: str = typer.Option("65", "--ref-temps", help="Comma-separated reference temperatures (e.g. 80,85,90)"),
    fields: str = typer.Option("tmpf,dwpf,relh", "--fields", help="Comma-separated IEM fields to request"),
    tz: str = typer.Option("UTC", "--tz", help="IANA timezone for timestamps"),
    outdir: Path = typer.Option(Path("outputs"), help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Compare climate metrics across multiple stations."""
    from weather_tool.core.compare import build_compare_summary
    from weather_tool.insights.compare_report import generate_compare_report_md
    from weather_tool.pipeline import run_station_pipeline
    from weather_tool.storage.io import save_compare_outputs

    ref_temps_list = [float(t.strip()) for t in ref_temps.split(",") if t.strip()]
    fields_list = [f.strip() for f in fields.split(",") if f.strip()]
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    if len(stations) < 2:
        typer.echo("ERROR: at least 2 stations are required for compare.", err=True)
        raise typer.Exit(1)

    station_results = []
    for i, sid in enumerate(stations, 1):
        typer.echo(f"[{i}/{len(stations)}] Running pipeline for {sid}...")
        cfg = RunConfig(
            mode="iem",
            station_id=sid,
            start=start_date,
            end=end_date,
            ref_temp=ref_temps_list[0],
            fields=fields_list,
            tz=tz,
            outdir=outdir,
            verbose=verbose,
        )
        cfg.validate()
        result = run_station_pipeline(cfg, echo=verbose)
        station_results.append(result)
        typer.echo(f"       {len(result.summary)} year(s), dt={result.interval_info['dt_minutes']} min.")

    typer.echo("Building comparison table...")
    compare_df = build_compare_summary(station_results, ref_temps_list)

    typer.echo("Generating comparison report...")
    report_md = generate_compare_report_md(
        compare_df=compare_df,
        stations=stations,
        window_start=start_date,
        window_end=end_date,
        fields=fields_list,
        ref_temps=ref_temps_list,
    )

    import pandas as pd
    yearly_concat = pd.concat(
        [r.summary.assign(station_id=r.cfg.station_id) for r in station_results],
        ignore_index=True,
    )

    import datetime as _dt
    metadata = {
        "stations": stations,
        "window_start": start,
        "window_end": end,
        "fields": fields_list,
        "ref_temps": ref_temps_list,
        "tz": tz,
        "run_timestamp": str(_dt.datetime.now(tz=_dt.timezone.utc).isoformat()),
        "per_station_dt": {
            r.cfg.station_id: r.interval_info["dt_minutes"]
            for r in station_results
        },
    }

    from weather_tool.storage.io import _compare_dir
    cdir = _compare_dir(outdir, start, end, stations)
    paths = save_compare_outputs(
        compare_df=compare_df,
        station_results=station_results,
        yearly_concat=yearly_concat,
        report_md=report_md,
        metadata=metadata,
        compare_dir=cdir,
    )

    typer.echo(f"  Compare summary: {paths['summary']}")
    typer.echo(f"  Compare report:  {paths['report']}")
    typer.echo(f"  Yearly parquet:  {paths['yearly']}")
    typer.echo(f"  Metadata JSON:   {paths['metadata']}")
    typer.echo("Done.")


if __name__ == "__main__":
    app()
