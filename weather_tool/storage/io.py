"""File I/O — export summary CSV, clean parquet, quality JSON, metadata JSON, insights MD."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from weather_tool.config import RunConfig


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
