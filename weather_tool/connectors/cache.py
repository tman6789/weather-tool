"""Versioned Parquet cache for raw IEM data — keyed by {station_id}_{year}."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

CACHE_SCHEMA_VERSION = 1
ALL_IEM_FIELDS = ("tmpf", "dwpf", "relh", "sknt", "drct", "gust")
CACHE_COLUMNS = ("timestamp", "station_id") + ALL_IEM_FIELDS  # 8 columns
DEFAULT_CACHE_DIR = Path(".cache")


def _log(verbose: bool, msg: str) -> None:
    """Emit a cache diagnostic when verbose mode is on."""
    if verbose:
        import typer

        typer.echo(msg)


def cache_dir_versioned(cache_dir: Path) -> Path:
    """Return the versioned subdirectory for the current schema."""
    return cache_dir / f"v{CACHE_SCHEMA_VERSION}"


def _cache_path(cache_dir: Path, station_id: str, year: int) -> Path:
    return cache_dir_versioned(cache_dir) / f"{station_id}_{year}.parquet"


def read_cached_year(
    cache_dir: Path,
    station_id: str,
    year: int,
    verbose: bool = False,
) -> pd.DataFrame | None:
    """Read a cached year from Parquet.

    Returns the DataFrame on hit, or None on miss / corrupt / incomplete.
    Corrupt or incomplete files are deleted (self-healing).
    """
    p = _cache_path(cache_dir, station_id, year)
    if not p.exists():
        _log(verbose, f"  [cache miss] station={station_id} year={year}")
        return None
    try:
        df = pd.read_parquet(p, engine="pyarrow")
    except Exception:
        _log(verbose, f"  [cache corrupt \u2192 refetch] station={station_id} year={year}")
        p.unlink(missing_ok=True)
        return None
    if not set(CACHE_COLUMNS).issubset(df.columns):
        _log(verbose, f"  [cache corrupt \u2192 refetch] station={station_id} year={year}")
        p.unlink(missing_ok=True)
        return None
    _log(verbose, f"  [cache hit] station={station_id} year={year}")
    return df


def write_cached_year(
    cache_dir: Path,
    station_id: str,
    year: int,
    df: pd.DataFrame,
    verbose: bool = False,
) -> None:
    """Write one year of raw IEM data to the versioned Parquet cache."""
    vdir = cache_dir_versioned(cache_dir)
    vdir.mkdir(parents=True, exist_ok=True)
    year_df = df[df["timestamp"].dt.year == year]
    if year_df.empty:
        return
    cols = [c for c in CACHE_COLUMNS if c in year_df.columns]
    year_df[cols].to_parquet(
        _cache_path(cache_dir, station_id, year),
        index=False,
        engine="pyarrow",
    )
    _log(verbose, f"  [cache write] station={station_id} year={year}")
