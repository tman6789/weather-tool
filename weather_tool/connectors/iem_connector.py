"""Download ASOS observations from Iowa Environmental Mesonet (IEM)."""

from __future__ import annotations

import io
import warnings
from datetime import date, datetime, timezone
from urllib.parse import urlencode

import pandas as pd
import requests

from weather_tool.config import RunConfig
from weather_tool.connectors.cache import (
    ALL_IEM_FIELDS,
    DEFAULT_CACHE_DIR,
    read_cached_year,
    write_cached_year,
)

_BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def _build_url(station: str, start: date, end: date, fields: list[str]) -> str:
    params: list[tuple[str, str | int]] = [
        ("station", station),
        *[("data", f) for f in fields],
        ("year1", start.year),
        ("month1", start.month),
        ("day1", start.day),
        ("year2", end.year),
        ("month2", end.month),
        ("day2", end.day),
        ("tz", "Etc/UTC"),
        ("format", "comma"),
        ("latlon", "no"),
        ("elev", "no"),
        ("missing", "empty"),
        ("trace", "empty"),
        ("direct", "no"),
        ("report_type", "3"),
    ]
    return _BASE_URL + "?" + urlencode(params)


def _contiguous_ranges(years: list[int]) -> list[tuple[int, int]]:
    """Group sorted years into contiguous ranges to minimise HTTP calls."""
    if not years:
        return []
    years = sorted(years)
    ranges: list[tuple[int, int]] = []
    start = years[0]
    prev = years[0]
    for y in years[1:]:
        if y == prev + 1:
            prev = y
        else:
            ranges.append((start, prev))
            start = y
            prev = y
    ranges.append((start, prev))
    return ranges


def _fetch_iem_raw(station_id: str, start: date, end: date) -> pd.DataFrame:
    """Fetch raw IEM data with ALL fields, return normalised DataFrame in UTC.

    Always requests all six IEM fields so the cache can serve any future
    field combination.
    """
    url = _build_url(station_id, start, end, list(ALL_IEM_FIELDS))

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    text = resp.text

    # IEM sometimes prepends comment lines starting with '#'
    lines = text.splitlines()
    data_lines = [ln for ln in lines if not ln.startswith("#")]
    clean_text = "\n".join(data_lines)

    df = pd.read_csv(io.StringIO(clean_text))

    # Normalise column names (strip whitespace, lowercase for lookup)
    col_lower = {c: c.strip().lower() for c in df.columns}
    col_map: dict[str, str] = {}
    for raw_col, lc in col_lower.items():
        if lc == "valid":
            col_map[raw_col] = "timestamp"
        elif lc == "station":
            col_map[raw_col] = "station_id"
        else:
            col_map[raw_col] = lc
    df = df.rename(columns=col_map)

    if "timestamp" not in df.columns:
        raise KeyError(
            f"Could not find timestamp column in IEM response. Columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    out["station_id"] = df.get("station_id", station_id)
    for fld in ALL_IEM_FIELDS:
        if fld in df.columns:
            out[fld] = pd.to_numeric(df[fld], errors="coerce")
        else:
            out[fld] = pd.Series(dtype="float64", index=out.index)
    return out


def load_iem(cfg: RunConfig) -> pd.DataFrame:
    """Download observations for *station_id* over [start, end].

    Uses a local Parquet cache (`.cache/v1/`) to avoid re-downloading
    previously fetched data.  The current UTC year is never cached
    because its data may be incomplete or backfilled.

    Returns
    -------
    pd.DataFrame
        Always contains: timestamp (datetime64[ns, TZ]), temp (float64), station_id (str).
        Additionally contains any requested fields that IEM returned
        (tmpf, dwpf, relh, sknt, drct, gust) as float64 columns.
    """
    if not cfg.station_id:
        raise ValueError("station_id is required for IEM mode")

    # Resolve requested fields (auto-include tmpf)
    fields = list(cfg.fields) if cfg.fields else ["tmpf"]
    if "tmpf" not in fields:
        warnings.warn(
            "tmpf (dry-bulb temperature) is not in --fields; auto-including it so "
            "the 'temp' column and hours_above_ref are populated. "
            "Add tmpf explicitly to suppress this warning.",
            stacklevel=2,
        )
        fields = ["tmpf"] + fields

    # Cache settings
    cache_dir = cfg.cache_dir or DEFAULT_CACHE_DIR
    use_cache = not cfg.no_cache
    verbose = cfg.verbose
    current_utc_year = datetime.now(timezone.utc).year

    # Determine year range
    years = list(range(cfg.start.year, cfg.end.year + 1))

    # Phase 1: check cache for each year
    cached_frames: list[pd.DataFrame] = []
    miss_years: list[int] = []
    for year in years:
        if year == current_utc_year:
            if verbose and use_cache:
                import typer
                typer.echo(f"  [current year \u2192 no cache] station={cfg.station_id} year={year}")
            miss_years.append(year)
        elif use_cache:
            cached = read_cached_year(cache_dir, cfg.station_id, year, verbose)
            if cached is not None:
                cached_frames.append(cached)
            else:
                miss_years.append(year)
        else:
            miss_years.append(year)

    # Phase 2: fetch missing years from IEM (grouped into contiguous ranges)
    fetched_frames: list[pd.DataFrame] = []
    for start_y, end_y in _contiguous_ranges(miss_years):
        fetch_start = date(start_y, 1, 1)
        fetch_end = date(end_y, 12, 31)
        raw = _fetch_iem_raw(cfg.station_id, fetch_start, fetch_end)
        fetched_frames.append(raw)
        # Write each completed year to cache (skip current year)
        if use_cache:
            for y in range(start_y, end_y + 1):
                if y != current_utc_year:
                    write_cached_year(cache_dir, cfg.station_id, y, raw, verbose)

    # Phase 3: assemble result
    all_frames = cached_frames + fetched_frames
    if not all_frames:
        # No data at all — return empty DataFrame with expected schema
        out = pd.DataFrame(columns=["timestamp", "station_id", "temp"] + fields)
        return out

    combined = pd.concat(all_frames, ignore_index=True)

    # Filter to exact requested date range
    ts_start = pd.Timestamp(cfg.start, tz="UTC")
    ts_end = pd.Timestamp(date(cfg.end.year, cfg.end.month, cfg.end.day), tz="UTC") + pd.Timedelta(days=1)
    mask = (combined["timestamp"] >= ts_start) & (combined["timestamp"] < ts_end)
    combined = combined[mask].reset_index(drop=True)

    # Select only requested fields
    out = pd.DataFrame()
    out["timestamp"] = combined["timestamp"]
    out["station_id"] = combined["station_id"]
    for fld in fields:
        if fld in combined.columns:
            out[fld] = combined[fld]

    # temp alias
    if "tmpf" in out.columns:
        out["temp"] = out["tmpf"]
    else:
        out["temp"] = pd.Series(dtype="float64", index=out.index)

    # Convert to requested timezone
    if cfg.tz != "UTC":
        out["timestamp"] = out["timestamp"].dt.tz_convert(cfg.tz)

    return out
