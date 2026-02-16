"""Download ASOS observations from Iowa Environmental Mesonet (IEM)."""

from __future__ import annotations

import io
import warnings
from datetime import date
from urllib.parse import urlencode

import pandas as pd
import requests

from weather_tool.config import RunConfig

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


def load_iem(cfg: RunConfig) -> pd.DataFrame:
    """Download observations for *station_id* over [start, end].

    Returns
    -------
    pd.DataFrame
        Always contains: timestamp (datetime64[ns, TZ]), temp (float64), station_id (str).
        Additionally contains any requested fields that IEM returned
        (tmpf, dwpf, relh, sknt, drct, gust) as float64 columns.
    """
    if not cfg.station_id:
        raise ValueError("station_id is required for IEM mode")

    fields = list(cfg.fields) if cfg.fields else ["tmpf"]
    if "tmpf" not in fields:
        warnings.warn(
            "tmpf (dry-bulb temperature) is not in --fields; auto-including it so "
            "the 'temp' column and hours_above_ref are populated. "
            "Add tmpf explicitly to suppress this warning.",
            stacklevel=2,
        )
        fields = ["tmpf"] + fields
    url = _build_url(cfg.station_id, cfg.start, cfg.end, fields)

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

    # Build rename map
    col_map: dict[str, str] = {}
    for raw_col, lc in col_lower.items():
        if lc == "valid":
            col_map[raw_col] = "timestamp"
        elif lc == "station":
            col_map[raw_col] = "station_id"
        else:
            # Keep IEM field names as-is (tmpf, dwpf, relh, sknt, drct, gust, …)
            col_map[raw_col] = lc

    df = df.rename(columns=col_map)

    if "timestamp" not in df.columns:
        raise KeyError(
            f"Could not find timestamp column in IEM response. Columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    out["station_id"] = df.get("station_id", cfg.station_id)

    # Copy requested numeric fields
    for field in fields:
        if field in df.columns:
            out[field] = pd.to_numeric(df[field], errors="coerce")

    # Ensure non-requested numeric fields that appeared in the response are dropped
    # (only keep what was asked for)

    # temp alias: prefer tmpf, otherwise leave temp as NaN series for compatibility
    if "tmpf" in out.columns:
        out["temp"] = out["tmpf"]
    else:
        out["temp"] = pd.Series(dtype="float64", index=out.index)

    # Convert to requested timezone
    if cfg.tz != "UTC":
        out["timestamp"] = out["timestamp"].dt.tz_convert(cfg.tz)

    return out
