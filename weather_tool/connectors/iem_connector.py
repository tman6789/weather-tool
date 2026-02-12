"""Download ASOS observations from Iowa Environmental Mesonet (IEM)."""

from __future__ import annotations

import io
from datetime import date

import pandas as pd
import requests

from weather_tool.config import RunConfig

IEM_URL = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    "?station={station}"
    "&data=tmpf"
    "&year1={y1}&month1={m1}&day1={d1}"
    "&year2={y2}&month2={m2}&day2={d2}"
    "&tz=Etc/UTC&format=comma&latlon=no&elev=no&missing=empty"
    "&trace=empty&direct=no&report_type=3"
)


def _build_url(station: str, start: date, end: date) -> str:
    return IEM_URL.format(
        station=station,
        y1=start.year, m1=start.month, d1=start.day,
        y2=end.year, m2=end.month, d2=end.day,
    )


def load_iem(cfg: RunConfig) -> pd.DataFrame:
    """Download observations for *station_id* over [start, end].

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (datetime64[ns, UTC]), temp (float64), station_id (str).
    """
    if not cfg.station_id:
        raise ValueError("station_id is required for IEM mode")

    url = _build_url(cfg.station_id, cfg.start, cfg.end)

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    text = resp.text

    # IEM sometimes prepends comment lines starting with '#'
    lines = text.splitlines()
    data_lines = [ln for ln in lines if not ln.startswith("#")]
    clean_text = "\n".join(data_lines)

    df = pd.read_csv(io.StringIO(clean_text))

    # IEM column names vary; normalise
    col_map: dict[str, str] = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "valid":
            col_map[c] = "timestamp"
        elif cl == "tmpf":
            col_map[c] = "temp"
        elif cl == "station":
            col_map[c] = "station_id"
    df = df.rename(columns=col_map)

    if "timestamp" not in df.columns:
        raise KeyError(
            f"Could not find timestamp column in IEM response. Columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    out["temp"] = pd.to_numeric(
        df.get("temp", pd.Series(dtype="float64")), errors="coerce"
    )
    out["station_id"] = df.get("station_id", cfg.station_id)

    # Convert to requested timezone
    if cfg.tz != "UTC":
        out["timestamp"] = out["timestamp"].dt.tz_convert(cfg.tz)

    return out
