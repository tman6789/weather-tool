"""Load weather data from a user-provided CSV file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from weather_tool.config import RunConfig


def load_csv(cfg: RunConfig) -> pd.DataFrame:
    """Read a CSV, returning a DataFrame with columns ``timestamp`` and ``temp``.

    Parameters
    ----------
    cfg : RunConfig
        Must have ``input_path``, ``timestamp_col``, ``temp_col`` set.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (datetime64[ns, tz]), temp (float64).
    """
    path = Path(cfg.input_path)  # type: ignore[arg-type]
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    if cfg.timestamp_col not in df.columns:
        raise KeyError(
            f"Timestamp column '{cfg.timestamp_col}' not found. "
            f"Available: {list(df.columns)}"
        )
    if cfg.temp_col not in df.columns:
        raise KeyError(
            f"Temperature column '{cfg.temp_col}' not found. "
            f"Available: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df[cfg.timestamp_col], utc=False)
    out["temp"] = pd.to_numeric(df[cfg.temp_col], errors="coerce")

    # Localize to the requested timezone
    if out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(cfg.tz)
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert(cfg.tz)

    if cfg.station_id:
        out["station_id"] = cfg.station_id
    else:
        out["station_id"] = "csv"

    return out
