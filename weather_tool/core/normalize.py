"""Timestamp normalisation, sorting, and de-duplication."""

from __future__ import annotations

from datetime import date

import pandas as pd


def normalize_timestamps(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """Ensure ``timestamp`` is tz-aware, sorted, with duplicates flagged.

    Returns a *copy* with columns:
        timestamp  – sorted, tz-aware
        temp       – as-is (may contain NaN)
        station_id – as-is
        _is_dup    – bool, True for duplicate timestamps (keeps first)
    """
    out = df.copy()

    # Ensure tz-aware
    if out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(tz)

    out = out.sort_values("timestamp").reset_index(drop=True)

    # Mark duplicates (keep first occurrence)
    out["_is_dup"] = out["timestamp"].duplicated(keep="first")

    return out


def filter_window(
    df: pd.DataFrame,
    start: date,
    end: date,
    tz: str = "UTC",
) -> pd.DataFrame:
    """Return rows whose timestamp falls within [start, end] inclusive (full days)."""
    ts_start = pd.Timestamp(start.isoformat()).tz_localize(tz)
    ts_end = pd.Timestamp(f"{end.isoformat()} 23:59:59").tz_localize(tz)
    mask = (df["timestamp"] >= ts_start) & (df["timestamp"] <= ts_end)
    return df.loc[mask].reset_index(drop=True)


def deduplicated(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the non-duplicate rows (keeps first of each timestamp)."""
    return df.loc[~df["_is_dup"]].reset_index(drop=True)
