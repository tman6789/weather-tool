"""Tests for weather_tool/insights/death_day.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from weather_tool.insights.death_day import find_death_day_candidates

_REQUIRED_KEYS = {
    "rank", "mode", "confidence", "window_hours",
    "start_ts", "end_ts", "stress_score", "z_tdb", "z_wb",
    "tdb_mean_f", "tdb_max_f", "twb_mean_f", "twb_max_f",
    "rh_mean_pct", "wind_mean_kt", "calm_pct",
}


def _make_windowed(
    n_hours: int = 500,
    inject_heatwave: bool = True,
    include_wb: bool = True,
    heatwave_start: int = 200,
    heatwave_len: int = 48,
    heatwave_tdb: float = 102.0,
    heatwave_wb: float = 82.0,
) -> pd.DataFrame:
    """Build a synthetic deduplicated windowed DataFrame."""
    ts = pd.date_range("2022-01-01", periods=n_hours, freq="60min", tz="UTC")
    temps = [70.0 + 5 * np.sin(2 * np.pi * i / 24) for i in range(n_hours)]
    wb = [62.0 + 4 * np.sin(2 * np.pi * i / 24) for i in range(n_hours)]

    if inject_heatwave:
        for i in range(heatwave_start, min(heatwave_start + heatwave_len, n_hours)):
            temps[i] = heatwave_tdb
            wb[i] = heatwave_wb

    df = pd.DataFrame({
        "timestamp": ts,
        "temp": temps,
        "_is_dup": False,
    })
    if include_wb:
        df["wetbulb_f"] = wb
        df["relh"] = 55.0
        df["wind_speed_kt"] = 5.0
        df["is_calm"] = False

    return df


# ── Helpers ────────────────────────────────────────────────────────────────────

_TDB_P99 = 82.0
_TDB_P996 = 85.0
_WB_P99 = 68.0
_WB_P996 = 72.0


def _run(df, **kwargs):
    defaults = dict(
        dt_minutes=60.0,
        window_hours=24,
        tdb_p99=_TDB_P99,
        tdb_p996=_TDB_P996,
        wb_p99=_WB_P99,
        wb_p996=_WB_P996,
        top_n=5,
    )
    defaults.update(kwargs)
    return find_death_day_candidates(df, **defaults)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_returns_nonempty_list_with_heatwave():
    df = _make_windowed()
    result = _run(df)
    assert isinstance(result, list)
    assert len(result) >= 1


def test_rank1_overlaps_injected_heatwave():
    """Rank 1 candidate window should overlap with the injected heatwave block."""
    df = _make_windowed(heatwave_start=200, heatwave_len=48)
    result = _run(df, top_n=3)
    assert len(result) >= 1
    r1 = result[0]
    # The end_ts isoformat may be tz-naive (numpy datetime64 strips tz in .values).
    # Compare offset from the epoch instead.
    end_ts = pd.Timestamp(r1["end_ts"])
    base = pd.Timestamp("2022-01-01")  # tz-naive to match
    heatwave_start_ts = base + pd.Timedelta(hours=200)
    heatwave_end_ts = base + pd.Timedelta(hours=247)
    assert heatwave_start_ts <= end_ts <= heatwave_end_ts + pd.Timedelta(hours=24)


def test_top_n_respected():
    df = _make_windowed()
    result = _run(df, top_n=2)
    assert len(result) <= 2


def test_stress_score_monotone():
    """Two separate heatwaves >24h apart should yield 2 non-overlapping candidates."""
    # Inject two heatwaves separated by 50 hours (> window_hours=24)
    n = 600
    ts = pd.date_range("2022-01-01", periods=n, freq="60min", tz="UTC")
    temps = [70.0 + 3 * np.sin(2 * np.pi * i / 24) for i in range(n)]
    wb = [62.0 + 2 * np.sin(2 * np.pi * i / 24) for i in range(n)]
    # First heatwave at 100-123
    for i in range(100, 124):
        temps[i] = 105.0
        wb[i] = 85.0
    # Second heatwave at 200-223 (100h gap > window_hours/2=12h)
    for i in range(200, 224):
        temps[i] = 103.0
        wb[i] = 83.0
    df = pd.DataFrame({"timestamp": ts, "temp": temps, "wetbulb_f": wb, "_is_dup": False})
    result = _run(df, top_n=3)
    assert len(result) >= 2
    scores = [r["stress_score"] for r in result if r["stress_score"] is not None]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], "Stress scores must be non-increasing by rank"


def test_no_wetbulb_returns_heat_day_mode():
    df = _make_windowed(include_wb=False)
    result = _run(df, wb_p99=None, wb_p996=None)
    assert len(result) >= 1
    assert result[0]["mode"] == "heat_day"
    assert result[0]["confidence"] == "low"
    assert result[0]["z_wb"] is None
    assert result[0]["twb_mean_f"] is None


def test_invalid_dt_returns_empty():
    df = _make_windowed()
    assert _run(df, dt_minutes=0) == []
    assert _run(df, dt_minutes=float("nan")) == []
    assert _run(df, dt_minutes=-1.0) == []


def test_empty_df_returns_empty():
    df = pd.DataFrame({"timestamp": pd.DatetimeIndex([]), "temp": pd.Series(dtype=float)})
    result = find_death_day_candidates(df, 60.0, 24, _TDB_P99, _TDB_P996, _WB_P99, _WB_P996)
    assert result == []


def test_all_nan_temp_returns_empty():
    ts = pd.date_range("2022-01-01", periods=100, freq="60min", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "temp": float("nan")})
    result = find_death_day_candidates(df, 60.0, 24, _TDB_P99, _TDB_P996, None, None)
    assert result == []


def test_output_keys_complete():
    df = _make_windowed()
    result = _run(df, top_n=1)
    assert len(result) == 1
    assert _REQUIRED_KEYS == set(result[0].keys())


def test_overlapping_windows_deduplicated():
    """Two spike blocks 4h apart with window_hours=24 should yield only 1 accepted candidate."""
    n = 400
    ts = pd.date_range("2022-01-01", periods=n, freq="60min", tz="UTC")
    temps = [70.0] * n
    wb = [62.0] * n
    # Two spikes only 4 hours apart (well within window_hours/2 = 12h)
    for i in range(100, 112):
        temps[i] = 105.0
        wb[i] = 85.0
    for i in range(104, 116):  # overlaps with first spike
        temps[i] = 104.0
        wb[i] = 84.0
    df = pd.DataFrame({"timestamp": ts, "temp": temps, "wetbulb_f": wb, "_is_dup": False})
    result = _run(df, top_n=5)
    # The two spikes are too close → should be deduplicated to 1
    assert len(result) == 1


def test_windows_13h_apart_deduplicated():
    """Two spikes 13h apart with window_hours=24 overlap by 11h → only 1 accepted.

    13h gap > half_window (12h) but < window (24h) — the old half_window criterion
    would have incorrectly accepted both.  With the corrected window_td criterion,
    only 1 candidate survives deduplication.
    """
    n = 400
    ts = pd.date_range("2022-01-01", periods=n, freq="60min", tz="UTC")
    temps = [70.0] * n
    wb = [62.0] * n
    for i in range(100, 112):   # spike 1
        temps[i] = 105.0
        wb[i] = 85.0
    for i in range(113, 125):   # spike 2, exactly 13h later (> 12h half-window)
        temps[i] = 104.0
        wb[i] = 84.0
    df = pd.DataFrame({"timestamp": ts, "temp": temps, "wetbulb_f": wb, "_is_dup": False})
    result = _run(df, top_n=5)
    assert len(result) == 1, f"Expected 1 (13h gap < 24h window = still overlaps), got {len(result)}"


def test_missing_wb_p99_falls_back_to_heat_day():
    """Passing wb_p99=None with present wetbulb_f column → heat_day mode (wb baselines missing)."""
    df = _make_windowed(include_wb=True)
    result = find_death_day_candidates(
        df, 60.0, 24, _TDB_P99, _TDB_P996, None, None
    )
    assert len(result) >= 1
    assert result[0]["mode"] == "heat_day"


def test_rank_field_sequential():
    df = _make_windowed()
    result = _run(df, top_n=3)
    for i, r in enumerate(result, 1):
        assert r["rank"] == i


def test_window_hours_in_output():
    df = _make_windowed()
    result = _run(df, window_hours=48, top_n=1)
    assert result[0]["window_hours"] == 48


def test_nan_tdb_p99_returns_empty():
    df = _make_windowed()
    result = find_death_day_candidates(
        df, 60.0, 24, float("nan"), _TDB_P996, _WB_P99, _WB_P996
    )
    assert result == []
