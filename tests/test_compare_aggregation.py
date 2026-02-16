"""Tests for window-level aggregation in core/compare.py."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from weather_tool.core.compare import aggregate_station_window
from weather_tool.config import MISSING_DATA_WARNING_THRESHOLD


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_summary(
    years: list[int],
    tmax: list[float],
    tmin: list[float],
    hours_above_ref: list[float],
    missing_pct: list[float | None] | None = None,
    wb_p99: list[float | None] | None = None,
    hours_below_32: list[float] | None = None,
    t_p99: list[float | None] | None = None,
    t_p996: list[float | None] | None = None,
    coverage_pct: list[float] | None = None,
) -> pd.DataFrame:
    n = len(years)
    data: dict = {
        "year": years,
        "tmax": tmax,
        "tmin": tmin,
        "hours_above_ref": hours_above_ref,
        "missing_pct": [m if m is not None else float("nan") for m in (missing_pct or [0.0] * n)],
        "coverage_pct": coverage_pct or [1.0] * n,
        "partial_coverage_flag": [False] * n,
        "interval_change_flag": [False] * n,
        "dt_minutes": [60.0] * n,
    }
    if wb_p99 is not None:
        data["wb_p99"] = [v if v is not None else float("nan") for v in wb_p99]
        data["wb_p996"] = [v if v is not None else float("nan") for v in wb_p99]
    if hours_below_32 is not None:
        data["hours_below_32"] = hours_below_32
    if t_p99 is not None:
        data["t_p99"] = [v if v is not None else float("nan") for v in t_p99]
        data["t_p996"] = [v if v is not None else float("nan") for v in (t_p996 or t_p99)]
    return pd.DataFrame(data)


def _make_windowed(temps: list[float], dt_minutes: float = 60.0) -> pd.DataFrame:
    """Build a minimal windowed DataFrame for aggregation tests."""
    n = len(temps)
    ts = pd.date_range("2022-01-01", periods=n, freq=f"{int(dt_minutes)}min", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "temp": temps})
    df["_is_dup"] = False
    return df


def _interval_info(dt: float = 60.0) -> dict:
    return {
        "dt_minutes": dt,
        "p10": dt,
        "p90": dt,
        "interval_change_flag": False,
        "unique_diff_counts": {dt: len([1]) - 1},
    }


# ── Tests: basic extremes ──────────────────────────────────────────────────────

class TestTemperatureExtremes:
    def test_tmax_max(self):
        summary = _make_summary([2020, 2021], tmax=[95.0, 102.0], tmin=[28.0, 35.0], hours_above_ref=[100.0, 200.0])
        windowed = _make_windowed([70.0] * 10)
        row = aggregate_station_window(summary, windowed, _interval_info(), [65.0], "KTEST")
        assert row["tmax_max"] == pytest.approx(102.0)

    def test_tmin_min(self):
        summary = _make_summary([2020, 2021], tmax=[95.0, 102.0], tmin=[28.0, 35.0], hours_above_ref=[100.0, 200.0])
        windowed = _make_windowed([70.0] * 10)
        row = aggregate_station_window(summary, windowed, _interval_info(), [65.0], "KTEST")
        assert row["tmin_min"] == pytest.approx(28.0)

    def test_t_p99_median(self):
        summary = _make_summary([2020, 2021, 2022], tmax=[95, 96, 97], tmin=[30, 31, 32],
                                 hours_above_ref=[100, 110, 120],
                                 t_p99=[90.0, 92.0, 94.0])
        windowed = _make_windowed([70.0] * 10)
        row = aggregate_station_window(summary, windowed, _interval_info(), [65.0], "KTEST")
        # median of [90, 92, 94] = 92
        assert row["t_p99_median"] == pytest.approx(92.0)


# ── Tests: hours above ref (computed from windowed series) ────────────────────

class TestHoursAboveRef:
    def test_hours_above_ref_sum_single_threshold(self):
        """All temps at 80°F, ref=65 → all count; dt=60min."""
        windowed = _make_windowed([80.0] * 24)  # 24 hours above ref
        summary = _make_summary([2022], tmax=[80.0], tmin=[80.0], hours_above_ref=[24.0])
        row = aggregate_station_window(summary, windowed, _interval_info(60.0), [65.0], "KTEST")
        assert row["hours_above_ref_65_sum"] == pytest.approx(24.0)

    def test_hours_above_ref_zero_when_all_below(self):
        windowed = _make_windowed([50.0] * 24)
        summary = _make_summary([2022], tmax=[50.0], tmin=[50.0], hours_above_ref=[0.0])
        row = aggregate_station_window(summary, windowed, _interval_info(60.0), [65.0], "KTEST")
        assert row["hours_above_ref_65_sum"] == 0.0

    def test_multiple_ref_temps(self):
        """Temps at 90°F above 65, 80, and 85 → all > 0; temps above 95 → 0."""
        windowed = _make_windowed([90.0] * 10)
        summary = _make_summary([2022], tmax=[90.0], tmin=[90.0], hours_above_ref=[10.0])
        row = aggregate_station_window(summary, windowed, _interval_info(60.0), [65.0, 80.0, 95.0], "KTEST")
        assert row["hours_above_ref_65_sum"] == pytest.approx(10.0)
        assert row["hours_above_ref_80_sum"] == pytest.approx(10.0)
        assert row["hours_above_ref_95_sum"] == 0.0


# ── Tests: freeze hours ────────────────────────────────────────────────────────

class TestFreezeHours:
    def test_freeze_hours_from_precomputed_column(self):
        """Uses hours_below_32 column from yearly summary when present."""
        summary = _make_summary([2020, 2021], tmax=[50, 50], tmin=[10, 15],
                                 hours_above_ref=[0, 0], hours_below_32=[200.0, 150.0])
        windowed = _make_windowed([25.0] * 10)
        row = aggregate_station_window(summary, windowed, _interval_info(60.0), [65.0], "KTEST")
        assert row["freeze_hours_sum"] == pytest.approx(350.0)

    def test_freeze_hours_fallback_from_windowed(self):
        """When hours_below_32 absent, computes directly from windowed temps."""
        windowed = _make_windowed([25.0] * 8)  # all below 32 → 8 hrs at 60min
        summary = _make_summary([2022], tmax=[25.0], tmin=[25.0], hours_above_ref=[0.0])
        # No hours_below_32 column in summary
        row = aggregate_station_window(summary, windowed, _interval_info(60.0), [65.0], "KTEST")
        assert row["freeze_hours_sum"] == pytest.approx(8.0)


# ── Tests: wet-bulb ───────────────────────────────────────────────────────────

class TestWetbulbAggregation:
    def test_wb_p99_median(self):
        summary = _make_summary([2020, 2021, 2022], tmax=[95, 95, 95], tmin=[50, 50, 50],
                                 hours_above_ref=[100, 110, 120],
                                 wb_p99=[72.0, 75.0, 78.0])
        windowed = _make_windowed([70.0] * 10)
        row = aggregate_station_window(summary, windowed, _interval_info(), [65.0], "KTEST")
        assert row["wb_p99_median"] == pytest.approx(75.0)

    def test_wb_absent_when_no_column(self):
        summary = _make_summary([2020], tmax=[95.0], tmin=[50.0], hours_above_ref=[100.0])
        windowed = _make_windowed([70.0] * 10)
        row = aggregate_station_window(summary, windowed, _interval_info(), [65.0], "KTEST")
        assert "wb_p99_median" not in row


# ── Tests: missing_pct NaN-safe ───────────────────────────────────────────────

class TestMissingPctNaNSafe:
    def test_nan_missing_pct_excluded_from_mean(self):
        """Years with interval_unknown_flag have NaN missing_pct — should be excluded."""
        summary = _make_summary([2020, 2021], tmax=[90, 90], tmin=[30, 30],
                                 hours_above_ref=[100, 100],
                                 missing_pct=[None, 0.05])  # None → NaN
        windowed = _make_windowed([70.0] * 10)
        row = aggregate_station_window(summary, windowed, _interval_info(), [65.0], "KTEST")
        # Only 0.05 is valid
        assert row["timestamp_missing_pct_avg"] == pytest.approx(0.05)

    def test_missing_data_warning_triggered(self):
        summary = _make_summary([2020], tmax=[90.0], tmin=[30.0], hours_above_ref=[100.0],
                                 missing_pct=[0.05])  # > 2% threshold
        windowed = _make_windowed([70.0] * 10)
        row = aggregate_station_window(summary, windowed, _interval_info(), [65.0], "KTEST")
        assert row["missing_data_warning"] is True

    def test_missing_data_warning_not_triggered_when_clean(self):
        summary = _make_summary([2020], tmax=[90.0], tmin=[30.0], hours_above_ref=[100.0],
                                 missing_pct=[0.001])
        windowed = _make_windowed([70.0] * 10)
        row = aggregate_station_window(summary, windowed, _interval_info(), [65.0], "KTEST")
        assert row["missing_data_warning"] is False
