"""Tests for the yearly summary aggregation pipeline."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from weather_tool.config import RunConfig
from weather_tool.core.aggregate import build_yearly_summary
from weather_tool.core.metrics import infer_interval
from weather_tool.core.normalize import normalize_timestamps, filter_window


def _build_dataset(
    start: str,
    periods: int,
    freq_minutes: int,
    temp_base: float = 70.0,
    temp_amplitude: float = 20.0,
    nan_every: int | None = None,
) -> pd.DataFrame:
    """Generate a synthetic weather dataset."""
    ts = pd.date_range(start, periods=periods, freq=f"{freq_minutes}min", tz="UTC")
    # Simple sinusoidal temp pattern
    hours = np.arange(periods) * freq_minutes / 60.0
    temps = temp_base + temp_amplitude * np.sin(2 * np.pi * hours / (24 * 365))
    if nan_every:
        temps[::nan_every] = np.nan
    return pd.DataFrame({
        "timestamp": ts,
        "temp": temps,
        "station_id": "TEST",
    })


class TestYearlySummary:
    def test_single_year_full(self):
        """Full year at 60-min intervals → 1 row, partial_coverage_flag False."""
        df = _build_dataset("2023-01-01", periods=8760, freq_minutes=60)
        cfg = RunConfig(
            mode="csv",
            input_path=None,  # not used here
            station_id="TEST",
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
            ref_temp=80.0,
            units="agnostic",
            tz="UTC",
        )
        normed = normalize_timestamps(df)
        windowed = filter_window(normed, cfg.start, cfg.end)
        from weather_tool.core.normalize import deduplicated
        dedup = deduplicated(windowed)
        interval = infer_interval(dedup["timestamp"])

        summary = build_yearly_summary(windowed, cfg, interval)
        assert len(summary) == 1
        row = summary.iloc[0]
        assert row["year"] == 2023
        assert row["dt_minutes"] == pytest.approx(60.0)
        assert row["partial_coverage_flag"] == False
        assert row["missing_pct"] < 0.01

    def test_partial_year_flag(self):
        """Jul–Dec window → partial_coverage_flag should be True."""
        # Generate full year but window is Jul-Dec
        df = _build_dataset("2023-01-01", periods=8760, freq_minutes=60)
        cfg = RunConfig(
            mode="csv",
            station_id="TEST",
            start=date(2023, 7, 1),
            end=date(2023, 12, 31),
            ref_temp=80.0,
            units="agnostic",
            tz="UTC",
        )
        normed = normalize_timestamps(df)
        windowed = filter_window(normed, cfg.start, cfg.end)
        from weather_tool.core.normalize import deduplicated
        dedup = deduplicated(windowed)
        interval = infer_interval(dedup["timestamp"])

        summary = build_yearly_summary(windowed, cfg, interval)
        assert len(summary) == 1
        row = summary.iloc[0]
        assert row["partial_coverage_flag"] == True
        assert row["coverage_pct"] < 0.98
        # Missing pct should still be low — data is complete within the window
        assert row["missing_pct"] < 0.02

    def test_multi_year(self):
        """2 years of data → 2 summary rows."""
        df = _build_dataset("2022-01-01", periods=8760 * 2, freq_minutes=60)
        cfg = RunConfig(
            mode="csv",
            station_id="TEST",
            start=date(2022, 1, 1),
            end=date(2023, 12, 31),
            ref_temp=80.0,
            units="agnostic",
            tz="UTC",
        )
        normed = normalize_timestamps(df)
        windowed = filter_window(normed, cfg.start, cfg.end)
        from weather_tool.core.normalize import deduplicated
        dedup = deduplicated(windowed)
        interval = infer_interval(dedup["timestamp"])

        summary = build_yearly_summary(windowed, cfg, interval)
        assert len(summary) == 2
        assert list(summary["year"]) == [2022, 2023]

    def test_hours_above_ref_20min(self):
        """20-min interval: hours = count_above * (20/60)."""
        # All temps at 70, ref=65 → all above
        periods = 72  # exactly 1 day at 20-min
        ts = pd.date_range("2023-07-01", periods=periods, freq="20min", tz="UTC")
        df = pd.DataFrame({
            "timestamp": ts,
            "temp": [70.0] * periods,
            "station_id": "TEST",
        })
        cfg = RunConfig(
            mode="csv",
            station_id="TEST",
            start=date(2023, 7, 1),
            end=date(2023, 7, 1),
            ref_temp=65.0,
            units="agnostic",
            tz="UTC",
        )
        normed = normalize_timestamps(df)
        windowed = filter_window(normed, cfg.start, cfg.end)
        from weather_tool.core.normalize import deduplicated
        dedup = deduplicated(windowed)
        interval = infer_interval(dedup["timestamp"])

        summary = build_yearly_summary(windowed, cfg, interval)
        row = summary.iloc[0]
        expected_hours = 72 * (20.0 / 60.0)  # = 24.0
        assert row["hours_above_ref"] == pytest.approx(expected_hours)

    def test_nan_temps_not_in_hours(self):
        """NaN temps should not contribute to hours_above_ref."""
        periods = 10
        ts = pd.date_range("2023-07-01", periods=periods, freq="60min", tz="UTC")
        temps = [80.0] * periods
        temps[0] = np.nan
        temps[5] = np.nan
        df = pd.DataFrame({
            "timestamp": ts,
            "temp": temps,
            "station_id": "TEST",
        })
        cfg = RunConfig(
            mode="csv",
            station_id="TEST",
            start=date(2023, 7, 1),
            end=date(2023, 7, 1),
            ref_temp=65.0,
            units="agnostic",
            tz="UTC",
        )
        normed = normalize_timestamps(df)
        windowed = filter_window(normed, cfg.start, cfg.end)
        from weather_tool.core.normalize import deduplicated
        dedup = deduplicated(windowed)
        interval = infer_interval(dedup["timestamp"])

        summary = build_yearly_summary(windowed, cfg, interval)
        row = summary.iloc[0]
        # 8 valid temps above 65, at 60-min → 8 hours
        assert row["hours_above_ref"] == pytest.approx(8.0)
        assert row["nan_temp_count"] == 2
