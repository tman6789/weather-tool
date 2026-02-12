"""Tests for missing_pct computation (windowed, not full-calendar-year)."""

import numpy as np
import pandas as pd
import pytest

from weather_tool.core.metrics import expected_records
from weather_tool.core.quality import compute_quality


def _make_slice_df(
    start: str,
    periods: int,
    freq_minutes: int,
    nan_indices: list[int] | None = None,
    dup_indices: list[int] | None = None,
) -> pd.DataFrame:
    """Build a small DataFrame simulating a year-slice."""
    ts = pd.date_range(start, periods=periods, freq=f"{freq_minutes}min", tz="UTC")
    temps = pd.Series(np.random.uniform(50, 90, size=periods))
    if nan_indices:
        for i in nan_indices:
            temps.iloc[i] = np.nan

    df = pd.DataFrame({"timestamp": ts, "temp": temps})
    df["_is_dup"] = False

    if dup_indices:
        # Add duplicate rows for specified indices
        dup_rows = df.iloc[dup_indices].copy()
        dup_rows["_is_dup"] = True
        df = pd.concat([df, dup_rows]).sort_values("timestamp").reset_index(drop=True)

    return df


class TestExpectedRecords:
    def test_one_day_60min(self):
        s = pd.Timestamp("2023-07-01 00:00:00", tz="UTC")
        e = pd.Timestamp("2023-07-01 23:59:59", tz="UTC")
        # 23:59:59 = 1439.983 min → floor(1439.983/60)+1 = 23+1 = 24
        assert expected_records(s, e, 60.0) == 24

    def test_one_day_20min(self):
        s = pd.Timestamp("2023-07-01 00:00:00", tz="UTC")
        e = pd.Timestamp("2023-07-01 23:59:59", tz="UTC")
        # 1439.98.. minutes / 20 = 71.99 → floor = 71 + 1 = 72
        assert expected_records(s, e, 20.0) == 72


class TestMissingPct:
    def test_perfect_coverage_missing_near_zero(self):
        """Complete July-Dec data at 60-min → missing_pct near 0, NOT ~50%."""
        start = "2023-07-01 00:00"
        # ~184 days * 24 records/day = 4416 records
        periods = 4416
        df = _make_slice_df(start, periods=periods, freq_minutes=60)

        slice_start = pd.Timestamp("2023-07-01", tz="UTC")
        slice_end = pd.Timestamp("2023-12-31 23:59:59", tz="UTC")

        q = compute_quality(df, slice_start, slice_end, dt_minutes=60.0, interval_change_flag=False)
        # missing_pct should be very small, NOT 0.5
        assert q["missing_pct"] < 0.02, (
            f"Expected near-zero missing for perfect Jul-Dec data, got {q['missing_pct']}"
        )

    def test_half_data_missing(self):
        """Half the expected records present → ~50% missing."""
        slice_start = pd.Timestamp("2023-01-01", tz="UTC")
        slice_end = pd.Timestamp("2023-01-01 23:59:59", tz="UTC")
        # Expected at 60-min for 1 day: 25 records
        # Provide only 12
        df = _make_slice_df("2023-01-01", periods=12, freq_minutes=60)
        q = compute_quality(df, slice_start, slice_end, dt_minutes=60.0, interval_change_flag=False)
        # Should be around 0.52 (1 - 12/25)
        assert 0.4 < q["missing_pct"] < 0.6

    def test_duplicates_counted(self):
        """Duplicates should be counted but not inflate unique timestamp count."""
        df = _make_slice_df("2023-01-01", periods=25, freq_minutes=60, dup_indices=[0, 1, 2])
        slice_start = pd.Timestamp("2023-01-01", tz="UTC")
        slice_end = pd.Timestamp("2023-01-01 23:59:59", tz="UTC")
        q = compute_quality(df, slice_start, slice_end, dt_minutes=60.0, interval_change_flag=False)
        assert q["duplicate_count"] == 3
        assert q["n_unique_timestamps"] == 25

    def test_nan_temps_counted(self):
        """NaN temps should be counted in nan_temp_count."""
        df = _make_slice_df("2023-01-01", periods=25, freq_minutes=60, nan_indices=[0, 5, 10])
        slice_start = pd.Timestamp("2023-01-01", tz="UTC")
        slice_end = pd.Timestamp("2023-01-01 23:59:59", tz="UTC")
        q = compute_quality(df, slice_start, slice_end, dt_minutes=60.0, interval_change_flag=False)
        assert q["nan_temp_count"] == 3
        assert q["n_records_with_temp"] == 22
