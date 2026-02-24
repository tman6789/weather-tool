"""Tests for compute_year_coverage() canonical hourly coverage."""
import math

import numpy as np
import pandas as pd
import pytest

from weather_tool.core.quality import compute_quality, compute_year_coverage


def _cov(timestamps, slice_start, slice_end, tz="UTC"):
    """Helper: wrap list/Series of timestamps + all-False is_dup."""
    ts = pd.Series(timestamps)
    dup = pd.Series([False] * len(ts))
    return compute_year_coverage(
        ts, dup,
        pd.Timestamp(slice_start, tz=tz),
        pd.Timestamp(slice_end, tz=tz),
    )


class TestComputeYearCoverage:

    def test_full_non_leap_year(self):
        """Full 2019 (non-leap): expected=8760, missing=0, pct=0.0."""
        ts = pd.date_range("2019-01-01", "2019-12-31 23:00", freq="1h", tz="UTC")
        r = _cov(ts, "2019-01-01", "2019-12-31 23:59:59")
        assert r["expected_hours"] == 8760
        assert r["observed_hours"] == 8760
        assert r["missing_hours"]  == 0
        assert r["missing_pct"]    == 0.0

    def test_full_leap_year(self):
        """Full 2020 (leap): expected=8784."""
        ts = pd.date_range("2020-01-01", "2020-12-31 23:00", freq="1h", tz="UTC")
        r = _cov(ts, "2020-01-01", "2020-12-31 23:59:59")
        assert r["expected_hours"] == 8784
        assert r["observed_hours"] == 8784
        assert r["missing_hours"]  == 0

    def test_partial_year_one_month(self):
        """June 2019 window: expected = 30*24 = 720."""
        ts = pd.date_range("2019-06-01", "2019-06-30 23:00", freq="1h", tz="UTC")
        r = _cov(ts, "2019-06-01", "2019-06-30 23:59:59")
        assert r["expected_hours"] == 720
        assert r["observed_hours"] == 720
        assert r["missing_hours"]  == 0
        assert r["missing_pct"]    == 0.0

    def test_missing_timestamps(self):
        """Drop first 100 hours → missing_hours=100, pct=100/8760."""
        ts_full = pd.date_range("2019-01-01", "2019-12-31 23:00", freq="1h", tz="UTC")
        ts = pd.Series(ts_full[100:])   # first 100 hours absent
        dup = pd.Series([False] * len(ts))
        r = compute_year_coverage(
            ts, dup,
            pd.Timestamp("2019-01-01", tz="UTC"),
            pd.Timestamp("2019-12-31 23:59:59", tz="UTC"),
        )
        assert r["expected_hours"] == 8760
        assert r["observed_hours"] == 8660
        assert r["missing_hours"]  == 100
        assert r["missing_pct"] == round(100 / 8760, 6)

    def test_sub_hourly_full_coverage(self):
        """20-min data covering a full day → 24 unique hourly buckets, missing=0."""
        ts = pd.date_range("2019-01-01", "2019-01-01 23:40", freq="20min", tz="UTC")
        r = _cov(ts, "2019-01-01", "2019-01-01 23:59:59")
        assert r["expected_hours"] == 24
        assert r["observed_hours"] == 24
        assert r["missing_hours"]  == 0
        assert r["missing_pct"]    == 0.0


class TestComputeQualityWithCoverage:

    def test_nan_temp_count_uses_dedup_rows(self):
        """nan_temp_count counts NaN temp in dedup rows only, not duplicate rows."""
        ts = pd.date_range("2023-01-01", periods=10, freq="1h", tz="UTC")
        temps = np.ones(10) * 70.0
        temps[0] = np.nan  # index 0 has NaN temp
        df = pd.DataFrame({"timestamp": ts, "temp": temps, "_is_dup": False})
        # Add 5 duplicate rows — all with NaN temp — should NOT be counted
        dup_rows = df.iloc[:5].copy()
        dup_rows["_is_dup"] = True
        dup_rows["temp"] = np.nan
        df = pd.concat([df, dup_rows], ignore_index=True)

        q = compute_quality(
            df,
            pd.Timestamp("2023-01-01", tz="UTC"),
            pd.Timestamp("2023-01-01 23:59:59", tz="UTC"),
            dt_minutes=60.0,
            interval_change_flag=False,
        )
        # Only the 1 dedup row has NaN temp, not the 5 dup rows
        assert q["nan_temp_count"] == 1

    def test_coverage_keys_present(self):
        """compute_quality emits expected_hours, observed_hours, missing_hours, missing_pct."""
        ts = pd.date_range("2023-01-01", periods=24, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": ts,
            "temp":      np.ones(24) * 70.0,
            "_is_dup":   False,
        })
        q = compute_quality(
            df,
            pd.Timestamp("2023-01-01", tz="UTC"),
            pd.Timestamp("2023-01-01 23:59:59", tz="UTC"),
            dt_minutes=60.0,
            interval_change_flag=False,
        )
        for key in ("expected_hours", "observed_hours", "missing_hours", "missing_pct"):
            assert key in q, f"key {key!r} missing from compute_quality output"
        assert q["expected_hours"] == 24
        assert q["missing_hours"]  == 0
