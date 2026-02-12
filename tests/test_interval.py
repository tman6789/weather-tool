"""Tests for sampling interval inference."""

import pandas as pd
import pytest

from weather_tool.core.metrics import infer_interval


def _ts_series(start: str, periods: int, freq_minutes: int) -> pd.Series:
    """Helper: generate a tz-aware timestamp series."""
    return pd.Series(
        pd.date_range(start, periods=periods, freq=f"{freq_minutes}min", tz="UTC")
    )


class TestInferInterval:
    def test_20min_interval(self):
        ts = _ts_series("2023-07-01", periods=100, freq_minutes=20)
        info = infer_interval(ts)
        assert info["dt_minutes"] == pytest.approx(20.0)
        assert info["interval_change_flag"] is False

    def test_60min_interval(self):
        ts = _ts_series("2023-01-01", periods=200, freq_minutes=60)
        info = infer_interval(ts)
        assert info["dt_minutes"] == pytest.approx(60.0)
        assert info["interval_change_flag"] is False

    def test_mixed_intervals_triggers_flag(self):
        """Half 20-min, half 60-min → interval_change_flag should be True."""
        ts1 = _ts_series("2023-01-01", periods=50, freq_minutes=20)
        ts2 = _ts_series("2023-06-01", periods=50, freq_minutes=60)
        combined = pd.concat([ts1, ts2]).sort_values().reset_index(drop=True)
        info = infer_interval(combined)
        assert info["interval_change_flag"] is True

    def test_single_timestamp(self):
        ts = pd.Series(pd.to_datetime(["2023-01-01"], utc=True))
        info = infer_interval(ts)
        assert pd.isna(info["dt_minutes"])

    def test_empty_series(self):
        ts = pd.Series(dtype="datetime64[ns, UTC]")
        info = infer_interval(ts)
        assert pd.isna(info["dt_minutes"])

    def test_diagnostics_present(self):
        ts = _ts_series("2023-01-01", periods=50, freq_minutes=15)
        info = infer_interval(ts)
        assert "p10" in info
        assert "p90" in info
        assert "unique_diff_counts" in info
        assert 15.0 in info["unique_diff_counts"]
