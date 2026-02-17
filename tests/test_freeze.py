"""Tests for freeze risk metrics (core/freeze.py)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from weather_tool.core.freeze import (
    compute_freeze_hours,
    compute_freeze_shoulder_hours,
    compute_total_hours_with_temp,
    detect_freeze_events,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ts_range(n: int, freq_minutes: int = 60, tz: str = "UTC") -> pd.Series:
    """Return a tz-aware timestamp Series with n observations at freq_minutes intervals."""
    idx = pd.date_range("2022-01-01", periods=n, freq=f"{freq_minutes}min", tz=tz)
    return pd.Series(idx)


def _ts_range_from(start: str, n: int, freq_minutes: int = 60, tz: str = "UTC") -> pd.Series:
    idx = pd.date_range(start, periods=n, freq=f"{freq_minutes}min", tz=tz)
    return pd.Series(idx)


# ── TestFreezeHours ────────────────────────────────────────────────────────────

class TestFreezeHours:
    def test_half_below_threshold(self):
        """48 hourly obs: 24 @ 30°F + 24 @ 40°F, threshold=32 → 24 hrs."""
        temps = pd.Series([30.0] * 24 + [40.0] * 24)
        result = compute_freeze_hours(temps, freeze_threshold_f=32.0, dt_minutes=60.0)
        assert result == pytest.approx(24.0)

    def test_all_above_returns_zero(self):
        """All temps above threshold → 0 hours."""
        temps = pd.Series([40.0, 50.0, 60.0])
        result = compute_freeze_hours(temps, freeze_threshold_f=32.0, dt_minutes=60.0)
        assert result == pytest.approx(0.0)

    def test_at_threshold_counts(self):
        """Temp exactly at 32°F should count (<=, inclusive)."""
        temps = pd.Series([32.0, 32.0, 32.0])
        result = compute_freeze_hours(temps, freeze_threshold_f=32.0, dt_minutes=60.0)
        assert result == pytest.approx(3.0)

    def test_nan_ignored(self):
        """NaN observations should not be counted as freeze hours."""
        temps = pd.Series([30.0, float("nan"), 30.0, float("nan")])
        result = compute_freeze_hours(temps, freeze_threshold_f=32.0, dt_minutes=60.0)
        assert result == pytest.approx(2.0)

    def test_invalid_dt_nan_returns_nan(self):
        """NaN dt_minutes → NaN result."""
        temps = pd.Series([30.0, 31.0])
        result = compute_freeze_hours(temps, freeze_threshold_f=32.0, dt_minutes=float("nan"))
        assert math.isnan(result)

    def test_invalid_dt_zero_returns_nan(self):
        """Zero dt_minutes → NaN result."""
        temps = pd.Series([30.0, 31.0])
        result = compute_freeze_hours(temps, freeze_threshold_f=32.0, dt_minutes=0.0)
        assert math.isnan(result)

    def test_invalid_dt_negative_returns_nan(self):
        """Negative dt_minutes → NaN result."""
        temps = pd.Series([30.0, 31.0])
        result = compute_freeze_hours(temps, freeze_threshold_f=32.0, dt_minutes=-60.0)
        assert math.isnan(result)

    def test_30min_interval(self):
        """Step-function: 4 obs at 30 min below threshold → 4 * 0.5 = 2.0 hrs."""
        temps = pd.Series([28.0, 29.0, 31.0, 30.0])
        result = compute_freeze_hours(temps, freeze_threshold_f=32.0, dt_minutes=30.0)
        assert result == pytest.approx(2.0)


# ── TestTotalHoursWithTemp ─────────────────────────────────────────────────────

class TestTotalHoursWithTemp:
    def test_all_valid(self):
        temps = pd.Series([30.0, 40.0, 50.0])
        result = compute_total_hours_with_temp(temps, dt_minutes=60.0)
        assert result == pytest.approx(3.0)

    def test_some_nan(self):
        temps = pd.Series([30.0, float("nan"), 50.0, float("nan")])
        result = compute_total_hours_with_temp(temps, dt_minutes=60.0)
        assert result == pytest.approx(2.0)

    def test_all_nan(self):
        temps = pd.Series([float("nan"), float("nan")])
        result = compute_total_hours_with_temp(temps, dt_minutes=60.0)
        assert result == pytest.approx(0.0)

    def test_invalid_dt_returns_nan(self):
        temps = pd.Series([30.0, 40.0])
        result = compute_total_hours_with_temp(temps, dt_minutes=float("nan"))
        assert math.isnan(result)


# ── TestFreezeShoulderHours ────────────────────────────────────────────────────

class TestFreezeShoulderHours:
    def test_shoulder_months_only(self):
        """12 months of hourly data (1 obs each), all at 30°F.
        Shoulder = [3, 4, 10, 11] → 4 freeze hours."""
        months = list(range(1, 13))
        timestamps = pd.Series(
            [pd.Timestamp(f"2022-{m:02d}-15 12:00:00", tz="UTC") for m in months]
        )
        temps = pd.Series([30.0] * 12)  # all below freeze
        result = compute_freeze_shoulder_hours(
            timestamps, temps,
            freeze_threshold_f=32.0,
            dt_minutes=60.0,
            shoulder_months=[3, 4, 10, 11],
        )
        assert result == pytest.approx(4.0)

    def test_non_shoulder_excluded(self):
        """Freeze obs in Jan and Jun should NOT appear in shoulder total."""
        timestamps = pd.Series([
            pd.Timestamp("2022-01-15 00:00:00", tz="UTC"),  # Jan — not shoulder
            pd.Timestamp("2022-06-15 00:00:00", tz="UTC"),  # Jun — not shoulder
            pd.Timestamp("2022-03-15 00:00:00", tz="UTC"),  # Mar — shoulder ✓
            pd.Timestamp("2022-11-15 00:00:00", tz="UTC"),  # Nov — shoulder ✓
        ])
        temps = pd.Series([28.0, 28.0, 28.0, 28.0])
        result = compute_freeze_shoulder_hours(
            timestamps, temps,
            freeze_threshold_f=32.0,
            dt_minutes=60.0,
            shoulder_months=[3, 4, 10, 11],
        )
        # Only Mar + Nov qualify → 2 hrs
        assert result == pytest.approx(2.0)

    def test_no_shoulder_obs(self):
        """All obs in non-shoulder months → 0 hrs."""
        timestamps = pd.Series([
            pd.Timestamp("2022-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2022-07-01 00:00:00", tz="UTC"),
        ])
        temps = pd.Series([20.0, 20.0])
        result = compute_freeze_shoulder_hours(
            timestamps, temps,
            freeze_threshold_f=32.0,
            dt_minutes=60.0,
            shoulder_months=[3, 4, 10, 11],
        )
        assert result == pytest.approx(0.0)

    def test_above_threshold_in_shoulder(self):
        """Shoulder-month obs above threshold do not count."""
        timestamps = pd.Series([
            pd.Timestamp("2022-03-15 00:00:00", tz="UTC"),  # Mar shoulder, but warm
            pd.Timestamp("2022-04-15 00:00:00", tz="UTC"),  # Apr shoulder, freeze
        ])
        temps = pd.Series([50.0, 28.0])
        result = compute_freeze_shoulder_hours(
            timestamps, temps,
            freeze_threshold_f=32.0,
            dt_minutes=60.0,
            shoulder_months=[3, 4, 10, 11],
        )
        assert result == pytest.approx(1.0)


# ── TestDetectFreezeEvents ─────────────────────────────────────────────────────

class TestDetectFreezeEvents:
    def test_single_event_5h(self):
        """5-hr continuous freeze run, min=3 → event_count=1, max_duration=5.0."""
        ts = _ts_range(5, freq_minutes=60)
        temps = pd.Series([28.0] * 5)
        result = detect_freeze_events(ts, temps, 32.0, 60.0, min_event_hours=3.0, gap_tolerance_mult=1.5)
        assert result["freeze_event_count"] == 1
        assert result["freeze_event_max_duration_hours"] == pytest.approx(5.0)

    def test_short_run_excluded(self):
        """2-hr run with min=3 → event_count=0, max_duration=NaN."""
        ts = _ts_range(2, freq_minutes=60)
        temps = pd.Series([28.0, 29.0])
        result = detect_freeze_events(ts, temps, 32.0, 60.0, min_event_hours=3.0, gap_tolerance_mult=1.5)
        assert result["freeze_event_count"] == 0
        assert math.isnan(result["freeze_event_max_duration_hours"])

    def test_gap_breaks_run(self):
        """5h freeze, then large gap (3h), then 5h freeze → event_count=2."""
        # First 5 obs: hourly, freeze
        ts1 = pd.date_range("2022-01-01 00:00", periods=5, freq="60min", tz="UTC")
        # Second 5 obs: start after a 3-hour gap (4h from last obs of ts1)
        ts2 = pd.date_range("2022-01-01 09:00", periods=5, freq="60min", tz="UTC")
        ts = pd.Series(list(ts1) + list(ts2))
        temps = pd.Series([28.0] * 10)
        # gap between ts1[-1]=05:00 and ts2[0]=09:00 → 4h gap > 1.5 * 60min = 90min → breaks run
        result = detect_freeze_events(ts, temps, 32.0, 60.0, min_event_hours=3.0, gap_tolerance_mult=1.5)
        assert result["freeze_event_count"] == 2
        assert result["freeze_event_max_duration_hours"] == pytest.approx(5.0)

    def test_no_gap_break_within_limit(self):
        """Gap < 1.5*dt_minutes → continues same run."""
        # 3 hourly obs, then 80-minute gap (< 90min = 1.5 * 60min), then 3 more
        ts1 = pd.date_range("2022-01-01 00:00", periods=3, freq="60min", tz="UTC")
        ts2 = pd.date_range("2022-01-01 04:20", periods=3, freq="60min", tz="UTC")
        # Gap: ts1[-1]=02:00, ts2[0]=04:20 → 140 min gap > 90min → WILL break
        # Use a tighter gap: ts2 starts at 03:20 → gap=80min < 90min
        ts2 = pd.date_range("2022-01-01 03:20", periods=3, freq="60min", tz="UTC")
        ts = pd.Series(list(ts1) + list(ts2))
        temps = pd.Series([28.0] * 6)
        result = detect_freeze_events(ts, temps, 32.0, 60.0, min_event_hours=3.0, gap_tolerance_mult=1.5)
        # 80-min gap < 90-min → same event; total run = 6 * 1h = 6h (but step-function uses run_count)
        # run_count counts freeze obs, not gap time; 6 obs → 6 * (60/60) = 6h
        assert result["freeze_event_count"] == 1
        assert result["freeze_event_max_duration_hours"] == pytest.approx(6.0)

    def test_nan_breaks_event(self):
        """3h freeze + NaN obs + 3h freeze → event_count=2 (NaN forces break)."""
        ts = _ts_range(7, freq_minutes=60)
        temps = pd.Series([28.0, 29.0, 30.0, float("nan"), 28.0, 29.0, 30.0])
        result = detect_freeze_events(ts, temps, 32.0, 60.0, min_event_hours=3.0, gap_tolerance_mult=1.5)
        assert result["freeze_event_count"] == 2
        assert result["freeze_event_max_duration_hours"] == pytest.approx(3.0)

    def test_above_threshold_breaks_run(self):
        """Non-freeze obs between two freeze runs → 2 separate events."""
        ts = _ts_range(7, freq_minutes=60)
        temps = pd.Series([28.0, 29.0, 30.0, 40.0, 28.0, 29.0, 30.0])
        result = detect_freeze_events(ts, temps, 32.0, 60.0, min_event_hours=3.0, gap_tolerance_mult=1.5)
        assert result["freeze_event_count"] == 2

    def test_empty_series_returns_zero(self):
        ts = pd.Series([], dtype="datetime64[ns, UTC]")
        temps = pd.Series([], dtype=float)
        result = detect_freeze_events(ts, temps, 32.0, 60.0, min_event_hours=3.0, gap_tolerance_mult=1.5)
        assert result["freeze_event_count"] == 0
        assert math.isnan(result["freeze_event_max_duration_hours"])

    def test_invalid_dt_returns_zero_count(self):
        ts = _ts_range(3, freq_minutes=60)
        temps = pd.Series([28.0, 29.0, 30.0])
        result = detect_freeze_events(ts, temps, 32.0, float("nan"), min_event_hours=3.0, gap_tolerance_mult=1.5)
        assert result["freeze_event_count"] == 0
        assert math.isnan(result["freeze_event_max_duration_hours"])

    def test_exactly_at_min_event_hours_qualifies(self):
        """Event exactly at min_event_hours threshold should qualify (>=)."""
        ts = _ts_range(3, freq_minutes=60)  # 3 obs → 3 hrs
        temps = pd.Series([28.0, 29.0, 30.0])
        result = detect_freeze_events(ts, temps, 32.0, 60.0, min_event_hours=3.0, gap_tolerance_mult=1.5)
        assert result["freeze_event_count"] == 1
        assert result["freeze_event_max_duration_hours"] == pytest.approx(3.0)

    def test_30min_interval_event(self):
        """30-min interval: 8 obs = 4h event, min=3 → qualifies."""
        ts = _ts_range(8, freq_minutes=30)
        temps = pd.Series([28.0] * 8)
        result = detect_freeze_events(ts, temps, 32.0, 30.0, min_event_hours=3.0, gap_tolerance_mult=1.5)
        assert result["freeze_event_count"] == 1
        assert result["freeze_event_max_duration_hours"] == pytest.approx(4.0)


# ── TestFreezeCompareAggregation ───────────────────────────────────────────────
# These tests exercise the aggregation logic that compare.py applies across years.
# We replicate the math here without importing compare.py, keeping tests fast & isolated.

class TestFreezeCompareAggregation:
    """Verify the aggregation rules documented in the plan:
    freeze_hours       → sum
    freeze_hours_shoulder → sum
    freeze_event_count → sum
    freeze_event_max_duration_hours → max of yearly maxima
    freeze_hours_pct_over_window → freeze_hours_sum / total_hours_with_temp_sum
    """

    def test_freeze_hours_sum(self):
        """Two years: 100 + 200 = 300 hrs."""
        summary = pd.DataFrame({"freeze_hours": [100.0, 200.0]})
        result = round(float(summary["freeze_hours"].dropna().sum()), 2)
        assert result == pytest.approx(300.0)

    def test_freeze_hours_shoulder_sum(self):
        """Two years: 30 + 70 = 100 hrs."""
        summary = pd.DataFrame({"freeze_hours_shoulder": [30.0, 70.0]})
        result = round(float(summary["freeze_hours_shoulder"].dropna().sum()), 2)
        assert result == pytest.approx(100.0)

    def test_event_count_sum(self):
        """Two years: 3 + 5 = 8 total events."""
        summary = pd.DataFrame({"freeze_event_count": [3, 5]})
        result = int(summary["freeze_event_count"].fillna(0).sum())
        assert result == 8

    def test_event_max_duration_max(self):
        """Two years: max(4.0, 6.0) = 6.0."""
        summary = pd.DataFrame({"freeze_event_max_duration_hours": [4.0, 6.0]})
        valid = summary["freeze_event_max_duration_hours"].dropna()
        result = float(valid.max())
        assert result == pytest.approx(6.0)

    def test_event_max_duration_nan_ignored(self):
        """Year with no events (NaN) → max from valid year."""
        summary = pd.DataFrame({"freeze_event_max_duration_hours": [float("nan"), 6.0]})
        valid = summary["freeze_event_max_duration_hours"].dropna()
        result = float(valid.max()) if len(valid) > 0 else float("nan")
        assert result == pytest.approx(6.0)

    def test_pct_over_window(self):
        """300 freeze hrs / 2400 total hrs = 0.125."""
        frz_sum = 300.0
        tht_sum = 2400.0
        result = round(frz_sum / tht_sum, 6)
        assert result == pytest.approx(0.125)

    def test_zero_denom_pct_is_nan(self):
        """total_hours_with_temp_sum == 0 → freeze_hours_pct_over_window is NaN."""
        frz_sum = 0.0
        tht_sum = 0.0
        if not math.isnan(frz_sum) and tht_sum > 0:
            result = round(frz_sum / tht_sum, 6)
        else:
            result = float("nan")
        assert math.isnan(result)

    def test_freeze_hours_sum_with_nan_year(self):
        """Year with NaN freeze hours is excluded from sum."""
        summary = pd.DataFrame({"freeze_hours": [100.0, float("nan"), 200.0]})
        result = round(float(summary["freeze_hours"].dropna().sum()), 2)
        assert result == pytest.approx(300.0)

    def test_event_count_sum_with_nan(self):
        """NaN event count treated as 0 in sum."""
        summary = pd.DataFrame({"freeze_event_count": [3, float("nan"), 5]})
        result = int(summary["freeze_event_count"].fillna(0).sum())
        assert result == 8
