"""Tests for economizer and cooling tower decision metrics (core/econ_tower.py)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from weather_tool.core.econ_tower import (
    compute_air_econ_hours,
    compute_lwt_proxy_metrics,
    compute_rolling_wb_max,
    compute_tower_stress_hours,
    compute_wec_hours,
)


# ── 7.1 Airside economizer hours ──────────────────────────────────────────────

class TestAirEconHours:
    def test_half_below_threshold(self):
        """48h: 24 at 50°F (≤55) + 24 at 70°F (>55) → 24 hrs."""
        temps = pd.Series([50.0] * 24 + [70.0] * 24)
        result = compute_air_econ_hours(temps, threshold_f=55.0, dt_minutes=60.0)
        assert result == pytest.approx(24.0)

    def test_all_above_threshold_returns_zero(self):
        temps = pd.Series([70.0] * 24)
        result = compute_air_econ_hours(temps, threshold_f=55.0, dt_minutes=60.0)
        assert result == 0.0

    def test_at_threshold_counts(self):
        """Exactly at threshold (<=) should count."""
        temps = pd.Series([55.0] * 12)
        result = compute_air_econ_hours(temps, threshold_f=55.0, dt_minutes=60.0)
        assert result == pytest.approx(12.0)

    def test_nan_rows_ignored(self):
        temps = pd.Series([50.0, float("nan"), 50.0])
        result = compute_air_econ_hours(temps, threshold_f=55.0, dt_minutes=60.0)
        assert result == pytest.approx(2.0)

    def test_invalid_dt_returns_nan(self):
        temps = pd.Series([50.0] * 10)
        assert math.isnan(compute_air_econ_hours(temps, 55.0, float("nan")))
        assert math.isnan(compute_air_econ_hours(temps, 55.0, 0.0))
        assert math.isnan(compute_air_econ_hours(temps, 55.0, -1.0))

    def test_30min_interval(self):
        """dt=30min, 4 obs below threshold → 4 * 0.5h = 2.0 hrs."""
        temps = pd.Series([50.0] * 4)
        result = compute_air_econ_hours(temps, threshold_f=55.0, dt_minutes=30.0)
        assert result == pytest.approx(2.0)


# ── 7.2 WEC hours ─────────────────────────────────────────────────────────────

class TestWecHours:
    def test_required_twb_max_correct(self):
        """required_twb_max = supply - tower_approach - hx_approach."""
        wb = pd.Series([30.0] * 10)
        req, _, _ = compute_wec_hours(wb, chw_supply_f=44.0, tower_approach_f=7.0, hx_approach_f=5.0, dt_minutes=60.0)
        assert req == pytest.approx(32.0)  # 44 - 7 - 5

    def test_all_below_required(self):
        wb = pd.Series([25.0] * 24)
        _, wec_hrs, _ = compute_wec_hours(wb, 44.0, 7.0, 5.0, 60.0)
        assert wec_hrs == pytest.approx(24.0)

    def test_none_below_required(self):
        wb = pd.Series([40.0] * 24)
        _, wec_hrs, _ = compute_wec_hours(wb, 44.0, 7.0, 5.0, 60.0)
        assert wec_hrs == 0.0

    def test_feasibility_pct_correct(self):
        """12 of 24 observations below required → pct = 0.5."""
        wb = pd.Series([25.0] * 12 + [40.0] * 12)
        _, wec_hrs, wec_pct = compute_wec_hours(wb, 44.0, 7.0, 5.0, 60.0)
        assert wec_hrs == pytest.approx(12.0)
        assert wec_pct == pytest.approx(0.5)

    def test_all_nan_wb_returns_nan(self):
        wb = pd.Series([float("nan")] * 10)
        _, wec_hrs, wec_pct = compute_wec_hours(wb, 44.0, 7.0, 5.0, 60.0)
        assert math.isnan(wec_hrs)
        assert math.isnan(wec_pct)


# ── 7.3 Tower stress thresholds ───────────────────────────────────────────────

class TestTowerStressHours:
    def _wb_series(self) -> pd.Series:
        """10h at 70, 5h at 76, 3h at 79, 2h at 81."""
        return pd.Series([70.0] * 10 + [76.0] * 5 + [79.0] * 3 + [81.0] * 2)

    def test_threshold_75(self):
        wb = self._wb_series()
        result = compute_tower_stress_hours(wb, [75.0], dt_minutes=60.0)
        # 76 (5) + 79 (3) + 81 (2) = 10 hrs
        assert result["tower_stress_hours_wb_gt_75"] == pytest.approx(10.0)

    def test_threshold_78(self):
        wb = self._wb_series()
        result = compute_tower_stress_hours(wb, [78.0], dt_minutes=60.0)
        # 79 (3) + 81 (2) = 5 hrs
        assert result["tower_stress_hours_wb_gt_78"] == pytest.approx(5.0)

    def test_threshold_80(self):
        wb = self._wb_series()
        result = compute_tower_stress_hours(wb, [80.0], dt_minutes=60.0)
        # 81 (2) = 2 hrs
        assert result["tower_stress_hours_wb_gt_80"] == pytest.approx(2.0)

    def test_multiple_thresholds_returns_all_keys(self):
        wb = self._wb_series()
        result = compute_tower_stress_hours(wb, [75.0, 78.0, 80.0], dt_minutes=60.0)
        assert set(result.keys()) == {
            "tower_stress_hours_wb_gt_75",
            "tower_stress_hours_wb_gt_78",
            "tower_stress_hours_wb_gt_80",
        }

    def test_nan_ignored(self):
        wb = pd.Series([76.0, float("nan"), 76.0])
        result = compute_tower_stress_hours(wb, [75.0], dt_minutes=60.0)
        assert result["tower_stress_hours_wb_gt_75"] == pytest.approx(2.0)

    def test_empty_series_returns_nan(self):
        """Empty wb series (no wetbulb column) must return NaN, not 0."""
        wb = pd.Series([], dtype=float)
        result = compute_tower_stress_hours(wb, [75.0, 78.0], dt_minutes=60.0)
        assert math.isnan(result["tower_stress_hours_wb_gt_75"])
        assert math.isnan(result["tower_stress_hours_wb_gt_78"])


# ── 7.4 Rolling wetbulb maxima ────────────────────────────────────────────────

def _make_wb_ts(n_hours: int, wb_val: float, dt_minutes: float = 60.0) -> tuple[pd.Series, pd.Series]:
    """Return (timestamps, wb_series) for a uniform hourly series."""
    ts = pd.date_range("2022-07-01", periods=n_hours, freq=f"{int(dt_minutes)}min", tz="UTC")
    return pd.Series(pd.DatetimeIndex(ts)), pd.Series([wb_val] * n_hours)


class TestRollingWbMax:
    def test_constant_72h(self):
        """120h constant 78°F series: rolling 72h max = 78."""
        ts, wb = _make_wb_ts(120, 78.0)
        result = compute_rolling_wb_max(wb, ts, window_hours=72, dt_minutes=60.0)
        assert result == pytest.approx(78.0)

    def test_constant_24h(self):
        ts, wb = _make_wb_ts(48, 75.0)
        result = compute_rolling_wb_max(wb, ts, window_hours=24, dt_minutes=60.0)
        assert result == pytest.approx(75.0)

    def test_below_completeness_returns_nan(self):
        """10 obs in a 72h window that needs 0.80×72 = 57 periods → NaN."""
        ts = pd.Series(pd.date_range("2022-07-01", periods=10, freq="60min", tz="UTC"))
        wb = pd.Series([78.0] * 10)
        result = compute_rolling_wb_max(wb, ts, window_hours=72, dt_minutes=60.0)
        assert math.isnan(result)

    def test_invalid_dt_returns_nan(self):
        ts, wb = _make_wb_ts(100, 78.0)
        assert math.isnan(compute_rolling_wb_max(wb, ts, 24, float("nan")))
        assert math.isnan(compute_rolling_wb_max(wb, ts, 24, 0.0))

    def test_all_nan_wb_returns_nan(self):
        ts, _ = _make_wb_ts(48, 78.0)
        wb = pd.Series([float("nan")] * 48)
        result = compute_rolling_wb_max(wb, ts, window_hours=24, dt_minutes=60.0)
        assert math.isnan(result)


# ── 7.5 LWT proxy metrics ─────────────────────────────────────────────────────

class TestLwtProxy:
    def test_p99_correct(self):
        wb = pd.Series([float(x) for x in range(1, 101)])  # 1–100
        result = compute_lwt_proxy_metrics(wb, tower_approach_f=7.0)
        expected_p99 = float(np.percentile(wb + 7.0, 99))
        assert result["lwt_proxy_p99"] == pytest.approx(expected_p99, rel=1e-3)

    def test_max_correct(self):
        wb = pd.Series([70.0, 75.0, 80.0])
        result = compute_lwt_proxy_metrics(wb, tower_approach_f=7.0)
        assert result["lwt_proxy_max"] == pytest.approx(87.0)

    def test_all_nan_returns_nan(self):
        wb = pd.Series([float("nan")] * 10)
        result = compute_lwt_proxy_metrics(wb, tower_approach_f=7.0)
        assert math.isnan(result["lwt_proxy_p99"])
        assert math.isnan(result["lwt_proxy_max"])

    def test_single_value(self):
        wb = pd.Series([72.0])
        result = compute_lwt_proxy_metrics(wb, tower_approach_f=7.0)
        assert result["lwt_proxy_p99"] == pytest.approx(79.0)
        assert result["lwt_proxy_max"] == pytest.approx(79.0)
