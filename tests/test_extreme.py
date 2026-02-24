"""Tests for the extreme value analysis module (core/extreme.py)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from weather_tool.core.extreme import (
    compute_design_day,
    compute_exceedance_hours,
    compute_extreme_yearly,
    compute_rolling_max,
    compute_rolling_window_max,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_ts(n_hours: int, dt_minutes: float = 60.0) -> pd.Series:
    """Return a UTC timestamp series."""
    ts = pd.date_range("2022-07-01", periods=n_hours, freq=f"{int(dt_minutes)}min", tz="UTC")
    return pd.Series(pd.DatetimeIndex(ts))


# ── compute_rolling_max ─────────────────────────────────────────────────────


class TestComputeRollingMax:
    def test_constant_series(self):
        """Constant 78°F over 120h → rolling 24h max = 78."""
        ts = _make_ts(120)
        vals = pd.Series([78.0] * 120)
        result = compute_rolling_max(vals, ts, window_hours=24, dt_minutes=60.0)
        assert result == pytest.approx(78.0)

    def test_peak_block_detected(self):
        """50h at 70 + 30h at 90 + 40h at 70 → 24h max ≈ 90."""
        ts = _make_ts(120)
        vals = pd.Series([70.0] * 50 + [90.0] * 30 + [70.0] * 40)
        result = compute_rolling_max(vals, ts, window_hours=24, dt_minutes=60.0)
        assert result == pytest.approx(90.0)

    def test_72h_window(self):
        ts = _make_ts(200)
        vals = pd.Series([75.0] * 200)
        result = compute_rolling_max(vals, ts, window_hours=72, dt_minutes=60.0)
        assert result == pytest.approx(75.0)

    def test_below_completeness_returns_nan(self):
        """10 obs in a 72h window that needs 0.80×72 = ~57 periods → NaN."""
        ts = _make_ts(10)
        vals = pd.Series([78.0] * 10)
        result = compute_rolling_max(vals, ts, window_hours=72, dt_minutes=60.0)
        assert math.isnan(result)

    def test_invalid_dt_nan(self):
        ts = _make_ts(100)
        vals = pd.Series([78.0] * 100)
        assert math.isnan(compute_rolling_max(vals, ts, 24, float("nan")))
        assert math.isnan(compute_rolling_max(vals, ts, 24, 0.0))
        assert math.isnan(compute_rolling_max(vals, ts, 24, -1.0))

    def test_empty_series(self):
        ts = pd.Series(dtype="datetime64[ns, UTC]")
        vals = pd.Series(dtype=float)
        assert math.isnan(compute_rolling_max(vals, ts, 24, 60.0))

    def test_all_nan_returns_nan(self):
        ts = _make_ts(48)
        vals = pd.Series([float("nan")] * 48)
        assert math.isnan(compute_rolling_max(vals, ts, 24, 60.0))


# ── compute_exceedance_hours ────────────────────────────────────────────────


class TestComputeExceedanceHours:
    def test_all_above(self):
        """24 obs at 100°F, threshold 95 → 24 × 1h = 24.0."""
        vals = pd.Series([100.0] * 24)
        assert compute_exceedance_hours(vals, 95.0, 60.0) == pytest.approx(24.0)

    def test_none_above(self):
        vals = pd.Series([80.0] * 24)
        assert compute_exceedance_hours(vals, 95.0, 60.0) == 0.0

    def test_nan_excluded(self):
        vals = pd.Series([100.0, float("nan"), 100.0])
        assert compute_exceedance_hours(vals, 95.0, 60.0) == pytest.approx(2.0)

    def test_invalid_dt_returns_zero(self):
        vals = pd.Series([100.0] * 10)
        assert compute_exceedance_hours(vals, 95.0, float("nan")) == 0.0
        assert compute_exceedance_hours(vals, 95.0, 0.0) == 0.0

    def test_nan_threshold_returns_nan(self):
        vals = pd.Series([100.0] * 10)
        assert math.isnan(compute_exceedance_hours(vals, float("nan"), 60.0))

    def test_30min_interval(self):
        """4 obs above threshold at 30min → 4 × 0.5 = 2.0 hrs."""
        vals = pd.Series([100.0] * 4 + [50.0] * 4)
        assert compute_exceedance_hours(vals, 95.0, 30.0) == pytest.approx(2.0)

    def test_at_threshold_counts(self):
        """Exactly at threshold (>=) should count."""
        vals = pd.Series([95.0] * 10)
        assert compute_exceedance_hours(vals, 95.0, 60.0) == pytest.approx(10.0)


# ── compute_extreme_yearly ──────────────────────────────────────────────────


class TestComputeExtremeYearly:
    def _make_dedup(self, n_hours: int = 200) -> pd.DataFrame:
        ts = _make_ts(n_hours)
        return pd.DataFrame({
            "timestamp": ts,
            "temp": [80.0 + 10 * np.sin(2 * np.pi * i / 24) for i in range(n_hours)],
            "wetbulb_f": [70.0 + 5 * np.sin(2 * np.pi * i / 24) for i in range(n_hours)],
        })

    def test_returns_all_keys(self):
        dedup = self._make_dedup()
        result = compute_extreme_yearly(dedup, 60.0, t_p99=88.0, wb_p99=74.0)
        expected_keys = {
            "tdb_mean_24h_max", "tdb_mean_72h_max",
            "exceedance_hours_tdb_p99", "exceedance_hours_twb_p99",
        }
        assert expected_keys == set(result.keys())

    def test_no_wetbulb_gives_nan_twb(self):
        dedup = self._make_dedup()
        dedup = dedup.drop(columns=["wetbulb_f"])
        result = compute_extreme_yearly(dedup, 60.0, t_p99=88.0, wb_p99=74.0)
        assert math.isnan(result["exceedance_hours_twb_p99"])
        assert not math.isnan(result["tdb_mean_24h_max"])

    def test_nan_t_p99_gives_nan_exceedance(self):
        dedup = self._make_dedup()
        result = compute_extreme_yearly(dedup, 60.0, t_p99=float("nan"), wb_p99=74.0)
        assert math.isnan(result["exceedance_hours_tdb_p99"])

    def test_wb_p99_none_gives_nan(self):
        dedup = self._make_dedup()
        result = compute_extreme_yearly(dedup, 60.0, t_p99=88.0, wb_p99=None)
        assert math.isnan(result["exceedance_hours_twb_p99"])

    def test_tdb_persistence_reasonable(self):
        """Sinusoidal temp with mean ~80: 24h rolling max should be close to mean."""
        dedup = self._make_dedup(200)
        result = compute_extreme_yearly(dedup, 60.0, t_p99=88.0, wb_p99=74.0)
        assert 75.0 < result["tdb_mean_24h_max"] < 90.0


# ── compute_design_day ──────────────────────────────────────────────────────


class TestComputeDesignDay:
    def _make_df(self, n_hours: int = 200) -> tuple[pd.DataFrame, pd.Series]:
        ts = _make_ts(n_hours)
        df = pd.DataFrame({
            "timestamp": ts,
            "temp": [80.0 + 10 * np.sin(2 * np.pi * i / 24) for i in range(n_hours)],
            "wetbulb_f": [70.0 + 5 * np.sin(2 * np.pi * i / 24) for i in range(n_hours)],
            "relh": [60.0] * n_hours,
        })
        return df, ts

    def test_returns_24_rows(self):
        df, ts = self._make_df()
        result = compute_design_day(df, ts, 60.0, metric="temp")
        assert len(result) == 24
        assert list(result["hour"]) == list(range(24))

    def test_expected_columns(self):
        df, ts = self._make_df()
        result = compute_design_day(df, ts, 60.0, metric="temp")
        expected = {"hour", "tdb", "twb", "rh", "wind_speed_kt", "wind_dir_deg"}
        assert set(result.columns) == expected

    def test_missing_columns_are_nan(self):
        """Wind columns not present → NaN in output."""
        df, ts = self._make_df()
        result = compute_design_day(df, ts, 60.0, metric="temp")
        assert result["wind_speed_kt"].isna().all()
        assert result["wind_dir_deg"].isna().all()

    def test_insufficient_data_returns_empty(self):
        """Very short series → empty DataFrame."""
        ts = _make_ts(3)
        df = pd.DataFrame({"timestamp": ts, "temp": [80.0, 81.0, 82.0]})
        result = compute_design_day(df, ts, 60.0, metric="temp")
        assert result.empty

    def test_fallback_to_temp_when_metric_missing(self):
        """If metric='wetbulb_f' but column absent, fall back to temp."""
        ts = _make_ts(200)
        df = pd.DataFrame({
            "timestamp": ts,
            "temp": [80.0] * 200,
        })
        result = compute_design_day(df, ts, 60.0, metric="wetbulb_f")
        assert len(result) == 24
        assert result["tdb"].notna().any()

    def test_invalid_dt_returns_empty(self):
        df, ts = self._make_df()
        assert compute_design_day(df, ts, float("nan"), metric="temp").empty
        assert compute_design_day(df, ts, 0.0, metric="temp").empty

    def test_20min_data_resampled_to_24_rows(self):
        """Non-hourly data (20min) still produces 24 hourly rows."""
        ts = _make_ts(500, dt_minutes=20.0)
        df = pd.DataFrame({
            "timestamp": ts,
            "temp": [80.0 + 10 * np.sin(2 * np.pi * i / 72) for i in range(500)],
        })
        result = compute_design_day(df, ts, 20.0, metric="temp")
        assert len(result) == 24

    def test_hottest_block_selected(self):
        """Inject a known hot block — design day tdb values should reflect it."""
        ts = _make_ts(200)
        temps = [70.0] * 100 + [100.0] * 30 + [70.0] * 70
        df = pd.DataFrame({"timestamp": ts, "temp": temps})
        result = compute_design_day(df, ts, 60.0, metric="temp")
        assert len(result) == 24
        # The hottest block is 100°F, so max tdb should be close to 100
        assert result["tdb"].max() >= 95.0


# ── Econ tower still works after rolling move ───────────────────────────────


class TestEconTowerStillWorks:
    """After moving rolling logic, econ_tower produces correct wb rolling maxima."""

    def test_wb_rolling_values_unchanged(self):
        from weather_tool.core.econ_tower import compute_econ_tower_yearly

        n = 200
        ts = _make_ts(n)
        dedup = pd.DataFrame({
            "timestamp": ts,
            "temp": [70.0] * n,
            "wetbulb_f": [65.0] * n,
        })
        result = compute_econ_tower_yearly(
            dedup=dedup,
            dt_minutes=60.0,
            air_econ_threshold_f=55.0,
            chw_supply_f=44.0,
            tower_approach_f=7.0,
            hx_approach_f=5.0,
            wb_stress_thresholds=[75.0, 78.0],
        )
        assert result["wb_mean_24h_max"] == pytest.approx(65.0)
        assert result["wb_mean_72h_max"] == pytest.approx(65.0)


# ── Extreme columns in aggregate ────────────────────────────────────────────


class TestExtremeInAggregate:
    """build_yearly_summary includes the new extreme columns."""

    def test_extreme_columns_present(self):
        from weather_tool.config import RunConfig
        from weather_tool.core.aggregate import build_yearly_summary

        n = 200
        ts = pd.date_range("2022-07-01", periods=n, freq="60min", tz="UTC")
        df = pd.DataFrame({
            "timestamp": ts,
            "temp": [80.0 + 10 * np.sin(2 * np.pi * i / 24) for i in range(n)],
            "_is_dup": [False] * n,
        })
        cfg = RunConfig(
            mode="csv",
            input_path=None,
            station_id="TEST",
            start=ts[0].date(),
            end=ts[-1].date(),
            ref_temp=65.0,
        )
        interval_info = {
            "dt_minutes": 60.0,
            "p10": 60.0,
            "p90": 60.0,
            "interval_change_flag": False,
            "unique_diff_counts": {60: n - 1},
            "interval_unknown_flag": False,
        }
        summary = build_yearly_summary(df, cfg, interval_info)
        assert "tdb_mean_24h_max" in summary.columns
        assert "tdb_mean_72h_max" in summary.columns
        assert "exceedance_hours_tdb_p99" in summary.columns
        # Values should be non-NaN for a good dataset
        assert summary["tdb_mean_24h_max"].notna().any()


# ── compute_rolling_window_max ───────────────────────────────────────────────


class TestComputeRollingWindowMax:
    def test_peak_captured(self):
        """Peak value inside a 6h window is returned even when most hours are lower."""
        ts   = pd.Series(pd.date_range("2020-07-01", periods=24, freq="1h", tz="UTC"))
        vals = pd.Series([70.0] * 12 + [80.0] + [70.0] * 11)
        result = compute_rolling_window_max(vals, ts, window_hours=6, dt_minutes=60.0)
        assert result == 80.0

    def test_insufficient_data_returns_nan(self):
        """Returns NaN when obs count is below min_periods in every window."""
        ts   = pd.Series(pd.date_range("2020-07-01", periods=3, freq="1h", tz="UTC"))
        vals = pd.Series([70.0, 72.0, 74.0])
        # 6h window → n_steps=6; min_periods=int(0.8*6)=4; only 3 obs → all NaN
        result = compute_rolling_window_max(vals, ts, window_hours=6, dt_minutes=60.0)
        assert math.isnan(result)

    def test_result_geq_rolling_mean(self):
        """Rolling window max >= rolling window mean for the same window / data."""
        ts   = pd.Series(pd.date_range("2020-07-01", periods=48, freq="1h", tz="UTC"))
        vals = pd.Series([70.0 + i * 0.5 for i in range(48)])
        mean_result = compute_rolling_max(vals, ts, window_hours=6, dt_minutes=60.0)
        max_result  = compute_rolling_window_max(vals, ts, window_hours=6, dt_minutes=60.0)
        assert not math.isnan(max_result)
        assert max_result >= mean_result - 0.01  # max always >= mean


class TestWb6hRollmaxInAggregate:
    def test_column_present_with_wetbulb(self):
        """wb_6h_rollmax_max_f appears in yearly summary when wet-bulb data is present
        and is <= wb_max (point max is the upper bound for any window max)."""
        from datetime import date

        import numpy as np

        from weather_tool.config import RunConfig
        from weather_tool.core.aggregate import build_yearly_summary

        cfg = RunConfig(
            mode="csv",
            station_id="TEST",
            start=date(2020, 1, 1),
            end=date(2020, 12, 31),
            ref_temp=65.0,
            fields=["tmpf", "dwpf", "relh"],
        )

        n    = 8760
        ts   = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
        tmpf = 80.0 + 10.0 * np.sin(np.arange(n) * 2 * np.pi / (365 * 24))
        relh = np.full(n, 60.0)
        # Stull (2011) wet-bulb in Celsius, then convert to °F
        t_c = (tmpf - 32.0) * 5.0 / 9.0
        rh  = relh
        twb_c = (
            t_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
            + np.arctan(t_c + rh)
            - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
            - 4.686035
        )
        wb = twb_c * 9.0 / 5.0 + 32.0

        df = pd.DataFrame(
            {
                "timestamp":  ts,
                "station_id": "TEST",
                "temp":       tmpf,
                "tmpf":       tmpf,
                "relh":       relh,
                "wetbulb_f":  wb,
                "_is_dup":    False,
            }
        )

        interval_info = {
            "dt_minutes": 60.0,
            "p10": 60.0,
            "p90": 60.0,
            "interval_change_flag": False,
            "unique_diff_counts": {60: n - 1},
            "interval_unknown_flag": False,
        }
        summary = build_yearly_summary(df, cfg, interval_info)

        assert "wb_6h_rollmax_max_f" in summary.columns, "column missing from summary"
        val    = summary.loc[summary["year"] == 2020, "wb_6h_rollmax_max_f"].iloc[0]
        wb_max = summary.loc[summary["year"] == 2020, "wb_max"].iloc[0]
        assert not math.isnan(val), f"expected non-NaN wb_6h_rollmax_max_f, got {val}"
        assert val <= wb_max + 0.1, (
            f"6h rollmax {val} implausibly exceeds point max {wb_max}"
        )
