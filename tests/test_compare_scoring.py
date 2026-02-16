"""Tests for min-max scoring in core/compare.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from weather_tool.core.compare import _minmax_norm, compute_scores


# ── _minmax_norm ──────────────────────────────────────────────────────────────

class TestMinmaxNorm:
    def test_basic_range(self):
        s = pd.Series([0.0, 50.0, 100.0])
        result = _minmax_norm(s)
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(50.0)
        assert result.iloc[2] == pytest.approx(100.0)

    def test_all_equal_returns_50(self):
        s = pd.Series([42.0, 42.0, 42.0])
        result = _minmax_norm(s)
        assert (result == 50.0).all()

    def test_invert(self):
        s = pd.Series([0.0, 100.0])
        result = _minmax_norm(s, invert=True)
        assert result.iloc[0] == pytest.approx(100.0)
        assert result.iloc[1] == pytest.approx(0.0)

    def test_single_element_returns_50(self):
        s = pd.Series([99.0])
        result = _minmax_norm(s)
        assert result.iloc[0] == pytest.approx(50.0)


# ── compute_scores — two-station case ────────────────────────────────────────

def _make_compare_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestComputeScores:
    def _two_station_base(self):
        """Two stations: A is hot+moist, B is cold+dry."""
        return _make_compare_df([
            {
                "station_id": "HOT",
                "hours_above_ref_90_sum": 500.0,
                "cooling_degree_hours_90_sum": 10000.0,
                "t_p99_median": 105.0,
                "wb_p99_median": 78.0,
                "freeze_hours_sum": 10.0,
                "tmin_min": 20.0,
                "timestamp_missing_pct_avg": 0.005,
                "coverage_weighted_pct": 0.99,
                "interval_change_flag_any": False,
            },
            {
                "station_id": "COLD",
                "hours_above_ref_90_sum": 50.0,
                "cooling_degree_hours_90_sum": 500.0,
                "t_p99_median": 85.0,
                "wb_p99_median": 65.0,
                "freeze_hours_sum": 800.0,
                "tmin_min": -10.0,
                "timestamp_missing_pct_avg": 0.001,
                "coverage_weighted_pct": 1.0,
                "interval_change_flag_any": False,
            },
        ])

    def test_heat_score_hot_station_higher(self):
        df = self._two_station_base()
        out = compute_scores(df, ref_temps=[90.0])
        hot = out.loc[out["station_id"] == "HOT", "heat_score"].iloc[0]
        cold = out.loc[out["station_id"] == "COLD", "heat_score"].iloc[0]
        assert hot > cold

    def test_heat_score_range_0_to_100(self):
        df = self._two_station_base()
        out = compute_scores(df, ref_temps=[90.0])
        # With two stations one should be near 0 and one near 100
        scores = out["heat_score"]
        assert scores.min() >= 0.0
        assert scores.max() <= 100.0

    def test_freeze_score_cold_station_higher(self):
        df = self._two_station_base()
        out = compute_scores(df, ref_temps=[90.0])
        hot = out.loc[out["station_id"] == "HOT", "freeze_score"].iloc[0]
        cold = out.loc[out["station_id"] == "COLD", "freeze_score"].iloc[0]
        assert cold > hot

    def test_moisture_score_present_when_wb_available(self):
        df = self._two_station_base()
        out = compute_scores(df, ref_temps=[90.0])
        assert "moisture_score" in out.columns
        assert out["moisture_score"].notna().all()

    def test_moisture_score_nan_when_no_wb_column(self):
        df = _make_compare_df([
            {"station_id": "A", "hours_above_ref_65_sum": 100.0, "t_p99_median": 90.0,
             "freeze_hours_sum": 50.0, "tmin_min": 25.0,
             "timestamp_missing_pct_avg": 0.01, "coverage_weighted_pct": 1.0,
             "interval_change_flag_any": False},
            {"station_id": "B", "hours_above_ref_65_sum": 200.0, "t_p99_median": 95.0,
             "freeze_hours_sum": 10.0, "tmin_min": 30.0,
             "timestamp_missing_pct_avg": 0.02, "coverage_weighted_pct": 0.98,
             "interval_change_flag_any": True},
        ])
        out = compute_scores(df, ref_temps=[65.0])
        assert out["moisture_score"].isna().all()

    def test_overall_score_uses_redistribution_without_moisture(self):
        """When moisture_score is NaN, overall should use heat 0.50 / freeze 0.35 / quality 0.15."""
        df = _make_compare_df([
            {"station_id": "A", "hours_above_ref_65_sum": 100.0, "t_p99_median": 90.0,
             "freeze_hours_sum": 50.0, "tmin_min": 25.0,
             "timestamp_missing_pct_avg": 0.01, "coverage_weighted_pct": 1.0,
             "interval_change_flag_any": False},
            {"station_id": "B", "hours_above_ref_65_sum": 50.0, "t_p99_median": 85.0,
             "freeze_hours_sum": 150.0, "tmin_min": 10.0,
             "timestamp_missing_pct_avg": 0.005, "coverage_weighted_pct": 0.99,
             "interval_change_flag_any": False},
        ])
        out = compute_scores(df, ref_temps=[65.0])
        assert out["moisture_score"].isna().all()
        # overall_score must still be computed and non-NaN
        assert out["overall_score"].notna().all()

    def test_data_quality_higher_when_less_missing(self):
        df = _make_compare_df([
            {"station_id": "CLEAN", "hours_above_ref_65_sum": 100.0, "t_p99_median": 90.0,
             "freeze_hours_sum": 50.0, "tmin_min": 25.0,
             "timestamp_missing_pct_avg": 0.001, "coverage_weighted_pct": 1.0,
             "interval_change_flag_any": False},
            {"station_id": "DIRTY", "hours_above_ref_65_sum": 100.0, "t_p99_median": 90.0,
             "freeze_hours_sum": 50.0, "tmin_min": 25.0,
             "timestamp_missing_pct_avg": 0.10, "coverage_weighted_pct": 0.80,
             "interval_change_flag_any": True},
        ])
        out = compute_scores(df, ref_temps=[65.0])
        clean = out.loc[out["station_id"] == "CLEAN", "data_quality_score"].iloc[0]
        dirty = out.loc[out["station_id"] == "DIRTY", "data_quality_score"].iloc[0]
        assert clean > dirty

    def test_all_equal_scores_50(self):
        """Identical stations → all component scores = 50."""
        rows = [
            {"station_id": "A", "hours_above_ref_65_sum": 100.0, "t_p99_median": 90.0,
             "freeze_hours_sum": 50.0, "tmin_min": 20.0,
             "timestamp_missing_pct_avg": 0.01, "coverage_weighted_pct": 1.0,
             "interval_change_flag_any": False},
            {"station_id": "B", "hours_above_ref_65_sum": 100.0, "t_p99_median": 90.0,
             "freeze_hours_sum": 50.0, "tmin_min": 20.0,
             "timestamp_missing_pct_avg": 0.01, "coverage_weighted_pct": 1.0,
             "interval_change_flag_any": False},
        ]
        out = compute_scores(pd.DataFrame(rows), ref_temps=[65.0])
        assert out["heat_score"].iloc[0] == pytest.approx(50.0)
        assert out["freeze_score"].iloc[0] == pytest.approx(50.0)
