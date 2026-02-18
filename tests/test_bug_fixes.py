"""Regression tests for bug fixes and structural improvements.

Bug 1: compare.py uses per-year dt for mixed-interval stations.
Bug 2: compare.py returns sentinel on empty summary (no crash).
Bug 3: quality.py field_missing_pct uses dedup rows only.
Bug 4: wind event_count correctly counts disjoint events when min_event_hours == 0.
Bug 5: relh path in compute_wetbulb_f clamps to [0, 100].
S1:    detect_exceedance_runs event_mask excludes NaN-gap rows.
S2:    wind_gap_tolerance_mult is a separate config field from freeze.
S3:    Econ confidence uses actual window hours (leap-year safe).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from weather_tool.config import RunConfig
from weather_tool.core.compare import aggregate_station_window
from weather_tool.core.metrics import compute_wetbulb_f
from weather_tool.core.quality import compute_quality
from weather_tool.core.wind import (
    compute_event_wind_stats,
    detect_exceedance_runs,
    normalize_wind,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_windowed_multi_year(
    year_configs: list[tuple[int, int, float]],
    temp: float = 80.0,
) -> pd.DataFrame:
    """Build a windowed DataFrame spanning multiple years with different intervals.

    Parameters
    ----------
    year_configs : list of (year, periods, freq_minutes)
    temp : uniform temperature for all observations
    """
    frames = []
    for year, periods, freq_minutes in year_configs:
        ts = pd.date_range(
            f"{year}-01-01", periods=periods, freq=f"{int(freq_minutes)}min", tz="UTC"
        )
        frames.append(pd.DataFrame({"timestamp": ts, "temp": [temp] * periods}))
    df = pd.concat(frames, ignore_index=True)
    df["_is_dup"] = False
    return df


def _make_summary_with_dt(
    years: list[int],
    dt_minutes: list[float],
    hours_above_ref: list[float] | None = None,
) -> pd.DataFrame:
    """Build a summary DataFrame with explicit per-year dt_minutes."""
    n = len(years)
    return pd.DataFrame({
        "year": years,
        "dt_minutes": dt_minutes,
        "tmax": [80.0] * n,
        "tmin": [80.0] * n,
        "hours_above_ref": hours_above_ref or [0.0] * n,
        "missing_pct": [0.0] * n,
        "coverage_pct": [1.0] * n,
        "partial_coverage_flag": [False] * n,
        "interval_change_flag": [True] * n,
    })


def _interval_info(dt: float = 60.0, interval_change_flag: bool = False) -> dict:
    return {
        "dt_minutes": dt,
        "p10": dt,
        "p90": dt,
        "interval_change_flag": interval_change_flag,
        "unique_diff_counts": {dt: 1},
    }


# ── Bug 1: Per-year dt in compare aggregate ───────────────────────────────────


class TestMixedIntervalHourCounts:
    """aggregate_station_window must use per-year dt from summary, not global dt."""

    def test_mixed_60_and_20_min_hours_above_ref(self):
        """Two years: 2022@60min (10 obs), 2023@20min (30 obs), all temps=90 > ref=65.

        Correct: 2022 = 10*(60/60) = 10h, 2023 = 30*(20/60) = 10h → total = 20h.
        Bug (global dt=60): 2022 = 10h correct, 2023 = 30*(60/60) = 30h → total = 40h.
        """
        windowed = _make_windowed_multi_year([
            (2022, 10, 60),
            (2023, 30, 20),
        ], temp=90.0)
        summary = _make_summary_with_dt(
            years=[2022, 2023],
            dt_minutes=[60.0, 20.0],
        )
        interval = _interval_info(dt=60.0, interval_change_flag=True)
        row = aggregate_station_window(summary, windowed, interval, [65.0], "KTEST")

        assert row["hours_above_ref_65_sum"] == pytest.approx(20.0)

    def test_cdh_uses_per_year_dt(self):
        """CDH must also use per-year dt.

        2022: 10 obs × excess(90-65=25) × (60/60) = 250
        2023: 30 obs × excess(90-65=25) × (20/60) = 250
        Total = 500.
        """
        windowed = _make_windowed_multi_year([
            (2022, 10, 60),
            (2023, 30, 20),
        ], temp=90.0)
        summary = _make_summary_with_dt(
            years=[2022, 2023],
            dt_minutes=[60.0, 20.0],
        )
        interval = _interval_info(dt=60.0, interval_change_flag=True)
        row = aggregate_station_window(summary, windowed, interval, [65.0], "KTEST")

        assert row["cooling_degree_hours_65_sum"] == pytest.approx(500.0)

    def test_uniform_interval_unaffected(self):
        """Sanity: uniform 60-min station still works correctly."""
        windowed = _make_windowed_multi_year([
            (2022, 24, 60),
            (2023, 24, 60),
        ], temp=80.0)
        summary = _make_summary_with_dt(
            years=[2022, 2023],
            dt_minutes=[60.0, 60.0],
        )
        interval = _interval_info(dt=60.0)
        row = aggregate_station_window(summary, windowed, interval, [65.0], "KTEST")
        # 48 obs × (60/60) = 48 hours
        assert row["hours_above_ref_65_sum"] == pytest.approx(48.0)

    def test_multiple_ref_temps_per_year_dt(self):
        """Per-year dt applied for every ref_temp."""
        windowed = _make_windowed_multi_year([
            (2022, 10, 60),
            (2023, 30, 20),
        ], temp=90.0)
        summary = _make_summary_with_dt(
            years=[2022, 2023],
            dt_minutes=[60.0, 20.0],
        )
        interval = _interval_info(dt=60.0, interval_change_flag=True)
        row = aggregate_station_window(summary, windowed, interval, [65.0, 80.0, 95.0], "KTEST")

        # All temps=90: above 65 and 80, but NOT above 95
        assert row["hours_above_ref_65_sum"] == pytest.approx(20.0)
        assert row["hours_above_ref_80_sum"] == pytest.approx(20.0)
        assert row["hours_above_ref_95_sum"] == pytest.approx(0.0)


# ── Bug 2: Empty summary guard ────────────────────────────────────────────────


class TestEmptySummaryGuard:
    """aggregate_station_window must not crash when summary has 0 rows."""

    def _empty_result(self, ref_temps: list[float] | None = None) -> dict:
        ref_temps = ref_temps or [65.0]
        summary = pd.DataFrame([])
        windowed = pd.DataFrame({
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "temp": pd.Series(dtype=float),
            "_is_dup": pd.Series(dtype=bool),
        })
        return aggregate_station_window(
            summary, windowed, _interval_info(), ref_temps, "KEMPTY"
        )

    def test_returns_dict_without_crash(self):
        result = self._empty_result()
        assert isinstance(result, dict)

    def test_station_id_preserved(self):
        result = self._empty_result()
        assert result["station_id"] == "KEMPTY"

    def test_years_covered_is_zero(self):
        result = self._empty_result()
        assert result["years_covered_count"] == 0

    def test_hours_above_ref_is_zero(self):
        result = self._empty_result([65.0, 80.0])
        assert result["hours_above_ref_65_sum"] == pytest.approx(0.0)
        assert result["hours_above_ref_80_sum"] == pytest.approx(0.0)

    def test_cdh_is_zero(self):
        result = self._empty_result([65.0])
        assert result["cooling_degree_hours_65_sum"] == pytest.approx(0.0)

    def test_nan_for_averages(self):
        result = self._empty_result()
        assert math.isnan(result["dt_minutes_median"])
        assert math.isnan(result["timestamp_missing_pct_avg"])
        assert math.isnan(result["coverage_weighted_pct"])

    def test_all_ref_temps_present(self):
        ref_temps = [55.0, 65.0, 75.0, 85.0]
        result = self._empty_result(ref_temps)
        for rt in ref_temps:
            key = f"hours_above_ref_{int(rt)}_sum"
            assert key in result
            assert result[key] == pytest.approx(0.0)


# ── Bug 3: Per-field missing pct dedup filter ─────────────────────────────────


class TestFieldMissingPctDedup:
    """compute_quality field_missing_pct must use deduplicated rows only."""

    def test_dup_nan_does_not_inflate_nan_count(self):
        """Duplicate row with NaN relh must not inflate nan_count_relh.

        10 unique rows with valid relh + 1 dup of row 0 with NaN relh.
        Expected nan_count_relh = 0.
        """
        ts = pd.date_range("2023-07-01", periods=10, freq="60min", tz="UTC")
        df = pd.DataFrame({"timestamp": ts, "temp": [75.0] * 10, "relh": [50.0] * 10})
        df["_is_dup"] = False

        dup = df.iloc[[0]].copy()
        dup["_is_dup"] = True
        dup["relh"] = float("nan")
        df = pd.concat([df, dup], ignore_index=True)

        q = compute_quality(
            df,
            pd.Timestamp("2023-07-01", tz="UTC"),
            pd.Timestamp("2023-07-01 09:59:59", tz="UTC"),
            dt_minutes=60.0,
            interval_change_flag=False,
            extra_fields=["relh"],
        )
        assert q["nan_count_relh"] == 0
        assert q["field_missing_pct_relh"] == pytest.approx(0.0)

    def test_many_dups_cannot_push_pct_above_one(self):
        """Multiple dup rows with NaN must not push field_missing_pct above 1.0."""
        n = 5
        ts = pd.date_range("2023-07-01", periods=n, freq="60min", tz="UTC")
        df = pd.DataFrame({"timestamp": ts, "temp": [75.0] * n, "relh": [50.0] * n})
        df["_is_dup"] = False

        # 10 duplicate rows of first timestamp, all with NaN relh
        dups = pd.concat(
            [df.iloc[[0]].assign(**{"_is_dup": True, "relh": float("nan")})] * 10,
            ignore_index=True,
        )
        df = pd.concat([df, dups], ignore_index=True)

        q = compute_quality(
            df,
            pd.Timestamp("2023-07-01", tz="UTC"),
            pd.Timestamp("2023-07-01 04:59:59", tz="UTC"),
            dt_minutes=60.0,
            interval_change_flag=False,
            extra_fields=["relh"],
        )
        assert q["field_missing_pct_relh"] <= 1.0

    def test_genuine_nan_in_unique_row_still_counted(self):
        """NaN in a unique (non-dup) row must still be counted."""
        n = 10
        ts = pd.date_range("2023-07-01", periods=n, freq="60min", tz="UTC")
        relh_vals = [50.0] * n
        relh_vals[3] = float("nan")
        relh_vals[7] = float("nan")

        df = pd.DataFrame({"timestamp": ts, "temp": [75.0] * n, "relh": relh_vals})
        df["_is_dup"] = False

        # Also add a dup with NaN — should NOT be counted
        dup = df.iloc[[0]].copy()
        dup["_is_dup"] = True
        dup["relh"] = float("nan")
        df = pd.concat([df, dup], ignore_index=True)

        q = compute_quality(
            df,
            pd.Timestamp("2023-07-01", tz="UTC"),
            pd.Timestamp("2023-07-01 09:59:59", tz="UTC"),
            dt_minutes=60.0,
            interval_change_flag=False,
            extra_fields=["relh"],
        )
        assert q["nan_count_relh"] == 2
        assert q["field_missing_pct_relh"] == pytest.approx(2 / 10, rel=1e-5)


# ── Bug 4: Wind event_count with disjoint events ─────────────────────────────


def _wind_df_with_metric(
    metric_values: list[float],
    freq_minutes: int = 60,
) -> pd.DataFrame:
    """Build a minimal wind DataFrame with timestamp, drct, sknt, and a metric column."""
    n = len(metric_values)
    ts = pd.date_range("2022-07-01", periods=n, freq=f"{freq_minutes}min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "drct": [180.0] * n,
        "sknt": [10.0] * n,
        "temp": metric_values,
    })
    return normalize_wind(df)


class TestWindEventCount:
    """compute_event_wind_stats must count disjoint events correctly."""

    def test_two_disjoint_events_min_hours_zero(self):
        """Two exceedance blocks separated by a gap → event_count = 2."""
        # 3 hot rows, 4 cool rows, 3 hot rows
        temps = [100.0] * 3 + [50.0] * 4 + [100.0] * 3
        df = _wind_df_with_metric(temps, freq_minutes=60)
        result = compute_event_wind_stats(
            df, "temp", 90.0, 60.0,
            min_event_hours=0.0, gap_tolerance_mult=1.5,
        )
        assert result["event_count"] == 2

    def test_single_event_min_hours_zero(self):
        """One contiguous exceedance block → event_count = 1."""
        temps = [100.0] * 5 + [50.0] * 5
        df = _wind_df_with_metric(temps, freq_minutes=60)
        result = compute_event_wind_stats(
            df, "temp", 90.0, 60.0,
            min_event_hours=0.0, gap_tolerance_mult=1.5,
        )
        assert result["event_count"] == 1

    def test_no_exceedance_min_hours_zero(self):
        """No rows above threshold → event_count = 0."""
        temps = [50.0] * 10
        df = _wind_df_with_metric(temps, freq_minutes=60)
        result = compute_event_wind_stats(
            df, "temp", 90.0, 60.0,
            min_event_hours=0.0, gap_tolerance_mult=1.5,
        )
        assert result["event_count"] == 0

    def test_event_hours_total_correct(self):
        """Total event hours must reflect actual exceedance count × dt."""
        # 3 hot + 4 cool + 3 hot = 6 exceedance rows × 1h = 6h
        temps = [100.0] * 3 + [50.0] * 4 + [100.0] * 3
        df = _wind_df_with_metric(temps, freq_minutes=60)
        result = compute_event_wind_stats(
            df, "temp", 90.0, 60.0,
            min_event_hours=0.0, gap_tolerance_mult=1.5,
        )
        assert result["event_hours_total"] == pytest.approx(6.0)


# ── Bug 5: RH clamping in compute_wetbulb_f ──────────────────────────────────


class TestRHClamping:
    """Direct relh path must clamp to [0, 100] before Stull computation."""

    def test_relh_above_100_clamped(self):
        """relh=110 must be clamped; resulting wet-bulb must be ≤ dry-bulb."""
        df = pd.DataFrame({"tmpf": [86.0], "relh": [110.0]})
        result = compute_wetbulb_f(df)
        assert result.iloc[0] <= 86.0 + 0.01  # physics: wb ≤ db

    def test_relh_below_0_clamped(self):
        """relh=-5 must be clamped to 0; result should not be NaN."""
        df = pd.DataFrame({"tmpf": [86.0], "relh": [-5.0]})
        result = compute_wetbulb_f(df)
        assert not pd.isna(result.iloc[0])
        assert result.iloc[0] <= 86.0 + 0.01

    def test_relh_50_unchanged(self):
        """Normal relh=50 must produce same result as before (no regression)."""
        df = pd.DataFrame({"tmpf": [86.0], "relh": [50.0]})
        result = compute_wetbulb_f(df)
        assert result.iloc[0] == pytest.approx(72.3, abs=2.0)

    def test_relh_100_wb_equals_db(self):
        """At 100% RH, wet-bulb should equal dry-bulb (within tolerance)."""
        df = pd.DataFrame({"tmpf": [86.0], "relh": [100.0]})
        result = compute_wetbulb_f(df)
        assert result.iloc[0] == pytest.approx(86.0, abs=0.5)


# ── S1: event_mask excludes NaN-gap rows ──────────────────────────────────────


class TestEventMaskNaNGap:
    """detect_exceedance_runs must NOT mark NaN-gap rows as True in event_mask."""

    def test_nan_gap_rows_not_in_mask(self):
        """A NaN row between two exceedance rows breaks the run and is not masked.

        [100, NaN, 100] with gap_mult=1.5, dt=60 → the NaN at index 1 must be False.
        """
        ts = pd.Series(pd.date_range("2022-07-01", periods=3, freq="60min", tz="UTC"))
        metric = pd.Series([100.0, float("nan"), 100.0])
        events, mask = detect_exceedance_runs(ts, metric, 90.0, 60.0, 0.0, 1.5)
        # NaN breaks run → two events of 1h each (both qualify when min_hours=0)
        assert len(events) == 2
        # The NaN row (index 1) must NOT be marked True
        assert mask.iloc[1] == False  # noqa: E712

    def test_qualifying_rows_are_in_mask(self):
        """Exceedance rows that qualify must be True."""
        ts = pd.Series(pd.date_range("2022-07-01", periods=5, freq="60min", tz="UTC"))
        metric = pd.Series([100.0, 100.0, 100.0, 50.0, 50.0])
        events, mask = detect_exceedance_runs(ts, metric, 90.0, 60.0, 0.0, 1.5)
        assert mask.iloc[0] == True  # noqa: E712
        assert mask.iloc[1] == True  # noqa: E712
        assert mask.iloc[2] == True  # noqa: E712
        assert mask.iloc[3] == False  # noqa: E712


# ── S2: wind_gap_tolerance_mult config field ──────────────────────────────────


class TestWindGapToleranceConfig:
    """wind_gap_tolerance_mult must be an independent config field."""

    def test_default_value(self):
        cfg = RunConfig()
        assert cfg.wind_gap_tolerance_mult == 1.5

    def test_independent_from_freeze(self):
        """Changing freeze_gap_tolerance_mult must not affect wind_gap_tolerance_mult."""
        cfg = RunConfig(freeze_gap_tolerance_mult=3.0)
        assert cfg.wind_gap_tolerance_mult == 1.5

    def test_both_can_be_set_independently(self):
        cfg = RunConfig(freeze_gap_tolerance_mult=3.0, wind_gap_tolerance_mult=2.0)
        assert cfg.freeze_gap_tolerance_mult == 3.0
        assert cfg.wind_gap_tolerance_mult == 2.0


# ── S3: Leap-year econ confidence ─────────────────────────────────────────────


class TestLeapYearEconConfidence:
    """Econ confidence must use actual window hours, not 8760 × years."""

    def _make_summary_with_window(
        self,
        years: list[int],
        window_starts: list[str],
        window_ends: list[str],
        hours_with_wetbulb: list[float],
    ) -> pd.DataFrame:
        n = len(years)
        return pd.DataFrame({
            "year": years,
            "dt_minutes": [60.0] * n,
            "tmax": [90.0] * n,
            "tmin": [30.0] * n,
            "hours_above_ref": [100.0] * n,
            "missing_pct": [0.0] * n,
            "coverage_pct": [1.0] * n,
            "partial_coverage_flag": [False] * n,
            "interval_change_flag": [False] * n,
            "hours_with_wetbulb": hours_with_wetbulb,
            "window_start": window_starts,
            "window_end": window_ends,
            "wetbulb_availability_pct": [99.0] * n,
        })

    def test_leap_year_window_hours(self):
        """A 2020 (leap year) full-year window has 8784 hours, not 8760.

        If hours_with_wetbulb = 7900, coverage = 1.0:
        - Old bug: 8760 * 1.0 = 8760 → 7900/8760 = 0.902 (above 0.90 threshold, no flag)
        - Correct: 8784 * 1.0 = 8784 → 7900/8784 = 0.899 (below 0.90, flag fires)
        """
        summary = self._make_summary_with_window(
            years=[2020],
            window_starts=["2020-01-01"],
            window_ends=["2020-12-31"],
            hours_with_wetbulb=[7900.0],
        )
        windowed = _make_windowed_multi_year([(2020, 100, 60)], temp=80.0)
        interval = _interval_info(dt=60.0)
        row = aggregate_station_window(summary, windowed, interval, [65.0], "KTEST")
        # With 8784 actual hours and 7900 wb hours: 7900/8784 ≈ 0.8994 < 0.90
        assert row["econ_confidence_flag"] is True

    def test_non_leap_year_window(self):
        """A 2021 (non-leap) full-year window has 8760 hours."""
        summary = self._make_summary_with_window(
            years=[2021],
            window_starts=["2021-01-01"],
            window_ends=["2021-12-31"],
            hours_with_wetbulb=[8000.0],
        )
        windowed = _make_windowed_multi_year([(2021, 100, 60)], temp=80.0)
        interval = _interval_info(dt=60.0)
        row = aggregate_station_window(summary, windowed, interval, [65.0], "KTEST")
        # 8000/8760 ≈ 0.913 > 0.90 → no low_wb_coverage flag
        # But missing_pct=0 and wetbulb_availability=99% → econ_confidence_flag should be False
        assert row["econ_confidence_flag"] is False


# ── Empty-summary stations must not receive scores ───────────────────────────


class TestEmptySummaryScoring:
    """Stations with years_covered_count == 0 must get NaN scores and True flags."""

    def test_sentinel_flags_are_true(self):
        """Empty-summary sentinel must set all confidence/warning flags to True."""
        summary = pd.DataFrame(columns=[
            "year", "dt_minutes", "tmax", "tmin", "hours_above_ref",
            "missing_pct", "coverage_pct", "partial_coverage_flag",
            "interval_change_flag",
        ])
        windowed = pd.DataFrame({
            "timestamp": pd.DatetimeIndex([], dtype="datetime64[ns]"),
            "temp": pd.Series([], dtype=float),
            "_is_dup": pd.Series([], dtype=bool),
        })
        interval = _interval_info(dt=60.0)
        row = aggregate_station_window(summary, windowed, interval, [65.0], "KEMPTY")

        assert row["missing_data_warning"] is True
        assert row["freeze_confidence_flag"] is True
        assert row["econ_confidence_flag"] is True

    def test_scores_are_nan_for_no_data_station(self):
        """compute_scores must set all scores to NaN when years_covered_count == 0."""
        from weather_tool.core.compare_scores import compute_scores

        df = pd.DataFrame([
            {
                "station_id": "KEMPTY",
                "years_covered_count": 0,
                "hours_above_ref_65_sum": 0.0,
                "cooling_degree_hours_65_sum": 0.0,
                "freeze_hours_sum": 0.0,
                "tmin_min": float("nan"),
                "timestamp_missing_pct_avg": float("nan"),
                "coverage_weighted_pct": float("nan"),
                "interval_change_flag_any": False,
                "missing_data_warning": True,
            },
            {
                "station_id": "KREAL",
                "years_covered_count": 2,
                "hours_above_ref_65_sum": 500.0,
                "cooling_degree_hours_65_sum": 1000.0,
                "freeze_hours_sum": 200.0,
                "tmin_min": 10.0,
                "timestamp_missing_pct_avg": 0.01,
                "coverage_weighted_pct": 0.99,
                "interval_change_flag_any": False,
                "missing_data_warning": False,
            },
        ])
        scored = compute_scores(df, [65.0])

        # Empty station: all scores NaN
        empty_row = scored[scored["station_id"] == "KEMPTY"].iloc[0]
        assert math.isnan(empty_row["heat_score"])
        assert math.isnan(empty_row["freeze_score"])
        assert math.isnan(empty_row["data_quality_score"])
        assert math.isnan(empty_row["overall_score"])

        # Real station: scores are numeric (not NaN)
        real_row = scored[scored["station_id"] == "KREAL"].iloc[0]
        assert not math.isnan(real_row["heat_score"])
        assert not math.isnan(real_row["overall_score"])
