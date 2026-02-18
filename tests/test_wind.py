"""Tests for wind analytics (core/wind.py)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from weather_tool.core.wind import (
    CALM_THRESHOLD_MPH,
    KT_TO_MPH,
    compute_event_wind_stats,
    compute_sector_deltas,
    detect_exceedance_runs,
    direction_bin_index,
    normalize_wind,
    prevailing_sector,
    resolve_threshold,
    sector_labels,
    wind_rose_table,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _wind_df(
    n: int,
    drct: float | list[float] | None = 180.0,
    sknt: float | list[float] | None = 10.0,
    freq_minutes: int = 60,
    tz: str = "UTC",
    temp: float | list[float] | None = None,
    wetbulb_f: float | list[float] | None = None,
) -> pd.DataFrame:
    """Build a synthetic DataFrame with wind columns."""
    ts = pd.date_range("2022-07-01", periods=n, freq=f"{freq_minutes}min", tz=tz)
    data: dict = {"timestamp": ts}

    if drct is not None:
        data["drct"] = [drct] * n if isinstance(drct, (int, float)) else drct
    if sknt is not None:
        data["sknt"] = [sknt] * n if isinstance(sknt, (int, float)) else sknt
    if temp is not None:
        data["temp"] = [temp] * n if isinstance(temp, (int, float)) else temp
    if wetbulb_f is not None:
        data["wetbulb_f"] = [wetbulb_f] * n if isinstance(wetbulb_f, (int, float)) else wetbulb_f

    return pd.DataFrame(data)


# ── TestNormalizeWind ──────────────────────────────────────────────────────────

class TestNormalizeWind:
    def test_adds_columns(self):
        """normalize_wind adds drct_deg, wind_speed_kt, wind_speed_mph, is_calm."""
        df = _wind_df(3, drct=180.0, sknt=10.0)
        out = normalize_wind(df)
        for col in ("drct_deg", "wind_speed_kt", "wind_speed_mph", "is_calm"):
            assert col in out.columns

    def test_speed_conversion(self):
        """10 kt → 11.5078 mph."""
        df = _wind_df(1, sknt=10.0)
        out = normalize_wind(df)
        assert out["wind_speed_mph"].iloc[0] == pytest.approx(10.0 * KT_TO_MPH, rel=1e-4)

    def test_calm_detection(self):
        """Very low speed (0.3 mph equivalent) → is_calm=True."""
        low_kt = 0.3 / KT_TO_MPH  # ~0.26 kt → 0.3 mph
        df = _wind_df(1, sknt=low_kt)
        out = normalize_wind(df)
        assert out["is_calm"].iloc[0] is True or out["is_calm"].iloc[0] == True

    def test_nan_speed_is_not_calm(self):
        """NaN speed → is_calm must be False (unknown, not calm)."""
        df = _wind_df(1, sknt=None)
        df["sknt"] = [float("nan")]
        out = normalize_wind(df)
        assert out["is_calm"].iloc[0] == False

    def test_missing_drct_col(self):
        """No drct column → drct_deg all NaN."""
        df = _wind_df(3, drct=None, sknt=10.0)
        out = normalize_wind(df)
        assert out["drct_deg"].isna().all()

    def test_missing_sknt_col(self):
        """No sknt column → speed NaN, is_calm all False."""
        df = _wind_df(3, drct=180.0, sknt=None)
        out = normalize_wind(df)
        assert out["wind_speed_kt"].isna().all()
        assert out["wind_speed_mph"].isna().all()
        assert (out["is_calm"] == False).all()

    def test_direction_wrapping_360(self):
        """360° → 0° (North)."""
        df = _wind_df(1, drct=360.0)
        out = normalize_wind(df)
        assert out["drct_deg"].iloc[0] == pytest.approx(0.0)

    def test_invalid_drct_becomes_nan(self):
        """Direction < 0 or > 360 → NaN."""
        df = _wind_df(3, drct=[-5.0, 400.0, 180.0])
        out = normalize_wind(df)
        assert pd.isna(out["drct_deg"].iloc[0])
        assert pd.isna(out["drct_deg"].iloc[1])
        assert out["drct_deg"].iloc[2] == pytest.approx(180.0)

    def test_preserves_other_columns(self):
        """Original columns are preserved."""
        df = _wind_df(2, drct=90.0, sknt=5.0, temp=72.0)
        out = normalize_wind(df)
        assert "temp" in out.columns
        assert "timestamp" in out.columns


# ── TestDirectionBinIndex ──────────────────────────────────────────────────────

class TestDirectionBinIndex:
    def test_north_is_bin_zero(self):
        assert direction_bin_index(0.0, 16) == pytest.approx(0.0)

    def test_south_is_bin_8(self):
        assert direction_bin_index(180.0, 16) == pytest.approx(8.0)

    def test_east_is_bin_4(self):
        assert direction_bin_index(90.0, 16) == pytest.approx(4.0)

    def test_boundary_near_north(self):
        """348.75° is on the boundary of N sector for 16 bins; should still be N (bin 0)."""
        # With centered binning: (348.75 + 11.25) / 22.5 = 16.0, %16 = 0
        assert direction_bin_index(348.75, 16) == pytest.approx(0.0)

    def test_wrapping_360(self):
        """360° wraps to 0°, bin 0."""
        assert direction_bin_index(360.0, 16) == pytest.approx(0.0)

    def test_vectorized(self):
        """Array input produces array output."""
        arr = np.array([0.0, 90.0, 180.0, 270.0])
        result = direction_bin_index(arr, 16)
        np.testing.assert_array_almost_equal(result, [0, 4, 8, 12])


# ── TestSectorLabels ──────────────────────────────────────────────────────────

class TestSectorLabels:
    def test_16_bins(self):
        labels = sector_labels(16)
        assert len(labels) == 16
        assert labels[0] == "N"
        assert labels[8] == "S"

    def test_8_bins(self):
        labels = sector_labels(8)
        assert len(labels) == 8
        assert labels[0] == "N"
        assert labels[4] == "S"

    def test_generic_bins(self):
        labels = sector_labels(4)
        assert labels == ["0", "1", "2", "3"]


# ── TestWindRoseTable ─────────────────────────────────────────────────────────

class TestWindRoseTable:
    def test_uniform_south_wind(self):
        """All obs are 10kt from South → all hours in S sector."""
        df = _wind_df(24, drct=180.0, sknt=10.0)
        df = normalize_wind(df)
        rose, meta = wind_rose_table(df, dt_minutes=60.0, dir_bins=16)
        s_total = rose.loc["S"].sum()
        assert s_total == pytest.approx(24.0)
        assert meta["calm_hours"] == pytest.approx(0.0)

    def test_all_calm(self):
        """All obs are calm → all hours in Calm row."""
        calm_kt = 0.1 / KT_TO_MPH  # well below calm threshold
        df = _wind_df(10, drct=90.0, sknt=calm_kt)
        df = normalize_wind(df)
        rose, meta = wind_rose_table(df, dt_minutes=60.0, dir_bins=16)
        assert meta["calm_hours"] == pytest.approx(10.0)
        # No hours in directional sectors
        sector_only = rose.drop(index="Calm")
        assert sector_only.sum().sum() == pytest.approx(0.0)

    def test_slice_mask_filters(self):
        """Slice mask limits observations used."""
        df = _wind_df(10, drct=180.0, sknt=10.0)
        df = normalize_wind(df)
        # Only use first 5 obs
        mask = pd.Series([True] * 5 + [False] * 5)
        rose, meta = wind_rose_table(df, dt_minutes=60.0, dir_bins=16, slice_mask=mask)
        assert meta["total_valid_hours"] == pytest.approx(5.0)

    def test_unknown_direction_tracked(self):
        """Valid speed + NaN direction → counted in unknown_dir_hours."""
        df = _wind_df(6, drct=[float("nan")] * 3 + [180.0] * 3, sknt=10.0)
        df = normalize_wind(df)
        rose, meta = wind_rose_table(df, dt_minutes=60.0, dir_bins=16)
        assert meta["unknown_dir_hours"] == pytest.approx(3.0)
        assert meta["unknown_dir_pct"] > 0

    def test_hours_sum_matches(self):
        """Total hours in matrix + calm + unknown_dir = total_valid_hours."""
        df = _wind_df(20, drct=[float("nan")] * 5 + [90.0] * 10 + [180.0] * 5, sknt=10.0)
        df = normalize_wind(df)
        rose, meta = wind_rose_table(df, dt_minutes=60.0, dir_bins=16)
        matrix_total = rose.sum().sum()
        # matrix includes calm row; unknown dir not in matrix
        assert (matrix_total + meta["unknown_dir_hours"]) == pytest.approx(meta["total_valid_hours"])


# ── TestPrevailingSector ──────────────────────────────────────────────────────

class TestPrevailingSector:
    def test_simple_case(self):
        """All wind from South → prevailing sector is S."""
        df = _wind_df(24, drct=180.0, sknt=10.0)
        df = normalize_wind(df)
        rose, _ = wind_rose_table(df, dt_minutes=60.0, dir_bins=16)
        assert prevailing_sector(rose) == "S"


# ── TestResolveThreshold ──────────────────────────────────────────────────────

class TestResolveThreshold:
    def test_p99(self):
        s = pd.Series(range(100))
        result = resolve_threshold("p99", s)
        assert result == pytest.approx(np.nanpercentile(s.values, 99))

    def test_numeric_string(self):
        s = pd.Series([50.0])
        result = resolve_threshold("80", s)
        assert result == pytest.approx(80.0)

    def test_empty_series_returns_nan(self):
        s = pd.Series([], dtype=float)
        result = resolve_threshold("p99", s)
        assert math.isnan(result)


# ── TestDetectExceedanceRuns ──────────────────────────────────────────────────

class TestDetectExceedanceRuns:
    def _ts(self, n, freq=60):
        return pd.Series(pd.date_range("2022-07-01", periods=n, freq=f"{freq}min", tz="UTC"))

    def test_single_event(self):
        """5-hr exceedance run, min=3 → 1 event."""
        ts = self._ts(5)
        metric = pd.Series([100.0] * 5)
        events, mask = detect_exceedance_runs(ts, metric, 95.0, 60.0, 3.0, 1.5)
        assert len(events) == 1
        assert events[0]["duration_hours"] == pytest.approx(5.0)
        assert mask.sum() == 5

    def test_below_threshold_no_events(self):
        """All below threshold → no events."""
        ts = self._ts(5)
        metric = pd.Series([80.0] * 5)
        events, mask = detect_exceedance_runs(ts, metric, 95.0, 60.0, 0.0, 1.5)
        assert len(events) == 0
        assert mask.sum() == 0

    def test_nan_breaks_run(self):
        """NaN in the middle breaks a run into two events."""
        ts = self._ts(7)
        metric = pd.Series([100.0, 100.0, 100.0, float("nan"), 100.0, 100.0, 100.0])
        events, mask = detect_exceedance_runs(ts, metric, 95.0, 60.0, 3.0, 1.5)
        assert len(events) == 2

    def test_min_hours_filters(self):
        """2-hr run with min=3 → no qualifying events."""
        ts = self._ts(2)
        metric = pd.Series([100.0, 100.0])
        events, mask = detect_exceedance_runs(ts, metric, 95.0, 60.0, 3.0, 1.5)
        assert len(events) == 0

    def test_returns_timestamps(self):
        """Events contain start_ts and end_ts as Timestamps."""
        ts = self._ts(4)
        metric = pd.Series([100.0] * 4)
        events, _ = detect_exceedance_runs(ts, metric, 95.0, 60.0, 0.0, 1.5)
        assert len(events) == 1
        assert hasattr(events[0]["start_ts"], "tzinfo")
        assert hasattr(events[0]["end_ts"], "tzinfo")


# ── TestEventWindStats ────────────────────────────────────────────────────────

class TestEventWindStats:
    def test_hot_event_detected(self):
        """Exceedance rows with wind data → non-zero event stats."""
        df = _wind_df(10, drct=180.0, sknt=15.0, temp=[100.0] * 5 + [60.0] * 5)
        df = normalize_wind(df)
        result = compute_event_wind_stats(
            df, "temp", 95.0, 60.0, dir_bins=16, min_event_hours=0.0,
        )
        assert result["event_count"] == 1
        assert result["event_hours_total"] == pytest.approx(5.0)
        assert result["mean_speed_kt"] == pytest.approx(15.0, rel=0.01)

    def test_no_exceedance_zero_events(self):
        """All below threshold → 0 events."""
        df = _wind_df(10, drct=180.0, sknt=10.0, temp=60.0)
        df = normalize_wind(df)
        result = compute_event_wind_stats(
            df, "temp", 95.0, 60.0, dir_bins=16, min_event_hours=0.0,
        )
        assert result["event_count"] == 0

    def test_min_hours_filter(self):
        """Short exceedance filtered by min_event_hours."""
        df = _wind_df(2, drct=180.0, sknt=10.0, temp=100.0)
        df = normalize_wind(df)
        result = compute_event_wind_stats(
            df, "temp", 95.0, 60.0, dir_bins=16, min_event_hours=5.0,
        )
        assert result["event_count"] == 0

    def test_missing_metric_col_graceful(self):
        """Missing metric column → graceful 0-event return."""
        df = _wind_df(5, drct=180.0, sknt=10.0)
        df = normalize_wind(df)
        result = compute_event_wind_stats(
            df, "nonexistent", 95.0, 60.0, dir_bins=16,
        )
        assert result["event_count"] == 0

    def test_top3_sectors_populated(self):
        """Event with consistent wind direction → top3_sectors has entries."""
        df = _wind_df(10, drct=180.0, sknt=15.0, temp=100.0)
        df = normalize_wind(df)
        result = compute_event_wind_stats(
            df, "temp", 95.0, 60.0, dir_bins=16, min_event_hours=0.0,
        )
        assert len(result["top3_sectors"]) >= 1
        assert "S" in result["top3_sectors"]


# ── TestSectorDeltas ──────────────────────────────────────────────────────────

class TestSectorDeltas:
    def test_overrepresentation(self):
        """Event concentrated in S vs uniform baseline → S overrepresented."""
        # Build a baseline rose with uniform hours across 16 sectors
        labels_16 = sector_labels(16)
        uniform_data = {f"0-5": [1.0] * 16 + [0.0]}  # 16 sectors + Calm
        baseline = pd.DataFrame(uniform_data, index=labels_16 + ["Calm"])

        # Event: 100% from South
        event_pcts = {lbl: 0.0 for lbl in labels_16}
        event_pcts["S"] = 100.0

        result = compute_sector_deltas(baseline, event_pcts)
        assert result["overrepresented_sector"] == "S"
        assert result["overrepresented_delta_pct"] > 0
