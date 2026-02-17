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

    def test_wb_summary_columns_present_when_wetbulb_available(self):
        """DataFrame with tmpf + relh columns → summary includes wb_p99, wb_p996, wb_max, wb_mean."""
        periods = 8760
        ts = pd.date_range("2023-01-01", periods=periods, freq="60min", tz="UTC")
        temps_f = np.full(periods, 86.0)   # 30 °C
        relh = np.full(periods, 50.0)      # 50 % RH → wb ≈ 72 °F

        from weather_tool.core.metrics import compute_wetbulb_f
        df = pd.DataFrame({
            "timestamp": ts,
            "tmpf": temps_f,
            "temp": temps_f,
            "relh": relh,
            "station_id": "TEST",
        })
        df["wetbulb_f"] = compute_wetbulb_f(df)

        cfg = RunConfig(
            mode="iem",
            station_id="TEST",
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
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
        assert len(summary) == 1
        row = summary.iloc[0]

        for col in ("wb_p99", "wb_p996", "wb_max", "wb_mean"):
            assert col in summary.columns, f"Expected column {col!r} in summary"
            assert row[col] is not None
            assert not np.isnan(float(row[col]))

        # wb values should be below dry-bulb (86 °F)
        assert float(row["wb_p99"]) < 86.0
        # wb_p99 ≈ wb_p996 ≈ wb_max when all values are identical
        assert float(row["wb_p99"]) == pytest.approx(float(row["wb_max"]), abs=0.1)

    def test_wb_summary_absent_when_no_wetbulb(self):
        """DataFrame without wetbulb_f → no wb_ columns in summary."""
        df = _build_dataset("2023-01-01", periods=8760, freq_minutes=60)
        cfg = RunConfig(
            mode="csv",
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
        for col in ("wb_p99", "wb_p996", "wb_max", "wb_mean"):
            assert col not in summary.columns

    def test_field_quality_columns_in_summary(self):
        """DataFrame with relh + wetbulb_f → quality cols appear in summary."""
        periods = 100
        ts = pd.date_range("2023-07-01", periods=periods, freq="60min", tz="UTC")
        temps_f = np.full(periods, 86.0)
        relh = np.full(periods, 50.0)
        relh[0] = np.nan   # one missing relh

        from weather_tool.core.metrics import compute_wetbulb_f
        df = pd.DataFrame({
            "timestamp": ts,
            "tmpf": temps_f,
            "temp": temps_f,
            "relh": relh,
            "station_id": "TEST",
        })
        df["wetbulb_f"] = compute_wetbulb_f(df)

        cfg = RunConfig(
            mode="iem",
            station_id="TEST",
            start=date(2023, 7, 1),
            end=date(2023, 7, 31),
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

        assert "nan_count_relh" in summary.columns
        assert "nan_count_wetbulb_f" in summary.columns
        assert "wetbulb_availability_pct" in summary.columns
        # One NaN in relh → nan_count_relh >= 1
        assert int(row["nan_count_relh"]) >= 1
        # wetbulb_availability_pct should be < 100 because one row has NaN wetbulb
        assert float(row["wetbulb_availability_pct"]) < 100.0

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

    def test_wetbulb_not_computed_when_only_drybulb(self):
        """Bug 1 regression: tmpf-only data must NOT create a wetbulb_f column.

        Before the fix, the pipeline gate fired on any of (tmpf|relh|dwpf),
        so a dry-bulb-only dataset with tmpf still called compute_wetbulb_f and
        added an all-NaN wetbulb_f column, which set wetbulb_availability_pct=0
        and tripped econ/freeze confidence flags spuriously.
        """
        periods = 100
        ts = pd.date_range("2023-07-01", periods=periods, freq="60min", tz="UTC")
        df = pd.DataFrame({
            "timestamp": ts,
            "tmpf": np.full(periods, 86.0),
            "temp": np.full(periods, 86.0),
            "station_id": "TEST",
            # no relh, no dwpf
        })
        cfg = RunConfig(
            mode="iem",
            station_id="TEST",
            start=date(2023, 7, 1),
            end=date(2023, 7, 31),
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
        # No wb_ columns — wet-bulb gate should not have fired
        for col in ("wb_p99", "wb_p996", "wb_max", "wb_mean"):
            assert col not in summary.columns, f"Unexpected column {col!r} in dry-bulb-only summary"
        # No wetbulb_availability_pct — confirms gate did not fire
        assert "wetbulb_availability_pct" not in summary.columns

    def test_wetbulb_availability_pct_ignores_duplicates(self):
        """Bug 2 regression: duplicate rows with valid temp but NaN wetbulb must not
        inflate the denominator and depress wetbulb_availability_pct.

        Before the fix, df (including _is_dup=True rows) was used for both numerator
        and denominator; a dup with temp != NaN but wetbulb_f = NaN would add 1 to the
        denominator without contributing to the numerator, giving < 100% availability
        even when all *unique* observations have valid wetbulb.
        """
        from weather_tool.core.metrics import compute_wetbulb_f
        periods = 5
        ts_unique = pd.date_range("2023-07-01", periods=periods, freq="60min", tz="UTC")
        # Insert a duplicate of ts_unique[0] at the end — same timestamp, valid temp
        timestamps = list(ts_unique) + [ts_unique[0]]
        temps_f = [86.0] * periods + [86.0]   # all valid including the dup
        relh_vals = [50.0] * periods + [float("nan")]  # dup has NaN relh → NaN wetbulb

        df = pd.DataFrame({
            "timestamp": timestamps,
            "tmpf": temps_f,
            "temp": temps_f,
            "relh": relh_vals,
            "station_id": "TEST",
        })
        df["wetbulb_f"] = compute_wetbulb_f(df)

        cfg = RunConfig(
            mode="iem",
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
        # All 5 unique timestamps have valid wetbulb (relh is non-NaN for originals)
        # Availability must be 100 % — the duplicate must not count in denominator
        assert "wetbulb_availability_pct" in summary.columns
        assert float(row["wetbulb_availability_pct"]) == pytest.approx(100.0)

    def test_per_year_dt_when_interval_changes(self):
        """Bug 3 regression: when interval_change_flag is True, each year must use
        its own inferred dt_minutes rather than the global median.

        Year 2022 at 60-min (8760 obs), year 2023 at 20-min (26280 obs).
        Global median is dominated by whichever interval has more diffs; the per-year
        fix must store the correct dt for each year so hours_above_ref is scaled right.
        """
        ts_2022 = pd.date_range("2022-01-01", periods=8760, freq="60min", tz="UTC")
        ts_2023 = pd.date_range("2023-01-01", periods=26280, freq="20min", tz="UTC")
        all_ts = list(ts_2022) + list(ts_2023)
        all_temps = [80.0] * len(all_ts)  # all above ref_temp=65

        df = pd.DataFrame({
            "timestamp": all_ts,
            "temp": all_temps,
            "station_id": "TEST",
        })
        cfg = RunConfig(
            mode="csv",
            station_id="TEST",
            start=date(2022, 1, 1),
            end=date(2023, 12, 31),
            ref_temp=65.0,
            units="agnostic",
            tz="UTC",
        )
        normed = normalize_timestamps(df)
        windowed = filter_window(normed, cfg.start, cfg.end)
        from weather_tool.core.normalize import deduplicated
        dedup = deduplicated(windowed)
        interval = infer_interval(dedup["timestamp"])

        # The mixed-interval dataset must raise interval_change_flag
        assert interval["interval_change_flag"] is True

        summary = build_yearly_summary(windowed, cfg, interval)
        assert len(summary) == 2

        row_2022 = summary[summary["year"] == 2022].iloc[0]
        row_2023 = summary[summary["year"] == 2023].iloc[0]

        # Per-year dt must match the actual interval for each year
        assert row_2022["dt_minutes"] == pytest.approx(60.0)
        assert row_2023["dt_minutes"] == pytest.approx(20.0)

        # hours_above_ref must be correctly scaled:
        # 2022: 8760 obs × (60/60) = 8760 hrs (non-leap year, all above ref)
        assert row_2022["hours_above_ref"] == pytest.approx(8760.0)
        # 2023: 26280 obs × (20/60) = 8760 hrs
        assert row_2023["hours_above_ref"] == pytest.approx(8760.0)
