"""Unit tests for wet-bulb temperature computation (Stull 2011 approximation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from weather_tool.core.metrics import compute_wetbulb_f


class TestComputeWetbulbF:
    def test_no_humidity_columns_returns_nan(self):
        """DataFrame with only temp column (no relh or dwpf) → all NaN."""
        df = pd.DataFrame({"temp": [86.0, 95.0, 70.0]})
        result = compute_wetbulb_f(df)
        assert result.isna().all()

    def test_missing_temp_column_returns_nan(self):
        """DataFrame with no temp or tmpf column → all NaN."""
        df = pd.DataFrame({"relh": [50.0, 60.0]})
        result = compute_wetbulb_f(df)
        assert result.isna().all()

    def test_stull_30c_50rh(self):
        """T=86°F (30°C), RH=50% → Twb ≈ 72.3°F (Stull), tolerance ±2°F."""
        df = pd.DataFrame({"tmpf": [86.0], "relh": [50.0]})
        result = compute_wetbulb_f(df)
        assert not result.isna().any()
        assert result.iloc[0] == pytest.approx(72.3, abs=2.0)

    def test_stull_95f_60rh(self):
        """T=95°F, RH=60% → Twb should be higher than T=86°F, RH=50% case."""
        df_low = pd.DataFrame({"tmpf": [86.0], "relh": [50.0]})
        df_high = pd.DataFrame({"tmpf": [95.0], "relh": [60.0]})
        wb_low = compute_wetbulb_f(df_low).iloc[0]
        wb_high = compute_wetbulb_f(df_high).iloc[0]
        assert wb_high > wb_low

    def test_wetbulb_le_drybulb(self):
        """Wet-bulb must always be ≤ dry-bulb (physics constraint)."""
        df = pd.DataFrame({
            "tmpf": [86.0, 95.0, 32.0, 104.0],
            "relh": [50.0, 60.0, 80.0, 30.0],
        })
        result = compute_wetbulb_f(df)
        assert (result.values <= df["tmpf"].values + 0.01).all()

    def test_100pct_rh_equals_drybulb(self):
        """At 100% RH, wet-bulb ≈ dry-bulb (tolerance ±0.5°F)."""
        df = pd.DataFrame({"tmpf": [75.0], "relh": [100.0]})
        result = compute_wetbulb_f(df)
        assert result.iloc[0] == pytest.approx(75.0, abs=0.5)

    def test_nan_tmpf_row_produces_nan_wetbulb(self):
        """Rows where tmpf is NaN produce NaN wetbulb_f; valid rows still computed."""
        df = pd.DataFrame({
            "tmpf": [np.nan, 86.0, np.nan],
            "relh": [50.0, 50.0, 50.0],
        })
        result = compute_wetbulb_f(df)
        assert np.isnan(result.iloc[0])
        assert not np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])

    def test_nan_relh_row_produces_nan_wetbulb(self):
        """Rows where relh is NaN produce NaN wetbulb_f."""
        df = pd.DataFrame({
            "tmpf": [86.0, 86.0],
            "relh": [np.nan, 50.0],
        })
        result = compute_wetbulb_f(df)
        assert np.isnan(result.iloc[0])
        assert not np.isnan(result.iloc[1])

    def test_fallback_from_dwpf(self):
        """No relh column, but dwpf present → non-NaN result close to direct relh path.

        At T=86°F (30°C), RH=50%, the Magnus dew-point is ~65.2°F.
        Two approximation chains (Magnus → RH → Stull) may introduce up to ±2°F.
        """
        # Dew point computed to match T=86°F, RH=50% via the same Magnus formula used
        # in the fallback code: Tdc ≈ 18.43°C ≈ 65.2°F
        df_direct = pd.DataFrame({"tmpf": [86.0], "relh": [50.0]})
        df_fallback = pd.DataFrame({"tmpf": [86.0], "dwpf": [65.2]})
        wb_direct = compute_wetbulb_f(df_direct).iloc[0]
        wb_fallback = compute_wetbulb_f(df_fallback).iloc[0]
        assert not np.isnan(wb_fallback)
        # Two-approximation chain: tolerance ±2°F
        assert wb_fallback == pytest.approx(wb_direct, abs=2.0)

    def test_uses_temp_column_when_no_tmpf(self):
        """Falls back to 'temp' column when 'tmpf' is not present."""
        df = pd.DataFrame({"temp": [86.0], "relh": [50.0]})
        result = compute_wetbulb_f(df)
        assert not result.isna().any()
        assert result.iloc[0] == pytest.approx(72.3, abs=2.0)

    def test_tmpf_takes_priority_over_temp(self):
        """When both tmpf and temp are present, tmpf is used."""
        df = pd.DataFrame({"tmpf": [86.0], "temp": [999.0], "relh": [50.0]})
        result = compute_wetbulb_f(df)
        # Should use tmpf=86 not temp=999
        assert result.iloc[0] == pytest.approx(72.3, abs=2.0)

    def test_returns_series_with_same_index(self):
        """Returned Series preserves the DataFrame index."""
        df = pd.DataFrame(
            {"tmpf": [86.0, 95.0], "relh": [50.0, 60.0]},
            index=[10, 20],
        )
        result = compute_wetbulb_f(df)
        assert list(result.index) == [10, 20]
