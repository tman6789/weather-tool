"""Tests for hours_above_ref computation."""

import math

import numpy as np
import pandas as pd
import pytest

from weather_tool.core.metrics import hours_above_ref


class TestHoursAboveRef:
    def test_basic_20min(self):
        """All 6 readings above ref → 6 * (20/60) = 2.0 hours."""
        temps = pd.Series([70.0, 72.0, 68.0, 75.0, 66.0, 80.0])
        result = hours_above_ref(temps, ref_temp=65.0, dt_minutes=20.0)
        assert result == pytest.approx(6 * (20 / 60))

    def test_basic_60min(self):
        """3 of 5 readings above ref → 3 * 1.0 = 3.0 hours."""
        temps = pd.Series([70.0, 60.0, 80.0, 50.0, 90.0])
        result = hours_above_ref(temps, ref_temp=65.0, dt_minutes=60.0)
        assert result == pytest.approx(3.0)

    def test_nan_temps_ignored(self):
        """NaN temps should not count toward hours above."""
        temps = pd.Series([70.0, np.nan, 80.0, np.nan, 90.0])
        result = hours_above_ref(temps, ref_temp=65.0, dt_minutes=60.0)
        # 3 valid temps above 65 → 3 hours
        assert result == pytest.approx(3.0)

    def test_all_nan(self):
        temps = pd.Series([np.nan, np.nan, np.nan])
        result = hours_above_ref(temps, ref_temp=65.0, dt_minutes=60.0)
        assert result == pytest.approx(0.0)

    def test_none_above(self):
        temps = pd.Series([60.0, 50.0, 40.0])
        result = hours_above_ref(temps, ref_temp=65.0, dt_minutes=20.0)
        assert result == pytest.approx(0.0)

    def test_exact_ref_not_counted(self):
        """Temp exactly equal to ref_temp should NOT count (strict >)."""
        temps = pd.Series([65.0, 65.0, 65.0])
        result = hours_above_ref(temps, ref_temp=65.0, dt_minutes=60.0)
        assert result == pytest.approx(0.0)

    def test_zero_dt(self):
        temps = pd.Series([70.0, 80.0])
        result = hours_above_ref(temps, ref_temp=65.0, dt_minutes=0.0)
        assert result == pytest.approx(0.0)

    def test_nan_dt(self):
        temps = pd.Series([70.0, 80.0])
        result = hours_above_ref(temps, ref_temp=65.0, dt_minutes=float("nan"))
        assert result == pytest.approx(0.0)
