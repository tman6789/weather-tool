"""Tests for weather_tool/insights/rules.py."""

from __future__ import annotations

import inspect

import pytest

from weather_tool.insights import rules as rules_module
from weather_tool.insights.rules import (
    PROFILES,
    Recommendation,
    evaluate_station_flags,
)


def _make_packet(
    wb_p996: float | None = 79.0,
    wb_mean_72h_max: float | None = 73.0,
    freeze_hours_sum: float | None = 200.0,
    freeze_event_max: float | None = 10.0,
    freeze_shoulder: float | None = 30.0,
    wec_pct: float | None = 0.40,
    air_econ_hrs: float | None = 8000.0,
    tower_stress78: float | None = 50.0,
    tdb_p996: float | None = 96.0,
    tdb_mean_72h_max: float | None = 88.0,
    missing_warning: bool = False,
) -> dict:
    return {
        "design_conditions": {
            "tdb_p99":        92.0,
            "tdb_p996":       tdb_p996,
            "wb_p99":         74.0,
            "wb_p996":        wb_p996,
            "tdb_mean_72h_max": tdb_mean_72h_max,
            "wb_mean_72h_max":  wb_mean_72h_max,
        },
        "operational_efficiency": {
            "air_econ_hours_sum":              air_econ_hrs,
            "wec_hours_sum":                   5000.0,
            "wec_feasible_pct_over_window":    wec_pct,
            "hours_with_wetbulb_sum":          12000.0,
            "tower_stress_hours_wb_gt_78_sum": tower_stress78,
        },
        "freeze_risk": {
            "freeze_hours_sum":                    freeze_hours_sum,
            "freeze_hours_shoulder_sum":            freeze_shoulder,
            "freeze_event_max_duration_hours_max":  freeze_event_max,
        },
        "quality": {
            "missing_data_warning": missing_warning,
        },
    }


# ── Structural tests ───────────────────────────────────────────────────────────

def test_no_pandas_import():
    """rules.py must not import pandas — pure dict-based logic only."""
    source = inspect.getsource(rules_module)
    assert "import pandas" not in source
    assert "from pandas" not in source


def test_profiles_exist():
    for name in ("datacenter", "economizer_first", "freeze_sensitive"):
        assert name in PROFILES


def test_recommendations_are_recommendation_objects():
    _, recs = evaluate_station_flags(_make_packet(wb_p996=82.0), PROFILES["datacenter"])
    assert all(isinstance(r, Recommendation) for r in recs)


# ── Tower heat rejection ───────────────────────────────────────────────────────

def test_high_wb_p996_triggers_high_severity():
    """wb_p996=81.0 ≥ 80.0 (datacenter high threshold) → high severity."""
    flags, _ = evaluate_station_flags(_make_packet(wb_p996=81.0), PROFILES["datacenter"])
    tower_flags = [f for f in flags if f["flag_id"] == "tower_heat_rejection"]
    assert len(tower_flags) == 1
    assert tower_flags[0]["severity"] == "high"


def test_medium_wb_p996_triggers_medium():
    """wb_p996=79.0 ≥ 78.0 but < 80.0 → medium severity."""
    flags, _ = evaluate_station_flags(_make_packet(wb_p996=79.0), PROFILES["datacenter"])
    tower_flags = [f for f in flags if f["flag_id"] == "tower_heat_rejection"]
    assert len(tower_flags) == 1
    assert tower_flags[0]["severity"] == "medium"


def test_below_threshold_no_tower_flag():
    """wb_p996=74.0 < 78.0 → no tower_heat_rejection flag."""
    flags, _ = evaluate_station_flags(_make_packet(wb_p996=74.0), PROFILES["datacenter"])
    assert not any(f["flag_id"] == "tower_heat_rejection" for f in flags)


def test_none_value_skips_flag():
    """wb_p996=None → no exception, no tower flag."""
    flags, recs = evaluate_station_flags(_make_packet(wb_p996=None), PROFILES["datacenter"])
    assert not any(f["flag_id"] == "tower_heat_rejection" for f in flags)


# ── Profile comparison ─────────────────────────────────────────────────────────

def test_freeze_sensitive_lower_threshold():
    """freeze_hours_sum=200: no flag for datacenter (medium=300), medium for freeze_sensitive (medium=100)."""
    packet = _make_packet(freeze_hours_sum=200.0)
    flags_dc, _ = evaluate_station_flags(packet, PROFILES["datacenter"])
    flags_fs, _ = evaluate_station_flags(packet, PROFILES["freeze_sensitive"])

    dc_freeze = [f for f in flags_dc if f["flag_id"] == "freeze_risk"]
    fs_freeze = [f for f in flags_fs if f["flag_id"] == "freeze_risk"]

    # 200 < 300 (datacenter medium threshold) → no flag
    assert len(dc_freeze) == 0
    # 200 >= 100 (freeze_sensitive medium) but < 500 (high) → medium
    assert len(fs_freeze) == 1
    assert fs_freeze[0]["severity"] == "medium"


def test_freeze_high_on_freeze_sensitive():
    """freeze_hours_sum=600 ≥ 500 (freeze_sensitive high) → high."""
    flags, _ = evaluate_station_flags(_make_packet(freeze_hours_sum=600.0), PROFILES["freeze_sensitive"])
    fr = [f for f in flags if f["flag_id"] == "freeze_risk"]
    assert fr and fr[0]["severity"] == "high"


# ── WEC and airside econ ───────────────────────────────────────────────────────

def test_wec_low_opportunity_flagged():
    """wec_pct=0.10 < 0.20 (datacenter low threshold) → wec_low_opportunity."""
    flags, _ = evaluate_station_flags(_make_packet(wec_pct=0.10), PROFILES["datacenter"])
    assert any(f["flag_id"] == "wec_low_opportunity" for f in flags)


def test_wec_high_opportunity_flagged():
    """wec_pct=0.55 ≥ 0.50 → wec_high_opportunity."""
    flags, _ = evaluate_station_flags(_make_packet(wec_pct=0.55), PROFILES["datacenter"])
    assert any(f["flag_id"] == "wec_high_opportunity" for f in flags)


def test_air_econ_low_flagged():
    """air_econ_hrs=400 < 500 → air_econ_low."""
    flags, _ = evaluate_station_flags(_make_packet(air_econ_hrs=400.0), PROFILES["datacenter"])
    assert any(f["flag_id"] == "air_econ_low" for f in flags)


# ── Data quality ───────────────────────────────────────────────────────────────

def test_missing_data_warning_flag():
    """missing_warning=True → data_quality_warning flag present."""
    flags, _ = evaluate_station_flags(_make_packet(missing_warning=True), PROFILES["datacenter"])
    assert any(f["flag_id"] == "data_quality_warning" for f in flags)


def test_no_missing_warning_no_quality_flag():
    flags, _ = evaluate_station_flags(_make_packet(missing_warning=False), PROFILES["datacenter"])
    assert not any(f["flag_id"] == "data_quality_warning" for f in flags)


# ── Recommendations ────────────────────────────────────────────────────────────

def test_recommendations_nonempty_when_high_flag():
    """Any high-severity flag should produce at least one recommendation."""
    _, recs = evaluate_station_flags(_make_packet(wb_p996=81.0), PROFILES["datacenter"])
    assert len(recs) >= 1


def test_recommendation_fields_present():
    _, recs = evaluate_station_flags(_make_packet(wb_p996=81.0), PROFILES["datacenter"])
    for r in recs:
        assert hasattr(r, "rec_id")
        assert hasattr(r, "title")
        assert hasattr(r, "severity")
        assert hasattr(r, "rationale")
        assert hasattr(r, "evidence")


# ── Extended freeze event ──────────────────────────────────────────────────────

def test_extended_freeze_event_flagged():
    """freeze_event_max=60 ≥ 48 (datacenter) → freeze_event_extended."""
    flags, _ = evaluate_station_flags(_make_packet(freeze_event_max=60.0), PROFILES["datacenter"])
    assert any(f["flag_id"] == "freeze_event_extended" for f in flags)


def test_short_freeze_event_not_flagged():
    flags, _ = evaluate_station_flags(_make_packet(freeze_event_max=10.0), PROFILES["datacenter"])
    assert not any(f["flag_id"] == "freeze_event_extended" for f in flags)


# ── Persistence ───────────────────────────────────────────────────────────────

def test_wb_persistence_flagged():
    """wb_mean_72h_max=79.0 ≥ 78.0 → wb_persistence_risk high."""
    flags, _ = evaluate_station_flags(_make_packet(wb_mean_72h_max=79.0), PROFILES["datacenter"])
    assert any(f["flag_id"] == "wb_persistence_risk" for f in flags)


def test_tdb_persistence_flagged():
    """tdb_mean_72h_max=96.0 ≥ 95.0 → tdb_persistence medium."""
    flags, _ = evaluate_station_flags(_make_packet(tdb_mean_72h_max=96.0), PROFILES["datacenter"])
    assert any(f["flag_id"] == "tdb_persistence" for f in flags)


# ── Missing / empty packet robustness ─────────────────────────────────────────

def test_empty_packet_no_crash():
    """Completely empty packet should not raise."""
    flags, recs = evaluate_station_flags({}, PROFILES["datacenter"])
    assert isinstance(flags, list)
    assert isinstance(recs, list)


def test_partial_packet_no_crash():
    """Packet with only quality key should not raise."""
    flags, recs = evaluate_station_flags({"quality": {"missing_data_warning": False}}, PROFILES["datacenter"])
    assert isinstance(flags, list)


def test_nan_value_skips_flag():
    """NaN metric values should be treated as absent → no flag."""
    import math
    flags, _ = evaluate_station_flags(_make_packet(wb_p996=float("nan")), PROFILES["datacenter"])
    assert not any(f["flag_id"] == "tower_heat_rejection" for f in flags)
