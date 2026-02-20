"""Tests for weather_tool/insights/packet.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from weather_tool.insights.packet import build_compare_packet, build_station_packet


# ── Minimal StationResult stub ─────────────────────────────────────────────────

@dataclass
class _FakeRunConfig:
    station_id: str | None = "KTEST"
    start: date = field(default_factory=lambda: date(2018, 1, 1))
    end: date = field(default_factory=lambda: date(2022, 12, 31))
    ref_temp: float = 65.0
    decision_profile: str = "datacenter"
    death_day_window_hours: int = 24
    death_day_top_n: int = 5


@dataclass
class _FakeStationResult:
    summary: pd.DataFrame
    windowed: pd.DataFrame
    interval_info: dict[str, Any]
    quality_report: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    cfg: Any = field(default_factory=_FakeRunConfig)
    wind_results: dict[str, Any] | None = None
    design_day: pd.DataFrame | None = None
    decision: dict[str, Any] | None = None


# ── Fixture helpers ────────────────────────────────────────────────────────────

def _make_summary(n_years: int = 3, include_wb: bool = True) -> pd.DataFrame:
    """Synthetic yearly summary (one row per year)."""
    years = list(range(2020, 2020 + n_years))
    data: dict[str, list] = {
        "year":    years,
        "t_p99":   [92.0 + i for i in range(n_years)],
        "t_p996":  [96.0 + i for i in range(n_years)],
        "tmax":    [105.0 + i for i in range(n_years)],
        "tmin":    [10.0 - i for i in range(n_years)],
        "missing_pct": [0.005] * n_years,
        "freeze_hours": [150.0 + i * 10 for i in range(n_years)],
        "freeze_hours_shoulder": [20.0] * n_years,
        "freeze_event_count": [5] * n_years,
        "freeze_event_max_duration_hours": [12.0] * n_years,
        "air_econ_hours": [6000.0] * n_years,
        "wec_hours": [4000.0] * n_years,
        "hours_with_wetbulb": [8000.0] * n_years,
        "tower_stress_hours_wb_gt_75": [100.0] * n_years,
        "tower_stress_hours_wb_gt_78": [50.0] * n_years,
        "tower_stress_hours_wb_gt_80": [20.0] * n_years,
        "tdb_mean_24h_max": [88.0] * n_years,
        "tdb_mean_72h_max": [85.0] * n_years,
        "lwt_proxy_p99": [68.0] * n_years,
    }
    if include_wb:
        data["wb_p99"]  = [72.0 + i * 0.5 for i in range(n_years)]
        data["wb_p996"] = [76.0 + i * 0.5 for i in range(n_years)]
        data["wb_max"]  = [83.0] * n_years
        data["wb_mean"] = [65.0] * n_years
        data["wb_mean_24h_max"] = [74.0] * n_years
        data["wb_mean_72h_max"] = [72.0] * n_years
        data["wetbulb_availability_pct"] = [95.0] * n_years
    return pd.DataFrame(data)


def _make_windowed(n_hours: int = 500, include_wb: bool = True) -> pd.DataFrame:
    """Synthetic hourly time series with an injected heatwave at hours 200-247."""
    ts = pd.date_range("2020-01-01", periods=n_hours, freq="60min", tz="UTC")
    temps = [70.0 + 5 * np.sin(2 * np.pi * i / 24) for i in range(n_hours)]
    df = pd.DataFrame({"timestamp": ts, "temp": temps, "_is_dup": False})
    # Inject heatwave
    for i in range(200, min(248, n_hours)):
        df.loc[i, "temp"] = 102.0
    if include_wb:
        wb = [62.0 + 4 * np.sin(2 * np.pi * i / 24) for i in range(n_hours)]
        for i in range(200, min(248, n_hours)):
            wb[i] = 82.0
        df["wetbulb_f"] = wb
        df["relh"] = 55.0
        df["wind_speed_kt"] = 5.0
        df["is_calm"] = False
    return df


def _make_result(include_wb: bool = True, n_years: int = 3) -> _FakeStationResult:
    return _FakeStationResult(
        summary=_make_summary(n_years=n_years, include_wb=include_wb),
        windowed=_make_windowed(include_wb=include_wb),
        interval_info={"dt_minutes": 60.0, "interval_change_flag": False},
        cfg=_FakeRunConfig(),
    )


# ── Station packet structure ───────────────────────────────────────────────────

def test_station_packet_has_required_keys():
    result = _make_result()
    pkt = build_station_packet(result.cfg, result, "datacenter", 24, 5)
    for key in ("meta", "quality", "design_conditions", "operational_efficiency",
                "freeze_risk", "death_day", "risk_flags", "recommendations"):
        assert key in pkt, f"Missing top-level key: {key!r}"


def test_meta_fields_present():
    result = _make_result()
    pkt = build_station_packet(result.cfg, result, "datacenter", 24, 5)
    meta = pkt["meta"]
    for field_name in ("station_id", "window_start", "window_end", "profile",
                       "dt_minutes", "years_covered", "ref_temp"):
        assert field_name in meta, f"Missing meta field: {field_name!r}"
    assert meta["station_id"] == "KTEST"
    assert meta["profile"] == "datacenter"


def test_design_conditions_uses_medians():
    """Known per-year values should produce the correct median in design_conditions."""
    result = _make_result(n_years=3)
    pkt = build_station_packet(result.cfg, result, "datacenter", 24, 5)
    dc = pkt["design_conditions"]
    # t_p99 is [92, 93, 94] → median = 93.0
    assert dc["tdb_p99"] == pytest.approx(93.0, abs=0.01)
    # t_p996 is [96, 97, 98] → median = 97.0
    assert dc["tdb_p996"] == pytest.approx(97.0, abs=0.01)
    # tmax is [105, 106, 107] → max = 107.0
    assert dc["tdb_max"] == pytest.approx(107.0, abs=0.01)


def test_baseline_provenance_fields():
    """baseline_years_count and baseline_method must be in design_conditions."""
    result = _make_result(n_years=3)
    pkt = build_station_packet(result.cfg, result, "datacenter", 24, 5)
    dc = pkt["design_conditions"]
    assert "baseline_years_count" in dc
    assert dc["baseline_years_count"] == 3
    assert dc["baseline_method"] == "median_over_years"


def test_no_wetbulb_still_builds_packet():
    """Without wetbulb_f data wb fields should be None and death_day mode 'heat_day'."""
    result = _make_result(include_wb=False)
    pkt = build_station_packet(result.cfg, result, "datacenter", 24, 5)
    dc = pkt["design_conditions"]
    assert dc["wb_p99"] is None
    assert dc["wb_p996"] is None
    assert pkt["death_day"]["mode"] == "heat_day"


def test_death_day_candidates_populated():
    """With injected heatwave, candidates should be non-empty."""
    result = _make_result(include_wb=True)
    pkt = build_station_packet(result.cfg, result, "datacenter", 24, 5)
    assert len(pkt["death_day"]["candidates"]) >= 1


def test_death_day_window_hours_in_packet():
    result = _make_result()
    pkt = build_station_packet(result.cfg, result, "datacenter", 48, 5)
    assert pkt["death_day"]["window_hours"] == 48


def test_flags_and_recs_are_lists():
    result = _make_result()
    pkt = build_station_packet(result.cfg, result, "datacenter", 24, 5)
    assert isinstance(pkt["risk_flags"], list)
    assert isinstance(pkt["recommendations"], list)


def test_recommendations_are_dicts():
    """Recommendation objects must be serialized to dicts (JSON-portable)."""
    result = _make_result()
    pkt = build_station_packet(result.cfg, result, "datacenter", 24, 5)
    for r in pkt["recommendations"]:
        assert isinstance(r, dict)
        for field_name in ("rec_id", "title", "severity", "rationale", "evidence"):
            assert field_name in r


def test_quality_block_present():
    result = _make_result()
    pkt = build_station_packet(result.cfg, result, "datacenter", 24, 5)
    q = pkt["quality"]
    assert "missing_data_warning" in q
    assert "wetbulb_availability_pct" in q
    assert "missing_pct_avg" in q
    assert "interval_change_flag" in q


def test_operational_efficiency_sums():
    """air_econ_hours_sum should equal the per-year values summed."""
    result = _make_result(n_years=3)
    pkt = build_station_packet(result.cfg, result, "datacenter", 24, 5)
    oe = pkt["operational_efficiency"]
    assert oe["air_econ_hours_sum"] == pytest.approx(3 * 6000.0, abs=0.1)


def test_freeze_risk_sums():
    result = _make_result(n_years=3)
    pkt = build_station_packet(result.cfg, result, "datacenter", 24, 5)
    fr = pkt["freeze_risk"]
    # freeze_hours is [150, 160, 170] → sum = 480
    assert fr["freeze_hours_sum"] == pytest.approx(480.0, abs=0.1)
    # tmin is [10, 9, 8] → min = 8
    assert fr["tmin_min"] == pytest.approx(8.0, abs=0.1)


def test_different_profiles_accepted():
    """All three built-in profiles must produce a valid packet."""
    result = _make_result()
    for profile in ("datacenter", "economizer_first", "freeze_sensitive"):
        pkt = build_station_packet(result.cfg, result, profile, 24, 5)
        assert pkt["meta"]["profile"] == profile


def test_single_year_summary():
    """A single-year summary should still produce a valid packet (median == value)."""
    result = _make_result(n_years=1)
    pkt = build_station_packet(result.cfg, result, "datacenter", 24, 5)
    assert pkt["design_conditions"]["baseline_years_count"] == 1
    # tdb_p99 for 1 year: [92.0] → median = 92.0
    assert pkt["design_conditions"]["tdb_p99"] == pytest.approx(92.0, abs=0.01)


# ── RunConfig.validate() Decision AI checks ───────────────────────────────────

def test_validate_rejects_zero_window_hours():
    """death_day_window_hours=0 with decision_ai=True must raise ValueError."""
    from weather_tool.config import RunConfig
    cfg = RunConfig(
        mode="iem", station_id="KORD",
        start=date(2020, 1, 1), end=date(2022, 12, 31),
        decision_ai=True, death_day_window_hours=0,
    )
    with pytest.raises(ValueError, match="death_day_window_hours"):
        cfg.validate()


def test_validate_rejects_negative_window_hours():
    """death_day_window_hours=-1 with decision_ai=True must raise ValueError."""
    from weather_tool.config import RunConfig
    cfg = RunConfig(
        mode="iem", station_id="KORD",
        start=date(2020, 1, 1), end=date(2022, 12, 31),
        decision_ai=True, death_day_window_hours=-1,
    )
    with pytest.raises(ValueError, match="death_day_window_hours"):
        cfg.validate()


def test_validate_rejects_zero_top_n():
    """death_day_top_n=0 with decision_ai=True must raise ValueError."""
    from weather_tool.config import RunConfig
    cfg = RunConfig(
        mode="iem", station_id="KORD",
        start=date(2020, 1, 1), end=date(2022, 12, 31),
        decision_ai=True, death_day_top_n=0,
    )
    with pytest.raises(ValueError, match="death_day_top_n"):
        cfg.validate()


def test_validate_accepts_valid_decision_ai_config():
    """Valid decision_ai config must pass validate() without raising."""
    from weather_tool.config import RunConfig
    cfg = RunConfig(
        mode="iem", station_id="KORD",
        start=date(2020, 1, 1), end=date(2022, 12, 31),
        decision_ai=True, decision_profile="datacenter",
        death_day_window_hours=24, death_day_top_n=5,
    )
    cfg.validate()  # should not raise


# ── Compare packet ─────────────────────────────────────────────────────────────

def _make_station_packets(n: int = 3) -> list[dict]:
    """Build n minimal station packets for compare tests."""
    packets = []
    ids = ["KST1", "KST2", "KST3"][:n]
    for i, sid in enumerate(ids):
        cfg = _FakeRunConfig(station_id=sid)
        result = _make_result(n_years=2)
        pkt = build_station_packet(cfg, result, "datacenter", 24, 5)
        packets.append(pkt)
    return packets


def _make_compare_df(station_ids: list[str]) -> pd.DataFrame:
    """Minimal compare_df as used by build_compare_packet."""
    return pd.DataFrame({
        "station_id": station_ids,
        "overall_score": [80.0, 60.0, 40.0][: len(station_ids)],
        "heat_score": [75.0, 55.0, 35.0][: len(station_ids)],
        "moisture_score": [70.0, 50.0, 30.0][: len(station_ids)],
        "freeze_score": [65.0, 45.0, 25.0][: len(station_ids)],
        "data_quality_score": [90.0, 70.0, 50.0][: len(station_ids)],
        "tmax_max": [105.0, 110.0, 100.0][: len(station_ids)],
        "tmin_min": [8.0, 15.0, 5.0][: len(station_ids)],
        "wb_p996_median": [76.0, 78.0, 72.0][: len(station_ids)],
        "air_econ_hours_sum": [18000.0, 12000.0, 9000.0][: len(station_ids)],
        "freeze_hours_sum": [480.0, 300.0, 600.0][: len(station_ids)],
    })


def test_build_compare_packet_lean_default():
    """full=False (default) → station_packets key should be None."""
    stations = ["KST1", "KST2", "KST3"]
    pkts = _make_station_packets(3)
    cmp = build_compare_packet("2018-01-01", "2022-12-31", _make_compare_df(stations), pkts, stations)
    assert cmp["station_packets"] is None


def test_build_compare_packet_full():
    """full=True → station_packets key should be populated."""
    stations = ["KST1", "KST2", "KST3"]
    pkts = _make_station_packets(3)
    cmp = build_compare_packet("2018-01-01", "2022-12-31", _make_compare_df(stations), pkts, stations, full=True)
    assert cmp["station_packets"] is not None
    assert len(cmp["station_packets"]) == 3


def test_compare_packet_station_count():
    stations = ["KST1", "KST2"]
    pkts = _make_station_packets(2)
    cmp = build_compare_packet("2018-01-01", "2022-12-31", _make_compare_df(stations), pkts, stations)
    assert cmp["meta"]["station_count"] == 2
    assert len(cmp["station_summaries"]) == 2


def test_compare_packet_has_required_keys():
    stations = ["KST1", "KST2"]
    pkts = _make_station_packets(2)
    cmp = build_compare_packet("2018-01-01", "2022-12-31", _make_compare_df(stations), pkts, stations)
    for key in ("meta", "station_summaries", "rankings", "cross_station_extremes", "aggregated_flags"):
        assert key in cmp


def test_compare_rankings_have_all_score_cols():
    stations = ["KST1", "KST2", "KST3"]
    pkts = _make_station_packets(3)
    cmp = build_compare_packet("2018-01-01", "2022-12-31", _make_compare_df(stations), pkts, stations)
    for col in ("overall_score", "heat_score", "moisture_score", "freeze_score", "data_quality_score"):
        assert col in cmp["rankings"]


def test_compare_station_summaries_fields():
    """Each station summary must contain at minimum: station_id, tdb_p996, wb_p996."""
    stations = ["KST1", "KST2"]
    pkts = _make_station_packets(2)
    cmp = build_compare_packet("2018-01-01", "2022-12-31", _make_compare_df(stations), pkts, stations)
    for ss in cmp["station_summaries"]:
        assert "station_id" in ss
        assert "tdb_p996" in ss
        assert "wb_p996" in ss


def test_aggregated_flags_only_high_severity():
    """aggregated_flags should contain only high-severity flags."""
    stations = ["KST1", "KST2"]
    pkts = _make_station_packets(2)
    cmp = build_compare_packet("2018-01-01", "2022-12-31", _make_compare_df(stations), pkts, stations)
    for f in cmp["aggregated_flags"]:
        assert f["severity"] == "high"


def test_cross_station_extremes_present():
    stations = ["KST1", "KST2", "KST3"]
    pkts = _make_station_packets(3)
    cmp = build_compare_packet("2018-01-01", "2022-12-31", _make_compare_df(stations), pkts, stations)
    cse = cmp["cross_station_extremes"]
    assert "hottest_station" in cse
    assert "coldest_station" in cse
    assert "highest_wb_p996" in cse
    assert "most_econ_hours" in cse
    assert "most_freeze_hours" in cse
