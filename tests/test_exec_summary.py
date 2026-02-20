"""Tests for weather_tool/insights/exec_summary.py."""

from __future__ import annotations

from weather_tool.insights.exec_summary import (
    render_exec_summary_compare,
    render_exec_summary_station,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_minimal_packet(
    station_id: str = "KTEST",
    profile: str = "datacenter",
    wb_p996: float | None = 80.5,
    death_day_mode: str = "death_day",
    include_candidates: bool = True,
) -> dict:
    candidate = {
        "rank": 1,
        "mode": death_day_mode,
        "confidence": "high" if death_day_mode == "death_day" else "low",
        "window_hours": 24,
        "start_ts": "2022-07-23T00:00:00+00:00",
        "end_ts": "2022-07-24T00:00:00+00:00",
        "stress_score": 1.37,
        "z_tdb": 1.21,
        "z_wb": 1.52,
        "tdb_mean_f": 96.2,
        "tdb_max_f": 101.0,
        "twb_mean_f": 79.1,
        "twb_max_f": 82.0,
        "rh_mean_pct": 54.2,
        "wind_mean_kt": 4.3,
        "calm_pct": 0.41,
    }
    flag = {
        "flag_id": "tower_heat_rejection",
        "severity": "high",
        "confidence": "high",
        "evidence": [{"metric": "wb_p996", "value": wb_p996, "threshold": 78.0}],
        "notes": "Wet-bulb design extreme is elevated.",
    }
    rec = {
        "rec_id": "size_tower_for_wb_p996",
        "title": "Size cooling towers above p99.6 wet-bulb",
        "severity": "high",
        "rationale": "The wb_p996 value exceeds the standard 78°F tower design point.",
        "evidence": [{"metric": "wb_p996", "value": wb_p996, "threshold": 78.0}],
    }
    return {
        "meta": {
            "station_id": station_id,
            "window_start": "2018-01-01",
            "window_end": "2022-12-31",
            "profile": profile,
            "dt_minutes": 60.0,
            "years_covered": 5,
            "ref_temp": 65.0,
        },
        "quality": {
            "missing_data_warning": False,
            "wetbulb_availability_pct": 95.0,
            "missing_pct_avg": 0.005,
            "interval_change_flag": False,
        },
        "design_conditions": {
            "tdb_p99": 92.0,
            "tdb_p996": 96.0,
            "tdb_max": 105.0,
            "wb_p99": 74.0,
            "wb_p996": wb_p996,
            "wb_max": 83.0,
            "wb_mean": 65.0,
            "tdb_mean_24h_max": 88.0,
            "tdb_mean_72h_max": 85.0,
            "wb_mean_24h_max": 74.0,
            "wb_mean_72h_max": 72.0,
            "lwt_proxy_p99": 68.0,
            "baseline_years_count": 5,
            "baseline_method": "median_over_years",
        },
        "operational_efficiency": {
            "air_econ_hours_sum": 18000.0,
            "wec_hours_sum": 12000.0,
            "hours_with_wetbulb_sum": 25000.0,
            "wec_feasible_pct_over_window": 0.48,
            "tower_stress_hours_wb_gt_75_sum": 300.0,
            "tower_stress_hours_wb_gt_78_sum": 100.0,
            "tower_stress_hours_wb_gt_80_sum": 40.0,
        },
        "freeze_risk": {
            "freeze_hours_sum": 480.0,
            "freeze_hours_shoulder_sum": 80.0,
            "freeze_event_count_sum": 15,
            "freeze_event_max_duration_hours_max": 18.0,
            "tmin_min": 8.0,
        },
        "death_day": {
            "mode": death_day_mode,
            "window_hours": 24,
            "candidates": [candidate] if include_candidates else [],
        },
        "risk_flags": [flag] if wb_p996 is not None and wb_p996 >= 78.0 else [],
        "recommendations": [rec] if wb_p996 is not None and wb_p996 >= 78.0 else [],
    }


def _make_compare_packet() -> dict:
    return {
        "meta": {
            "window_start": "2018-01-01",
            "window_end": "2022-12-31",
            "stations": ["KORD", "KPHX", "KIAD"],
            "station_count": 3,
        },
        "station_summaries": [
            {
                "station_id": "KORD",
                "tdb_p996": 92.0,
                "wb_p996": 76.0,
                "air_econ_hours_sum": 22000.0,
                "freeze_hours_sum": 900.0,
                "death_day_rank1": None,
                "risk_flags": [],
            },
            {
                "station_id": "KPHX",
                "tdb_p996": 108.0,
                "wb_p996": 75.0,
                "air_econ_hours_sum": 5000.0,
                "freeze_hours_sum": 10.0,
                "death_day_rank1": None,
                "risk_flags": [
                    {
                        "flag_id": "heat_design_extreme",
                        "severity": "high",
                        "confidence": "high",
                        "evidence": [{"metric": "tdb_p996", "value": 108.0, "threshold": 105.0}],
                        "notes": "Extreme dry-bulb design point.",
                    }
                ],
            },
            {
                "station_id": "KIAD",
                "tdb_p996": 95.0,
                "wb_p996": 78.5,
                "air_econ_hours_sum": 14000.0,
                "freeze_hours_sum": 300.0,
                "death_day_rank1": None,
                "risk_flags": [],
            },
        ],
        "station_packets": None,
        "rankings": {
            "overall_score": [
                {"station_id": "KPHX", "value": 80.0, "rank": 1},
                {"station_id": "KIAD", "value": 55.0, "rank": 2},
                {"station_id": "KORD", "value": 40.0, "rank": 3},
            ],
            "heat_score": [
                {"station_id": "KPHX", "value": 100.0, "rank": 1},
                {"station_id": "KIAD", "value": 60.0, "rank": 2},
                {"station_id": "KORD", "value": 30.0, "rank": 3},
            ],
            "moisture_score": [],
            "freeze_score": [
                {"station_id": "KORD", "value": 90.0, "rank": 1},
                {"station_id": "KIAD", "value": 50.0, "rank": 2},
                {"station_id": "KPHX", "value": 10.0, "rank": 3},
            ],
            "data_quality_score": [
                {"station_id": "KORD", "value": 95.0, "rank": 1},
                {"station_id": "KIAD", "value": 80.0, "rank": 2},
                {"station_id": "KPHX", "value": 75.0, "rank": 3},
            ],
        },
        "cross_station_extremes": {
            "hottest_station": {"station_id": "KPHX", "value": 115.0},
            "coldest_station": {"station_id": "KORD", "value": 5.0},
            "highest_wb_p996": {"station_id": "KIAD", "value": 78.5},
            "most_econ_hours": {"station_id": "KORD", "value": 22000.0},
            "most_freeze_hours": {"station_id": "KORD", "value": 900.0},
        },
        "aggregated_flags": [
            {
                "flag_id": "heat_design_extreme",
                "severity": "high",
                "station_id": "KPHX",
                "confidence": "high",
                "evidence": [{"metric": "tdb_p996", "value": 108.0, "threshold": 105.0}],
                "notes": "Extreme dry-bulb design point.",
            }
        ],
    }


# ── Station render tests ───────────────────────────────────────────────────────

def test_render_station_returns_string():
    pkt = _make_minimal_packet()
    result = render_exec_summary_station(pkt)
    assert isinstance(result, str)
    assert len(result) > 100


def test_contains_all_section_headers():
    pkt = _make_minimal_packet()
    result = render_exec_summary_station(pkt)
    for i in range(1, 6):
        assert f"## {i}." in result, f"Missing section header ## {i}."


def test_station_id_appears_in_output():
    pkt = _make_minimal_packet(station_id="KMIA")
    result = render_exec_summary_station(pkt)
    assert "KMIA" in result


def test_none_values_rendered_as_na():
    pkt = _make_minimal_packet(wb_p996=None)
    result = render_exec_summary_station(pkt)
    assert "N/A" in result


def test_empty_flags_renders_gracefully():
    pkt = _make_minimal_packet(wb_p996=70.0)  # below thresholds → no flags
    result = render_exec_summary_station(pkt)
    assert "No critical flags" in result or "## 1." in result


def test_heat_day_mode_note_appears():
    """When death_day mode is 'heat_day', a low-confidence note should appear."""
    pkt = _make_minimal_packet(death_day_mode="heat_day")
    result = render_exec_summary_station(pkt)
    assert "low confidence" in result.lower() or "wet-bulb data unavailable" in result.lower()


def test_recommendations_rendered_as_numbered_list():
    pkt = _make_minimal_packet(wb_p996=80.5)
    result = render_exec_summary_station(pkt)
    # At least one numbered recommendation entry
    assert "1. **" in result


def test_death_day_candidates_table_rendered():
    pkt = _make_minimal_packet(include_candidates=True)
    result = render_exec_summary_station(pkt)
    assert "stress_score" in result.lower() or "Stress" in result


def test_no_candidates_message_rendered():
    pkt = _make_minimal_packet(include_candidates=False)
    result = render_exec_summary_station(pkt)
    assert "No candidates identified" in result


def test_profile_appears_in_output():
    pkt = _make_minimal_packet(profile="freeze_sensitive")
    result = render_exec_summary_station(pkt)
    assert "freeze_sensitive" in result


def test_high_flag_evidence_in_resilience_section():
    pkt = _make_minimal_packet(wb_p996=80.5)
    result = render_exec_summary_station(pkt)
    # Section 4 should mention the highest-priority flag
    assert "tower_heat_rejection" in result


# ── Compare render tests ───────────────────────────────────────────────────────

def test_render_compare_returns_string():
    pkt = _make_compare_packet()
    result = render_exec_summary_compare(pkt)
    assert isinstance(result, str)
    assert len(result) > 100


def test_compare_contains_all_section_headers():
    pkt = _make_compare_packet()
    result = render_exec_summary_compare(pkt)
    for i in range(1, 6):
        assert f"## {i}." in result, f"Missing compare section header ## {i}."


def test_compare_station_ids_appear():
    pkt = _make_compare_packet()
    result = render_exec_summary_compare(pkt)
    for sid in ("KORD", "KPHX", "KIAD"):
        assert sid in result


def test_compare_cross_station_extremes_rendered():
    pkt = _make_compare_packet()
    result = render_exec_summary_compare(pkt)
    assert "Hottest station" in result
    assert "Coldest station" in result


def test_compare_aggregated_flags_rendered():
    pkt = _make_compare_packet()
    result = render_exec_summary_compare(pkt)
    assert "heat_design_extreme" in result


def test_compare_no_flags_renders_gracefully():
    pkt = _make_compare_packet()
    pkt["aggregated_flags"] = []
    result = render_exec_summary_compare(pkt)
    assert "No cross-station mitigations" in result or "## 5." in result


# ── LLM exec summary ──────────────────────────────────────────────────────────

def test_compare_section5_full_mode_renders_recommendations():
    """When station_packets is present (full mode), section 5 renders actual rec objects."""
    pkt = _make_compare_packet()
    pkt["station_packets"] = {
        "KPHX": {
            "recommendations": [
                {
                    "rec_id": "size_tower_for_tdb_p996",
                    "title": "Upsize cooling system for extreme Tdb",
                    "severity": "high",
                    "rationale": "tdb_p996 exceeds the 105°F high threshold.",
                    "evidence": [{"metric": "tdb_p996", "value": 108.0, "threshold": 105.0}],
                }
            ]
        }
    }
    result = render_exec_summary_compare(pkt)
    assert "Upsize cooling system" in result
    assert "Prioritised actions" in result


def test_llm_returns_none_without_api_key(monkeypatch):
    """generate_llm_exec_summary must return None when no API key is set."""
    import os
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from weather_tool.insights import llm_exec_summary as llm_mod
    result = llm_mod.generate_llm_exec_summary(_make_minimal_packet())
    assert result is None
