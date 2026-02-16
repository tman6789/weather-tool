"""Tests for the deterministic comparison report generator."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from weather_tool.insights.compare_report import generate_compare_report_md


def _make_compare_df(stations: list[str]) -> pd.DataFrame:
    """Build a minimal compare_df for report generation tests."""
    rows = []
    for i, sid in enumerate(stations):
        rows.append({
            "station_id": sid,
            "years_covered_count": 3,
            "tmax_max": 100.0 + i,
            "tmin_min": 10.0 - i,
            "t_p99_median": 95.0 + i,
            "wb_p99_median": 72.0 + i,
            "hours_above_ref_80_sum": 500.0 * (i + 1),
            "hours_above_ref_90_sum": 200.0 * (i + 1),
            "freeze_hours_sum": 300.0 - i * 50,
            "heat_score": 80.0 - i * 20,
            "moisture_score": 75.0 - i * 10,
            "freeze_score": 60.0 - i * 15,
            "data_quality_score": 90.0,
            "overall_score": 78.0 - i * 15,
            "coverage_weighted_pct": 0.99,
            "timestamp_missing_pct_avg": 0.005,
            "missing_data_warning": False,
            "interval_change_flag_any": False,
        })
    return pd.DataFrame(rows)


class TestGenerateCompareReportMd:
    def test_contains_all_station_ids(self):
        stations = ["KIAD", "KDEN", "KPHX"]
        df = _make_compare_df(stations)
        report = generate_compare_report_md(
            df, stations, date(2020, 1, 1), date(2022, 12, 31),
            ["tmpf", "dwpf", "relh"], [80.0, 90.0]
        )
        for sid in stations:
            assert sid in report

    def test_contains_required_sections(self):
        stations = ["KORD", "KLAX"]
        df = _make_compare_df(stations)
        report = generate_compare_report_md(
            df, stations, date(2021, 1, 1), date(2023, 12, 31),
            ["tmpf", "relh"], [65.0]
        )
        assert "## Data Quality Summary" in report
        assert "## Rankings" in report
        assert "## Key Differences" in report
        assert "## Comparison Table" in report

    def test_weak_ranks_warning_for_two_stations(self):
        stations = ["KORD", "KLAX"]
        df = _make_compare_df(stations)
        report = generate_compare_report_md(
            df, stations, date(2021, 1, 1), date(2023, 12, 31),
            ["tmpf", "relh"], [65.0]
        )
        assert "fewer than 3 stations" in report.lower() or "< 3 stations" in report.lower() or "weak" in report.lower()

    def test_no_weak_ranks_warning_for_three_stations(self):
        stations = ["KORD", "KLAX", "KPHX"]
        df = _make_compare_df(stations)
        report = generate_compare_report_md(
            df, stations, date(2021, 1, 1), date(2023, 12, 31),
            ["tmpf", "relh"], [65.0]
        )
        assert "fewer than 3 stations" not in report

    def test_window_and_fields_in_header(self):
        stations = ["KIAD", "KDEN"]
        df = _make_compare_df(stations)
        report = generate_compare_report_md(
            df, stations, date(2020, 6, 1), date(2023, 8, 31),
            ["tmpf", "dwpf", "relh"], [80.0, 85.0]
        )
        assert "2020-06-01" in report
        assert "2023-08-31" in report
        assert "tmpf" in report
        assert "80" in report

    def test_quality_warning_surfaced_in_report(self):
        stations = ["KGOOD", "KBAD"]
        df = _make_compare_df(stations)
        # Inject a bad missing_pct for KBAD
        df.loc[df["station_id"] == "KBAD", "timestamp_missing_pct_avg"] = 0.08
        report = generate_compare_report_md(
            df, stations, date(2021, 1, 1), date(2022, 12, 31),
            ["tmpf"], [65.0]
        )
        assert "KBAD" in report
        assert "missing" in report.lower() or "8.0%" in report

    def test_key_differences_section_contains_freeze_line(self):
        stations = ["KWARM", "KCOLD"]
        df = _make_compare_df(stations)
        report = generate_compare_report_md(
            df, stations, date(2020, 1, 1), date(2022, 12, 31),
            ["tmpf", "dwpf", "relh"], [65.0]
        )
        assert "freeze" in report.lower() or "32°F" in report
