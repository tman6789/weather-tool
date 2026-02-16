"""Tests for the 'compare' CLI command parsing and structure."""

from __future__ import annotations

from typer.testing import CliRunner

from weather_tool.cli import app


runner = CliRunner()


class TestCompareHelp:
    def test_compare_help_exits_zero(self):
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0

    def test_compare_help_lists_start_option(self):
        result = runner.invoke(app, ["compare", "--help"])
        assert "--start" in result.output

    def test_compare_help_lists_end_option(self):
        result = runner.invoke(app, ["compare", "--help"])
        assert "--end" in result.output

    def test_compare_help_lists_ref_temps_option(self):
        result = runner.invoke(app, ["compare", "--help"])
        assert "--ref-temps" in result.output

    def test_compare_help_lists_fields_option(self):
        result = runner.invoke(app, ["compare", "--help"])
        assert "--fields" in result.output

    def test_compare_help_shows_stations_argument(self):
        result = runner.invoke(app, ["compare", "--help"])
        # stations is a Typer Argument; its help text should appear
        assert "STATIONS" in result.output.upper() or "station" in result.output.lower()


class TestCompareValidation:
    def test_fewer_than_two_stations_exits_nonzero(self):
        """Passing a single station should fail gracefully."""
        result = runner.invoke(app, [
            "compare", "KORD",
            "--start", "2022-01-01",
            "--end", "2022-12-31",
        ])
        assert result.exit_code != 0

    def test_missing_start_exits_nonzero(self):
        result = runner.invoke(app, ["compare", "KORD", "KPHX", "--end", "2022-12-31"])
        assert result.exit_code != 0

    def test_missing_end_exits_nonzero(self):
        result = runner.invoke(app, ["compare", "KORD", "KPHX", "--start", "2022-01-01"])
        assert result.exit_code != 0


class TestRunCommandStillWorks:
    def test_run_help_exits_zero(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0

    def test_run_help_shows_ref_temp(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--ref-temp" in result.output

    def test_run_help_shows_fields(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--fields" in result.output
