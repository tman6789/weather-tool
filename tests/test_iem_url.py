"""Unit tests for the IEM URL builder."""

from __future__ import annotations

from datetime import date
from urllib.parse import parse_qs, urlparse

from weather_tool.connectors.iem_connector import _build_url


class TestBuildUrl:
    def _parse(self, url: str) -> dict:
        parsed = urlparse(url)
        return parse_qs(parsed.query)

    def test_single_field_contains_data_tmpf(self):
        """fields=['tmpf'] → exactly one data=tmpf parameter."""
        url = _build_url("KORD", date(2023, 1, 1), date(2023, 12, 31), ["tmpf"])
        qs = self._parse(url)
        assert qs.get("data") == ["tmpf"]

    def test_multi_field_contains_all_data_params(self):
        """fields=['tmpf','dwpf','relh'] → three data= params in correct order."""
        url = _build_url("KORD", date(2023, 1, 1), date(2023, 12, 31), ["tmpf", "dwpf", "relh"])
        qs = self._parse(url)
        assert qs.get("data") == ["tmpf", "dwpf", "relh"]

    def test_all_six_fields(self):
        """All six HVAC-relevant fields are included when requested."""
        fields = ["tmpf", "dwpf", "relh", "sknt", "drct", "gust"]
        url = _build_url("KORD", date(2023, 1, 1), date(2023, 12, 31), fields)
        qs = self._parse(url)
        assert qs.get("data") == fields

    def test_station_in_url(self):
        """station parameter is set correctly."""
        url = _build_url("KLAX", date(2022, 6, 15), date(2022, 9, 30), ["tmpf"])
        qs = self._parse(url)
        assert qs.get("station") == ["KLAX"]

    def test_date_parameters(self):
        """year1/month1/day1 and year2/month2/day2 are set correctly."""
        url = _build_url("KORD", date(2022, 3, 5), date(2023, 11, 20), ["tmpf"])
        qs = self._parse(url)
        assert qs["year1"] == ["2022"]
        assert qs["month1"] == ["3"]
        assert qs["day1"] == ["5"]
        assert qs["year2"] == ["2023"]
        assert qs["month2"] == ["11"]
        assert qs["day2"] == ["20"]

    def test_fixed_params_present(self):
        """Mandatory fixed parameters (tz, format, report_type, etc.) are present."""
        url = _build_url("KORD", date(2023, 1, 1), date(2023, 12, 31), ["tmpf"])
        qs = self._parse(url)
        assert qs.get("tz") == ["Etc/UTC"]
        assert qs.get("format") == ["comma"]
        assert qs.get("report_type") == ["3"]
        assert qs.get("missing") == ["empty"]

    def test_url_base(self):
        """URL starts with the correct IEM base URL."""
        url = _build_url("KORD", date(2023, 1, 1), date(2023, 12, 31), ["tmpf"])
        assert url.startswith("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py")
