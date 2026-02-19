"""Tests for the Parquet caching layer (weather_tool/connectors/cache.py + iem_connector.py)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from weather_tool.connectors.cache import (
    ALL_IEM_FIELDS,
    CACHE_COLUMNS,
    CACHE_SCHEMA_VERSION,
    _cache_path,
    cache_dir_versioned,
    read_cached_year,
    write_cached_year,
)
from weather_tool.connectors.iem_connector import _contiguous_ranges
from weather_tool.config import RunConfig


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_year_df(station: str, year: int, n_rows: int = 100) -> pd.DataFrame:
    """Create a synthetic cached-format DataFrame for one year."""
    ts = pd.date_range(
        f"{year}-01-01", periods=n_rows, freq="h", tz="UTC"
    )
    df = pd.DataFrame({"timestamp": ts, "station_id": station})
    rng = np.random.default_rng(year)
    for fld in ALL_IEM_FIELDS:
        df[fld] = rng.uniform(0, 100, size=n_rows)
    return df


# ── cache.py unit tests ─────────────────────────────────────────────────────


class TestCachePath:
    """_cache_path includes version directory."""

    def test_format(self, tmp_path: Path):
        p = _cache_path(tmp_path, "KORD", 2022)
        assert p == tmp_path / f"v{CACHE_SCHEMA_VERSION}" / "KORD_2022.parquet"

    def test_versioned_dir(self, tmp_path: Path):
        vd = cache_dir_versioned(tmp_path)
        assert vd == tmp_path / f"v{CACHE_SCHEMA_VERSION}"


class TestWriteReadRoundtrip:
    """write_cached_year → read_cached_year preserves data."""

    def test_roundtrip(self, tmp_path: Path):
        df = _make_year_df("KORD", 2022, n_rows=50)
        write_cached_year(tmp_path, "KORD", 2022, df)
        result = read_cached_year(tmp_path, "KORD", 2022)

        assert result is not None
        assert len(result) == 50
        assert set(CACHE_COLUMNS).issubset(result.columns)
        pd.testing.assert_series_equal(
            result["tmpf"].reset_index(drop=True),
            df["tmpf"].reset_index(drop=True),
        )

    def test_filters_to_year(self, tmp_path: Path):
        """write_cached_year only stores rows for the requested year."""
        df1 = _make_year_df("KORD", 2022, 50)
        df2 = _make_year_df("KORD", 2023, 30)
        combined = pd.concat([df1, df2], ignore_index=True)

        write_cached_year(tmp_path, "KORD", 2022, combined)
        result = read_cached_year(tmp_path, "KORD", 2022)

        assert result is not None
        assert len(result) == 50
        assert (result["timestamp"].dt.year == 2022).all()

    def test_empty_year_not_written(self, tmp_path: Path):
        """If the df has no rows for the target year, no file is created."""
        df = _make_year_df("KORD", 2022, 50)
        write_cached_year(tmp_path, "KORD", 2023, df)

        p = _cache_path(tmp_path, "KORD", 2023)
        assert not p.exists()


class TestCacheMiss:
    """read_cached_year returns None for missing / corrupt / incomplete files."""

    def test_missing_file(self, tmp_path: Path):
        assert read_cached_year(tmp_path, "KORD", 2022) is None

    def test_incomplete_fields(self, tmp_path: Path):
        """A Parquet with missing IEM columns is treated as a cache miss + deleted."""
        vdir = cache_dir_versioned(tmp_path)
        vdir.mkdir(parents=True)
        p = _cache_path(tmp_path, "KORD", 2022)

        # Write a partial file (missing most IEM fields)
        partial = pd.DataFrame({
            "timestamp": pd.date_range("2022-01-01", periods=5, freq="h", tz="UTC"),
            "station_id": "KORD",
            "tmpf": [70.0] * 5,
        })
        partial.to_parquet(p, index=False, engine="pyarrow")
        assert p.exists()

        result = read_cached_year(tmp_path, "KORD", 2022)
        assert result is None
        assert not p.exists()  # self-healing: file was deleted

    def test_corrupt_file(self, tmp_path: Path):
        """Garbage bytes in a .parquet file → cache miss + file deleted."""
        vdir = cache_dir_versioned(tmp_path)
        vdir.mkdir(parents=True)
        p = _cache_path(tmp_path, "KORD", 2022)
        p.write_bytes(b"this is not a parquet file")
        assert p.exists()

        result = read_cached_year(tmp_path, "KORD", 2022)
        assert result is None
        assert not p.exists()  # self-healing


# ── _contiguous_ranges tests ────────────────────────────────────────────────


class TestContiguousRanges:
    """_contiguous_ranges groups years into contiguous ranges."""

    def test_empty(self):
        assert _contiguous_ranges([]) == []

    def test_single_year(self):
        assert _contiguous_ranges([2022]) == [(2022, 2022)]

    def test_contiguous(self):
        assert _contiguous_ranges([2020, 2021, 2022]) == [(2020, 2022)]

    def test_gap(self):
        assert _contiguous_ranges([2020, 2023]) == [(2020, 2020), (2023, 2023)]

    def test_mixed(self):
        assert _contiguous_ranges([2018, 2019, 2022, 2023, 2024]) == [
            (2018, 2019),
            (2022, 2024),
        ]

    def test_unsorted_input(self):
        assert _contiguous_ranges([2023, 2020, 2021]) == [(2020, 2021), (2023, 2023)]


# ── Integration tests (mocked HTTP) ─────────────────────────────────────────


def _fake_iem_response(station_id: str, start: date, end: date) -> str:
    """Build a fake IEM CSV response text."""
    ts = pd.date_range(start, end, freq="h", tz="UTC")
    rows = []
    for t in ts:
        rows.append(f"{station_id},{t.strftime('%Y-%m-%d %H:%M')},70.0,55.0,60.0,10.0,180.0,15.0")
    header = "station,valid,tmpf,dwpf,relh,sknt,drct,gust"
    return header + "\n" + "\n".join(rows)


class _FakeResponse:
    """Minimal mock for requests.Response."""

    def __init__(self, text: str):
        self.text = text
        self.status_code = 200

    def raise_for_status(self) -> None:
        pass


class TestCacheIntegration:
    """Integration tests verifying cache hit/miss behaviour with mocked HTTP."""

    def test_second_call_uses_cache(self, tmp_path: Path):
        """First call fetches from IEM; second call uses cache (no HTTP)."""
        from weather_tool.connectors.iem_connector import load_iem

        cfg = RunConfig(
            mode="iem",
            station_id="KTEST",
            start=date(2022, 1, 1),
            end=date(2022, 12, 31),
            ref_temp=65.0,
            fields=["tmpf", "dwpf", "relh", "sknt", "drct", "gust"],
            cache_dir=tmp_path,
        )

        fake_text = _fake_iem_response("KTEST", date(2022, 1, 1), date(2022, 12, 31))
        fake_resp = _FakeResponse(fake_text)

        with patch("weather_tool.connectors.iem_connector.requests.get", return_value=fake_resp) as mock_get:
            # First call — cache miss, HTTP fetch
            df1 = load_iem(cfg)
            assert mock_get.call_count == 1
            assert len(df1) > 0

            # Second call — cache hit, no HTTP
            df2 = load_iem(cfg)
            assert mock_get.call_count == 1  # still 1, no new call
            assert len(df2) == len(df1)

    def test_no_cache_flag_always_fetches(self, tmp_path: Path):
        """With no_cache=True, every call hits HTTP."""
        from weather_tool.connectors.iem_connector import load_iem

        cfg = RunConfig(
            mode="iem",
            station_id="KTEST",
            start=date(2022, 1, 1),
            end=date(2022, 12, 31),
            ref_temp=65.0,
            fields=["tmpf"],
            cache_dir=tmp_path,
            no_cache=True,
        )

        fake_text = _fake_iem_response("KTEST", date(2022, 1, 1), date(2022, 12, 31))
        fake_resp = _FakeResponse(fake_text)

        with patch("weather_tool.connectors.iem_connector.requests.get", return_value=fake_resp) as mock_get:
            load_iem(cfg)
            assert mock_get.call_count == 1
            load_iem(cfg)
            assert mock_get.call_count == 2  # fetched again

    def test_current_year_never_cached(self, tmp_path: Path):
        """Current UTC year is always fetched fresh and never written to cache."""
        from weather_tool.connectors.iem_connector import load_iem

        current_year = datetime.now(timezone.utc).year
        cfg = RunConfig(
            mode="iem",
            station_id="KTEST",
            start=date(current_year, 1, 1),
            end=date(current_year, 2, 1),
            ref_temp=65.0,
            fields=["tmpf", "dwpf", "relh", "sknt", "drct", "gust"],
            cache_dir=tmp_path,
        )

        fake_text = _fake_iem_response(
            "KTEST", date(current_year, 1, 1), date(current_year, 12, 31)
        )
        fake_resp = _FakeResponse(fake_text)

        with patch("weather_tool.connectors.iem_connector.requests.get", return_value=fake_resp) as mock_get:
            load_iem(cfg)
            assert mock_get.call_count == 1

            # Cache file should NOT exist for current year
            p = _cache_path(tmp_path, "KTEST", current_year)
            assert not p.exists()

            # Second call should fetch again (not cached)
            load_iem(cfg)
            assert mock_get.call_count == 2

    def test_partial_cache_fetches_only_missing(self, tmp_path: Path):
        """With 2022 cached and 2023 missing, only 2023 is fetched."""
        from weather_tool.connectors.iem_connector import load_iem

        # Pre-populate cache for 2022
        df_2022 = _make_year_df("KTEST", 2022, n_rows=200)
        write_cached_year(tmp_path, "KTEST", 2022, df_2022)

        cfg = RunConfig(
            mode="iem",
            station_id="KTEST",
            start=date(2022, 1, 1),
            end=date(2023, 12, 31),
            ref_temp=65.0,
            fields=["tmpf", "dwpf", "relh", "sknt", "drct", "gust"],
            cache_dir=tmp_path,
        )

        fake_text = _fake_iem_response("KTEST", date(2023, 1, 1), date(2023, 12, 31))
        fake_resp = _FakeResponse(fake_text)

        with patch("weather_tool.connectors.iem_connector.requests.get", return_value=fake_resp) as mock_get:
            df = load_iem(cfg)
            assert mock_get.call_count == 1
            assert len(df) > 0
            # Verify the fetch was for 2023 (full year), not 2022
            call_url = mock_get.call_args[0][0]
            assert "year1=2023" in call_url
            assert "year2=2023" in call_url

    def test_self_heal_corrupt_then_refetch(self, tmp_path: Path):
        """A corrupt cache file is deleted and refetched automatically."""
        from weather_tool.connectors.iem_connector import load_iem

        # Write a corrupt cache file for 2022
        vdir = cache_dir_versioned(tmp_path)
        vdir.mkdir(parents=True)
        corrupt_path = _cache_path(tmp_path, "KTEST", 2022)
        corrupt_path.write_bytes(b"corrupt data here")

        cfg = RunConfig(
            mode="iem",
            station_id="KTEST",
            start=date(2022, 6, 1),
            end=date(2022, 12, 31),
            ref_temp=65.0,
            fields=["tmpf"],
            cache_dir=tmp_path,
        )

        fake_text = _fake_iem_response("KTEST", date(2022, 1, 1), date(2022, 12, 31))
        fake_resp = _FakeResponse(fake_text)

        with patch("weather_tool.connectors.iem_connector.requests.get", return_value=fake_resp) as mock_get:
            df = load_iem(cfg)
            assert mock_get.call_count == 1  # had to refetch
            assert len(df) > 0

            # Cache should now be valid
            cached = read_cached_year(tmp_path, "KTEST", 2022)
            assert cached is not None
