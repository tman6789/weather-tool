"""Unit tests for weather_tool.ui.backend — flags_table() and load_summary_csv().

No Streamlit server required. Cached functions expose their unwrapped implementation
via the __wrapped__ attribute (set by @st.cache_data).
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# flags_table
# ---------------------------------------------------------------------------

def _minimal_packet(
    station_id: str = "KORD",
    years_covered: int = 6,
    freeze_hours_sum: float = 9600.0,
    flags: list | None = None,
) -> dict:
    return {
        "meta": {"station_id": station_id, "years_covered": years_covered},
        "freeze_risk": {"freeze_hours_sum": freeze_hours_sum},
        "risk_flags": flags if flags is not None else [],
    }


def _freeze_flag(flag_id: str = "freeze_hours_high", evidence: list | None = None) -> dict:
    return {
        "flag_id": flag_id,
        "severity": "high",
        "confidence": 0.9,
        "evidence": evidence if evidence is not None else [],
        "notes": "High freeze hours.",
    }


def _heat_flag() -> dict:
    return {
        "flag_id": "heat_stress_high",
        "severity": "medium",
        "confidence": 0.7,
        "evidence": [],
        "notes": "High heat stress.",
    }


def test_flags_table_returns_dataframe_for_empty_packets() -> None:
    from weather_tool.ui.backend import flags_table

    df = flags_table({})
    assert df.empty


def test_flags_table_no_evidence_crash() -> None:
    """Evidence with non-float and None values must not raise."""
    evidence = [
        {"metric": "freeze_hours", "value": None, "threshold": 1800.0},
        {"metric": "label", "value": "N/A", "threshold": None},
        {"metric": "count", "value": 42, "threshold": 30},
    ]
    packets = {"KORD": _minimal_packet(flags=[_freeze_flag(evidence=evidence)])}

    from weather_tool.ui.backend import flags_table

    df = flags_table(packets, "per_year")
    assert not df.empty
    assert "evidence_summary" in df.columns
    # Must not contain raw Python object repr
    assert "[object" not in df["evidence_summary"].iloc[0]
    assert df["evidence_summary"].iloc[0] != ""


def test_flags_table_per_year_normalization() -> None:
    """freeze_hours_norm should be sum / years for per_year mode."""
    packets = {"KORD": _minimal_packet(freeze_hours_sum=9600.0, years_covered=6,
                                        flags=[_freeze_flag()])}

    from weather_tool.ui.backend import flags_table

    df = flags_table(packets, "per_year")
    assert df["freeze_hours_norm"].iloc[0] == pytest.approx(1600.0)
    assert "1,600" in df["freeze_hours_display"].iloc[0]
    assert "yr" in df["freeze_hours_display"].iloc[0]


def test_flags_table_aggregate_normalization() -> None:
    """freeze_hours_norm should be the raw sum for aggregate mode."""
    packets = {"KORD": _minimal_packet(freeze_hours_sum=9600.0, years_covered=6,
                                        flags=[_freeze_flag()])}

    from weather_tool.ui.backend import flags_table

    df = flags_table(packets, "aggregate")
    assert df["freeze_hours_norm"].iloc[0] == pytest.approx(9600.0)
    assert "9,600" in df["freeze_hours_display"].iloc[0]
    assert "total" in df["freeze_hours_display"].iloc[0]


def test_flags_table_non_freeze_no_normalization() -> None:
    """Non-freeze flags must not have freeze normalization columns populated."""
    packets = {"KORD": _minimal_packet(flags=[_heat_flag()])}

    from weather_tool.ui.backend import flags_table

    df = flags_table(packets)
    assert df["freeze_hours_norm"].iloc[0] is None
    assert df["freeze_hours_display"].iloc[0] == ""
    assert df["normalization_mode"].iloc[0] == ""


def test_flags_table_scalar_columns_only() -> None:
    """All output columns must be scalar types — no lists or dicts."""
    import pandas as pd

    evidence = [{"metric": "wb_mean_72h_max", "value": 78.3, "threshold": 78.0}]
    packets = {"KORD": _minimal_packet(flags=[_freeze_flag(evidence=evidence), _heat_flag()])}

    from weather_tool.ui.backend import flags_table

    df = flags_table(packets, "per_year")
    for col in df.columns:
        for val in df[col]:
            assert not isinstance(val, (list, dict)), (
                f"Column '{col}' contains non-scalar value: {val!r}"
            )


def test_flags_table_mixed_stations() -> None:
    """Results from multiple stations should all appear in the output."""
    packets = {
        "KORD": _minimal_packet("KORD", flags=[_freeze_flag()]),
        "KPHX": _minimal_packet("KPHX", flags=[_heat_flag()]),
    }

    from weather_tool.ui.backend import flags_table

    df = flags_table(packets)
    assert set(df["station"].tolist()) == {"KORD", "KPHX"}


# ---------------------------------------------------------------------------
# load_summary_csv
# ---------------------------------------------------------------------------

def test_load_summary_csv_finds_matching_file(tmp_path) -> None:
    """Should find summary_KORD_*.csv under run_dir."""
    csv = tmp_path / "summary_KORD_2022-01-01_2023-12-31.csv"
    csv.write_text("year,tmax\n2022,95\n2023,98\n")

    from weather_tool.ui.backend import load_summary_csv

    result = load_summary_csv.__wrapped__(str(tmp_path), "KORD")
    assert result is not None
    assert list(result.columns) == ["year", "tmax"]
    assert len(result) == 2


def test_load_summary_csv_returns_none_for_missing(tmp_path) -> None:
    """Should return None when no matching CSV exists."""
    from weather_tool.ui.backend import load_summary_csv

    result = load_summary_csv.__wrapped__(str(tmp_path), "KORD")
    assert result is None


def test_load_summary_csv_station_id_specificity(tmp_path) -> None:
    """Should not return KPHX CSV when KORD is requested."""
    csv = tmp_path / "summary_KPHX_2022-01-01_2023-12-31.csv"
    csv.write_text("year,tmax\n2022,110\n")

    from weather_tool.ui.backend import load_summary_csv

    result = load_summary_csv.__wrapped__(str(tmp_path), "KORD")
    assert result is None
