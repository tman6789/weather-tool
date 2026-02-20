"""Formatting helpers for the Streamlit UI.

No weather math — pure string/display utilities only.
"""

from __future__ import annotations

SEV_ICON: dict[str, str] = {"high": "🔴", "medium": "🟡", "low": "🟢"}


def safe_fmt_float(v: object, fmt: str = ".1f") -> str:
    """Format a value as a float string, falling back to str() for non-numeric values.

    Handles None, NaN, strings, and any other non-numeric type without raising.
    """
    if isinstance(v, (int, float)) and v == v:  # v == v excludes NaN
        return format(v, fmt)
    return str(v) if v is not None else ""


def format_freeze_hours(freeze_hours_sum: float, years_covered: int, mode: str) -> str:
    """Return a human-readable freeze hours string for the given normalization mode.

    Parameters
    ----------
    freeze_hours_sum : aggregate freeze hours across all years.
    years_covered    : number of years in the analysis window.
    mode             : "per_year" or "aggregate".
    """
    if mode == "per_year" and years_covered > 0:
        avg = freeze_hours_sum / years_covered
        return f"{avg:,.0f} h/yr (avg over {years_covered} yrs)"
    return f"{freeze_hours_sum:,.0f} h total ({years_covered} yrs)"
