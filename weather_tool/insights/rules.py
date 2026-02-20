"""Deterministic risk flag engine for the Decision AI layer.

All evaluation is purely dict-based — no pandas, no DataFrames.
Missing packet fields silently skip their respective flags.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ── Recommendation taxonomy ────────────────────────────────────────────────────

@dataclass
class Recommendation:
    """A structured engineering recommendation."""

    rec_id: str
    title: str
    severity: str   # "high" | "medium" | "low"
    rationale: str
    evidence: list[dict[str, Any]] = field(default_factory=list)


# ── Decision profile ───────────────────────────────────────────────────────────

@dataclass
class DecisionProfile:
    """Threshold configuration for a named engineering decision context."""

    name: str

    # Tower / Moisture
    wb_p996_tower_high: float       # wb_p996 ≥ this → "high" tower rejection flag
    wb_p996_tower_medium: float     # wb_p996 ≥ this → "medium"
    tower_stress_78_sum_high: float
    tower_stress_78_sum_medium: float
    wb_mean_72h_high: float

    # Economizer
    wec_feasible_pct_low: float    # < this → "low econ opportunity"
    wec_feasible_pct_medium: float # ≥ this → "high econ opportunity" (positive flag)
    air_econ_hours_low: float

    # Freeze
    freeze_hours_high: float
    freeze_hours_medium: float
    freeze_event_max_duration_high: float
    freeze_shoulder_high: float

    # Heat / Tdb
    tdb_p996_high: float
    tdb_p996_medium: float
    tdb_mean_72h_high: float


PROFILES: dict[str, DecisionProfile] = {
    "datacenter": DecisionProfile(
        name="datacenter",
        # Cooling towers at datacenters are typically rated for 78°F WB
        wb_p996_tower_high=80.0,
        wb_p996_tower_medium=78.0,
        tower_stress_78_sum_high=400.0,
        tower_stress_78_sum_medium=100.0,
        wb_mean_72h_high=78.0,
        # WEC: datacenters value maximum uptime waterside economizer
        wec_feasible_pct_low=0.20,
        wec_feasible_pct_medium=0.50,
        air_econ_hours_low=500.0,
        # Freeze — chilled water pipe risk even in well-insulated sites
        freeze_hours_high=1000.0,
        freeze_hours_medium=300.0,
        freeze_event_max_duration_high=48.0,
        freeze_shoulder_high=100.0,
        # Heat
        tdb_p996_high=105.0,
        tdb_p996_medium=95.0,
        tdb_mean_72h_high=95.0,
    ),
    "economizer_first": DecisionProfile(
        name="economizer_first",
        wb_p996_tower_high=82.0,
        wb_p996_tower_medium=78.0,
        tower_stress_78_sum_high=600.0,
        tower_stress_78_sum_medium=200.0,
        wb_mean_72h_high=80.0,
        # Stricter econ thresholds — econ feasibility is the primary concern
        wec_feasible_pct_low=0.30,
        wec_feasible_pct_medium=0.60,
        air_econ_hours_low=1000.0,
        freeze_hours_high=1500.0,
        freeze_hours_medium=500.0,
        freeze_event_max_duration_high=72.0,
        freeze_shoulder_high=150.0,
        tdb_p996_high=105.0,
        tdb_p996_medium=95.0,
        tdb_mean_72h_high=95.0,
    ),
    "freeze_sensitive": DecisionProfile(
        name="freeze_sensitive",
        wb_p996_tower_high=82.0,
        wb_p996_tower_medium=80.0,
        tower_stress_78_sum_high=800.0,
        tower_stress_78_sum_medium=300.0,
        wb_mean_72h_high=78.0,
        wec_feasible_pct_low=0.20,
        wec_feasible_pct_medium=0.50,
        air_econ_hours_low=500.0,
        # Much stricter freeze thresholds
        freeze_hours_high=500.0,
        freeze_hours_medium=100.0,
        freeze_event_max_duration_high=24.0,
        freeze_shoulder_high=50.0,
        tdb_p996_high=105.0,
        tdb_p996_medium=95.0,
        tdb_mean_72h_high=95.0,
    ),
}


# ── Flag evaluation ────────────────────────────────────────────────────────────

def _flag(flag_id: str, severity: str, confidence: str,
          metric: str, value: float, threshold: float, notes: str) -> dict[str, Any]:
    return {
        "flag_id":    flag_id,
        "severity":   severity,
        "confidence": confidence,
        "evidence":   [{"metric": metric, "value": value, "threshold": threshold}],
        "notes":      notes,
    }


def _is_valid(v: Any) -> bool:
    """Return True if v is a non-None, non-NaN number."""
    if v is None:
        return False
    try:
        return not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def evaluate_station_flags(
    packet: dict[str, Any],
    profile: DecisionProfile,
) -> tuple[list[dict[str, Any]], list[Recommendation]]:
    """Evaluate risk flags from a station packet.

    Parameters
    ----------
    packet : station packet dict (must have design_conditions, operational_efficiency,
             freeze_risk, quality sub-dicts).  Missing keys are silently skipped.
    profile : DecisionProfile to use for thresholds.

    Returns
    -------
    (flags, recommendations) — both are lists; flags contain structured dicts,
    recommendations contain Recommendation dataclass instances.
    """
    dc  = packet.get("design_conditions", {}) or {}
    oe  = packet.get("operational_efficiency", {}) or {}
    fr  = packet.get("freeze_risk", {}) or {}
    qua = packet.get("quality", {}) or {}

    flags: list[dict[str, Any]] = []
    recs: list[Recommendation] = []

    data_quality_low = bool(qua.get("missing_data_warning", False))
    confidence = "low" if data_quality_low else "high"

    # ── Tower heat rejection ───────────────────────────────────────────────────
    wb_p996 = dc.get("wb_p996")
    if _is_valid(wb_p996):
        v = float(wb_p996)  # type: ignore[arg-type]
        if v >= profile.wb_p996_tower_high:
            sev = "high"
        elif v >= profile.wb_p996_tower_medium:
            sev = "medium"
        else:
            sev = None
        if sev:
            flags.append(_flag(
                "tower_heat_rejection", sev, confidence,
                "wb_p996", v, profile.wb_p996_tower_medium,
                f"Wet-bulb design extreme (p99.6) of {v:.1f}°F approaches or exceeds "
                f"typical {profile.wb_p996_tower_medium:.0f}°F tower design point.",
            ))
            recs.append(Recommendation(
                rec_id="size_tower_for_wb_p996",
                title=f"Size cooling towers above {v:.0f}°F WB design",
                severity=sev,
                rationale=(
                    f"Local wb_p996 of {v:.1f}°F {'exceeds' if sev == 'high' else 'approaches'} "
                    f"the {profile.wb_p996_tower_medium:.0f}°F tower design threshold. "
                    "Specify tower capacity at or above the site p99.6 wet-bulb; "
                    "consider a hybrid dry cooler for extreme-weather backup."
                ),
                evidence=[{"metric": "wb_p996", "value": v, "threshold": profile.wb_p996_tower_medium}],
            ))

    # ── Tower stress hours (accumulated) ──────────────────────────────────────
    stress78 = oe.get("tower_stress_hours_wb_gt_78_sum")
    if _is_valid(stress78):
        v = float(stress78)  # type: ignore[arg-type]
        if v >= profile.tower_stress_78_sum_high:
            sev = "high"
        elif v >= profile.tower_stress_78_sum_medium:
            sev = "medium"
        else:
            sev = None
        if sev:
            flags.append(_flag(
                "tower_stress_elevated", sev, confidence,
                "tower_stress_hours_wb_gt_78_sum", v, profile.tower_stress_78_sum_medium,
                f"Accumulated tower stress hours at Twb ≥ 78°F total {v:.0f} h over the analysis window.",
            ))
            recs.append(Recommendation(
                rec_id="increase_tower_capacity_for_stress",
                title="Increase cooling tower capacity margin for stress hours",
                severity=sev,
                rationale=(
                    f"The site accumulates {v:.0f} h with Twb ≥ 78°F. "
                    "Increase tower design safety margin or add supplemental cooling "
                    "capacity to maintain approach temperature during peak periods."
                ),
                evidence=[{"metric": "tower_stress_hours_wb_gt_78_sum", "value": v,
                           "threshold": profile.tower_stress_78_sum_medium}],
            ))

    # ── Wet-bulb persistence risk ──────────────────────────────────────────────
    wb72 = dc.get("wb_mean_72h_max")
    if _is_valid(wb72):
        v = float(wb72)  # type: ignore[arg-type]
        if v >= profile.wb_mean_72h_high:
            flags.append(_flag(
                "wb_persistence_risk", "high", confidence,
                "wb_mean_72h_max", v, profile.wb_mean_72h_high,
                f"Worst sustained 72h mean wet-bulb of {v:.1f}°F indicates multi-day heat rejection stress.",
            ))
            recs.append(Recommendation(
                rec_id="design_for_wb_persistence",
                title="Design for multi-day wet-bulb persistence events",
                severity="high",
                rationale=(
                    f"The worst 72h mean wet-bulb reaches {v:.1f}°F. "
                    "Ensure tower thermal capacity and water supply account for sustained "
                    "multi-day operation near peak wet-bulb."
                ),
                evidence=[{"metric": "wb_mean_72h_max", "value": v,
                           "threshold": profile.wb_mean_72h_high}],
            ))

    # ── WEC low opportunity ────────────────────────────────────────────────────
    wec_pct = oe.get("wec_feasible_pct_over_window")
    if _is_valid(wec_pct):
        v = float(wec_pct)  # type: ignore[arg-type]
        if v < profile.wec_feasible_pct_low:
            flags.append(_flag(
                "wec_low_opportunity", "medium", confidence,
                "wec_feasible_pct_over_window", v, profile.wec_feasible_pct_low,
                f"WEC proxy feasibility is {v * 100:.1f}% of wetbulb hours — limited waterside economizer opportunity.",
            ))
            recs.append(Recommendation(
                rec_id="evaluate_wec_viability",
                title="Evaluate waterside economizer viability",
                severity="medium",
                rationale=(
                    f"WEC feasibility is only {v * 100:.1f}% of hours with wet-bulb data. "
                    "A waterside economizer may not provide significant energy savings at this location; "
                    "consider airside economizer as primary free-cooling strategy instead."
                ),
                evidence=[{"metric": "wec_feasible_pct_over_window", "value": v,
                           "threshold": profile.wec_feasible_pct_low}],
            ))
        elif v >= profile.wec_feasible_pct_medium:
            flags.append(_flag(
                "wec_high_opportunity", "low", confidence,
                "wec_feasible_pct_over_window", v, profile.wec_feasible_pct_medium,
                f"WEC proxy feasibility is {v * 100:.1f}% — strong waterside economizer opportunity.",
            ))
            recs.append(Recommendation(
                rec_id="prioritize_wec",
                title="Prioritize waterside economizer in cooling design",
                severity="low",
                rationale=(
                    f"WEC feasibility reaches {v * 100:.1f}% of hours with wet-bulb data. "
                    "A plate-frame waterside economizer should be a primary cooling strategy to "
                    "maximize PUE at this location."
                ),
                evidence=[{"metric": "wec_feasible_pct_over_window", "value": v,
                           "threshold": profile.wec_feasible_pct_medium}],
            ))

    # ── Airside econ low opportunity ───────────────────────────────────────────
    ae_hrs = oe.get("air_econ_hours_sum")
    if _is_valid(ae_hrs):
        v = float(ae_hrs)  # type: ignore[arg-type]
        if v < profile.air_econ_hours_low:
            flags.append(_flag(
                "air_econ_low", "medium", confidence,
                "air_econ_hours_sum", v, profile.air_econ_hours_low,
                f"Airside econ hours total {v:.0f} h — limited free-cooling from airside economizer.",
            ))
            recs.append(Recommendation(
                rec_id="evaluate_airside_econ",
                title="Evaluate airside economizer effectiveness",
                severity="medium",
                rationale=(
                    f"Total airside economizer hours are {v:.0f} h. "
                    "Verify that installed airside economizer equipment meets energy code "
                    "requirements despite the limited climate opportunity."
                ),
                evidence=[{"metric": "air_econ_hours_sum", "value": v,
                           "threshold": profile.air_econ_hours_low}],
            ))

    # ── Freeze risk ────────────────────────────────────────────────────────────
    freeze_hrs = fr.get("freeze_hours_sum")
    if _is_valid(freeze_hrs):
        v = float(freeze_hrs)  # type: ignore[arg-type]
        if v >= profile.freeze_hours_high:
            sev = "high"
        elif v >= profile.freeze_hours_medium:
            sev = "medium"
        else:
            sev = None
        if sev:
            flags.append(_flag(
                "freeze_risk", sev, confidence,
                "freeze_hours_sum", v, profile.freeze_hours_medium,
                f"Total freeze hours of {v:.0f} h indicate meaningful below-freezing exposure.",
            ))
            recs.append(Recommendation(
                rec_id="freeze_protection_measures",
                title="Implement freeze protection for water-based systems",
                severity=sev,
                rationale=(
                    f"The site accumulates {v:.0f} h below the freeze threshold. "
                    "Install heat tracing, insulation, and glycol systems for all exposed "
                    "water-based cooling infrastructure."
                ),
                evidence=[{"metric": "freeze_hours_sum", "value": v,
                           "threshold": profile.freeze_hours_medium}],
            ))

    # ── Extended freeze event ─────────────────────────────────────────────────
    max_event = fr.get("freeze_event_max_duration_hours_max")
    if _is_valid(max_event):
        v = float(max_event)  # type: ignore[arg-type]
        if v >= profile.freeze_event_max_duration_high:
            flags.append(_flag(
                "freeze_event_extended", "high", confidence,
                "freeze_event_max_duration_hours_max", v, profile.freeze_event_max_duration_high,
                f"Longest freeze event of {v:.1f} h requires extended continuous freeze protection.",
            ))
            recs.append(Recommendation(
                rec_id="design_for_extended_freeze",
                title=f"Design freeze protection for events ≥ {v:.0f} h",
                severity="high",
                rationale=(
                    f"The longest observed contiguous freeze event is {v:.1f} h. "
                    "Ensure heat tracing, backup glycol capacity, and generator runtime "
                    "are sufficient to sustain freeze protection through multi-day events."
                ),
                evidence=[{"metric": "freeze_event_max_duration_hours_max", "value": v,
                           "threshold": profile.freeze_event_max_duration_high}],
            ))

    # ── Shoulder-season freeze exposure ───────────────────────────────────────
    shoulder_hrs = fr.get("freeze_hours_shoulder_sum")
    if _is_valid(shoulder_hrs):
        v = float(shoulder_hrs)  # type: ignore[arg-type]
        if v >= profile.freeze_shoulder_high:
            flags.append(_flag(
                "freeze_shoulder_exposure", "medium", confidence,
                "freeze_hours_shoulder_sum", v, profile.freeze_shoulder_high,
                f"Shoulder-season freeze hours of {v:.0f} h increase risk during commissioning / maintenance windows.",
            ))
            recs.append(Recommendation(
                rec_id="shoulder_season_freeze_protocol",
                title="Add shoulder-season freeze protection protocol",
                severity="medium",
                rationale=(
                    f"The site has {v:.0f} h of below-freeze exposure in shoulder months. "
                    "Establish maintenance windows to avoid exposing un-pressurized water systems "
                    "during spring and fall shoulder seasons."
                ),
                evidence=[{"metric": "freeze_hours_shoulder_sum", "value": v,
                           "threshold": profile.freeze_shoulder_high}],
            ))

    # ── Heat design extreme (dry-bulb) ────────────────────────────────────────
    tdb_p996 = dc.get("tdb_p996")
    if _is_valid(tdb_p996):
        v = float(tdb_p996)  # type: ignore[arg-type]
        if v >= profile.tdb_p996_high:
            sev = "high"
        elif v >= profile.tdb_p996_medium:
            sev = "medium"
        else:
            sev = None
        if sev:
            flags.append(_flag(
                "heat_design_extreme", sev, confidence,
                "tdb_p996", v, profile.tdb_p996_medium,
                f"Dry-bulb design extreme (p99.6) of {v:.1f}°F imposes high sensible heat load on cooling systems.",
            ))
            recs.append(Recommendation(
                rec_id="size_for_tdb_p996",
                title=f"Size mechanical cooling for {v:.0f}°F Tdb design extreme",
                severity=sev,
                rationale=(
                    f"The local tdb_p996 of {v:.1f}°F exceeds the {profile.tdb_p996_medium:.0f}°F threshold. "
                    "Ensure mechanical cooling capacity is sized for the site-specific design dry-bulb, "
                    "not generic ASHRAE climate zone defaults."
                ),
                evidence=[{"metric": "tdb_p996", "value": v, "threshold": profile.tdb_p996_medium}],
            ))

    # ── Dry-bulb persistence ──────────────────────────────────────────────────
    tdb72 = dc.get("tdb_mean_72h_max")
    if _is_valid(tdb72):
        v = float(tdb72)  # type: ignore[arg-type]
        if v >= profile.tdb_mean_72h_high:
            flags.append(_flag(
                "tdb_persistence", "medium", confidence,
                "tdb_mean_72h_max", v, profile.tdb_mean_72h_high,
                f"Worst 72h mean dry-bulb of {v:.1f}°F indicates sustained sensible heat load.",
            ))
            recs.append(Recommendation(
                rec_id="design_for_tdb_persistence",
                title="Design cooling for sustained dry-bulb heat events",
                severity="medium",
                rationale=(
                    f"The worst 72h mean dry-bulb reaches {v:.1f}°F. "
                    "Verify that CRAH/CRAC units and mechanical cooling can sustain "
                    "rated capacity through multi-day dry-bulb events."
                ),
                evidence=[{"metric": "tdb_mean_72h_max", "value": v,
                           "threshold": profile.tdb_mean_72h_high}],
            ))

    # ── Data quality warning ───────────────────────────────────────────────────
    if data_quality_low:
        flags.append({
            "flag_id":    "data_quality_warning",
            "severity":   "medium",
            "confidence": "high",
            "evidence":   [],
            "notes":      "Elevated missing data or low wet-bulb availability — metric-based flags have reduced confidence.",
        })
        recs.append(Recommendation(
            rec_id="verify_data_quality",
            title="Verify data quality before finalizing design parameters",
            severity="medium",
            rationale=(
                "Missing data or low wet-bulb availability was detected. "
                "Supplement with additional data sources or extend the analysis window "
                "before making final equipment sizing decisions."
            ),
            evidence=[],
        ))

    return flags, recs
