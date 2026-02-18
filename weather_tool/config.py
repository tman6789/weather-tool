"""Central configuration dataclass for a weather-tool analysis run."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal


Units = Literal["agnostic", "F", "C", "K", "auto"]
Mode = Literal["csv", "iem"]

PARTIAL_COVERAGE_THRESHOLD = 0.98  # coverage_pct below this → partial flag
INTERVAL_CHANGE_RATIO = 0.20       # fraction of diffs that may deviate
INTERVAL_CHANGE_TOL = 0.10         # fractional tolerance around median dt

# Compare engine constants
FREEZE_THRESHOLD_F: float = 32.0
MISSING_DATA_WARNING_THRESHOLD: float = 0.02   # 2% missing timestamps
WETBULB_AVAIL_WARNING_THRESHOLD: float = 70.0  # 70% wet-bulb availability

SCORE_WEIGHTS: dict[str, float] = {
    "heat": 0.35, "moisture": 0.35, "freeze": 0.20, "quality": 0.10
}
HEAT_SCORE_WEIGHTS: dict[str, float] = {"hours_above_ref": 0.60, "t_p99": 0.40}
FREEZE_SCORE_WEIGHTS: dict[str, float] = {"freeze_hours": 0.70, "tmin_min": 0.30}

# Economizer / Tower defaults
AIR_ECON_THRESHOLD_F: float = 55.0
TOWER_APPROACH_F: float = 7.0
HX_APPROACH_F: float = 5.0          # heat-exchanger approach delta (WEC proxy)
# TOWER_RANGE_F = 10.0              # reserved for future use — not wired into any metric yet
CHW_SUPPLY_F: float = 44.0
WB_STRESS_THRESHOLDS_F: list[float] = [75.0, 78.0, 80.0]
ROLLING_COMPLETENESS_MIN_FRAC: float = 0.80
ECON_WB_COVERAGE_MIN_FRAC: float = 0.90  # min wetbulb coverage for high-confidence econ flag
FREEZE_CONFIDENCE_TEMP_AVAIL_MIN_FRAC: float = 0.90  # min temp availability for high-confidence freeze flag

IEM_UNITS: dict[str, str] = {
    "tmpf": "F",
    "dwpf": "F",
    "relh": "%",
    "sknt": "kt",
    "drct": "deg",
    "gust": "kt",
    "wetbulb_f": "F",
}


@dataclass
class RunConfig:
    """All parameters for a single analysis run."""

    mode: Mode = "csv"

    # CSV-mode fields
    input_path: Path | None = None
    timestamp_col: str = "timestamp"
    temp_col: str = "temp"

    # IEM-mode fields
    station_id: str | None = None
    fields: list[str] = field(default_factory=lambda: ["tmpf"])

    # Common fields
    start: date = field(default_factory=lambda: date(2020, 1, 1))
    end: date = field(default_factory=lambda: date(2020, 12, 31))
    ref_temp: float = 65.0
    units: Units = "agnostic"
    tz: str = "UTC"
    outdir: Path = Path("outputs")
    use_llm: bool = False
    verbose: bool = False

    # Economizer / tower params
    air_econ_threshold_f: float = 55.0
    tower_approach_f: float = 7.0
    hx_approach_f: float = 5.0          # heat-exchanger approach for WEC proxy
    # tower_range_f: float = 10.0       # reserved — not yet used in any metric
    chw_supply_f: float = 44.0
    wb_stress_thresholds_f: list[float] = field(default_factory=lambda: [75.0, 78.0, 80.0])

    # Freeze risk params
    freeze_threshold_f: float = 32.0
    freeze_min_event_hours: float = 3.0
    freeze_gap_tolerance_mult: float = 1.5
    freeze_shoulder_months: list[int] = field(default_factory=lambda: [3, 4, 10, 11])

    # Wind analytics
    wind_rose: bool = False
    wind_dir_bins: int = 16
    wind_speed_bins: list[float] = field(default_factory=lambda: [0.0, 5.0, 10.0, 15.0, 20.0, 30.0])
    wind_speed_units: str = "mph"
    wind_rose_slices: list[str] = field(default_factory=lambda: ["annual", "summer", "winter"])
    wind_event_metric: str = "wetbulb"       # "tdb" or "wetbulb"
    wind_event_thresholds: list[str] = field(default_factory=lambda: ["p99"])
    wind_event_min_hours: float = 0.0

    def validate(self) -> None:
        """Raise ValueError on invalid config combinations."""
        if self.mode == "csv" and self.input_path is None:
            raise ValueError("--input is required in csv mode")
        if self.mode == "iem" and not self.station_id:
            raise ValueError("--station-id is required in iem mode")
        if self.start > self.end:
            raise ValueError(f"start ({self.start}) must be <= end ({self.end})")

    @property
    def file_tag(self) -> str:
        """Return a tag string used in output filenames."""
        station = self.station_id or "csv"
        return f"{station}_{self.start}_{self.end}"
