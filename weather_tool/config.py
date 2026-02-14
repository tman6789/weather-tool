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
