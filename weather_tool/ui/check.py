"""Smoke validator for weather-tool UI artifacts.

Usage
-----
    python -m weather_tool.ui.check [run_dir]

Validates:
  - At least one station_packet_*.json is findable under run_dir
  - Each packet contains the 8 required top-level keys

Exit codes
----------
  0  All packets found and valid
  1  No packets found, or at least one packet has missing/unreadable keys
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_KEYS: frozenset[str] = frozenset({
    "meta",
    "quality",
    "design_conditions",
    "operational_efficiency",
    "freeze_risk",
    "death_day",
    "risk_flags",
    "recommendations",
})


def check(run_dir: Path) -> bool:
    """Return True if all packets in run_dir are valid. Prints results to stdout."""
    packets = sorted(run_dir.rglob("station_packet_*.json"))
    if not packets:
        print(f"ERROR: No station_packet_*.json found in {run_dir}", file=sys.stderr)
        return False

    all_ok = True
    for p in packets:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"ERROR {p.name}: {exc}")
            all_ok = False
            continue
        missing = REQUIRED_KEYS - set(data.keys())
        if missing:
            print(f"WARN  {p.name}: missing keys {sorted(missing)}")
            all_ok = False
        else:
            print(f"OK    {p.name}")

    status = "All valid." if all_ok else "Warnings above — some packets have missing keys."
    print(f"\n{len(packets)} packet(s) checked. {status}")
    return all_ok


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs")
    ok = check(run_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
