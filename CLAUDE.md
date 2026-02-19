# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Optional extras
pip install -e ".[llm]"        # OpenAI LLM narrative (requires OPENAI_API_KEY)
pip install -e ".[viz]"        # matplotlib wind rose PNGs
pip install -e ".[streamlit]"  # Streamlit UI (not yet implemented)

# Run tests
pytest -q
pytest --cov=weather_tool      # with coverage

# Run single test file
pytest tests/test_freeze.py -q

# Run the tool (after install)
weather-tool run --mode iem --station-id KORD --start 2022-01-01 --end 2023-12-31 --ref-temp 65
weather-tool compare KORD KPHX KIAD --start 2018-01-01 --end 2023-12-31 --ref-temps 80,85

# Alternative without install (Windows py launcher)
py -3.13 -m weather_tool.cli run --mode iem --station-id KORD --start 2022-01-01 --end 2023-12-31 --ref-temp 65
py -3.13 -m pytest
```

No linter, formatter, or type-checker is configured. No CI/CD pipeline exists.

## Architecture

### Pipeline Pattern

The system follows a strict **pure-pipeline** architecture:

1. **CLI** (`cli.py`) parses args into a `RunConfig` dataclass, calls `run_station_pipeline()`, then saves outputs via `storage/io.py`
2. **Pipeline** (`pipeline.py`) is the pure computational core — takes a validated `RunConfig`, returns a `StationResult` dataclass. **No file I/O happens inside the pipeline.**
3. **Connectors** (`connectors/`) fetch raw data (IEM API or CSV file) into DataFrames with a common schema: `timestamp`, `temp`, `station_id` columns
4. **Core modules** (`core/`) are all pure functions with no side effects — normalize, compute metrics, aggregate yearly summaries
5. **Insights** (`insights/`) generate deterministic markdown reports; LLM narrative is opt-in and only interprets pre-computed metrics
6. **Storage** (`storage/io.py`) handles all file I/O — CSV, Parquet, JSON, Markdown outputs

### Key Design Rules

- **Deterministic first:** All metrics are computed via pure functions. The LLM (opt-in via `--llm`) only generates narrative text from already-computed summaries — it never computes metrics.
- **Step-function convention:** All hour-based metrics use `count_of_qualifying_records × (dt_minutes / 60)`. NaN values are excluded (not treated as 0).
- **RunConfig is the single source of truth:** A flat `@dataclass` with a `validate()` method, threaded through the entire call stack. All tunable parameters live here.
- **Output file tagging:** All output filenames use `{station_id}_{start}_{end}` pattern via `RunConfig.file_tag`.
- **Optional dependencies degrade gracefully:** matplotlib (wind roses), openai (LLM narrative), and streamlit are optional extras that don't break the tool when absent.

### Two CLI Commands

- `weather-tool run` — single-station analysis (IEM or CSV mode)
- `weather-tool compare` — multi-station IEM comparison with scoring (heat 35%, moisture 35%, freeze 20%, quality 10%)

### Data Source

IEM (Iowa Environmental Mesonet) ASOS API at `mesonet.agron.iastate.edu`. Six fields available: `tmpf`, `dwpf`, `relh`, `sknt`, `drct`, `gust`. When `tmpf` and `relh` are present, wet-bulb is auto-computed via Stull (2011) approximation.

### Test Conventions

Tests live in `tests/` and use synthetic in-memory DataFrames (e.g., `pd.date_range` + sinusoidal temps). No network calls or file fixtures needed. Tests mirror the module they cover (e.g., `test_freeze.py` tests `core/freeze.py`).

### Current Roadmap

    Phase 1: DX & Performance - Implement Parquet caching and RunConfig profiles.

    Phase 2: Design Day Engine - ASHRAE percentiles (p99.6,p98) and MCWB logic.

    Phase 3: Simulation - Cooling tower approach and economizer hours.

    Phase 4: UI - Streamlit dashboard for comparison.