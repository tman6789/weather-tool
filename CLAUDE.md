# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Optional extras
pip install -e ".[llm]"        # OpenAI LLM narrative (requires OPENAI_API_KEY)
pip install -e ".[viz]"        # matplotlib wind rose PNGs
pip install -e ".[ui]"         # Streamlit UI dashboard

# ── Streamlit UI ──────────────────────────────────────────────────────────────

# Install UI extra
pip install -e ".[ui]"

# Run the dashboard
streamlit run weather_tool/ui/app.py

# Load an existing run folder (no pipeline re-run, no network needed)
# → open sidebar → select "Load existing folder"
# → enter path:  outputs/
#     or a compare dir:  outputs/compare_2018-01-01_2023-12-31_3stations/
# → click "Load"

# Smoke-test: validate packet schema in a run folder
python -m weather_tool.ui.check outputs/
python -m weather_tool.ui.check "outputs/compare_2018-01-01_2023-12-31_3stations"

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
3. **Connectors** (`connectors/`) fetch raw data (IEM API or CSV file). The IEM connector applies a Parquet cache keyed by `{station_id}_{year}` under `.cache/v1/` (schema version in `connectors/cache.py:CACHE_SCHEMA_VERSION`). Corrupt or missing cache files are self-healed on next fetch.
4. **Core modules** (`core/`) are all pure functions with no side effects — normalize, compute metrics, aggregate yearly summaries
5. **Insights** (`insights/`) generate deterministic markdown reports; LLM narrative is opt-in and only interprets pre-computed metrics
6. **Storage** (`storage/io.py`) handles all file I/O — CSV, Parquet, JSON, Markdown outputs

### Key Design Rules

- **Deterministic first:** All metrics are computed via pure functions. The LLM (opt-in via `--llm`) only generates narrative text from already-computed summaries — it never computes metrics.
- **Step-function convention:** All hour-based metrics use `count_of_qualifying_records × (dt_minutes / 60)`. NaN values are excluded (not treated as 0).
- **RunConfig is the single source of truth:** A flat `@dataclass` with a `validate()` method, threaded through the entire call stack. All tunable parameters live here. Module-level constants in `config.py` provide defaults that mirror the `RunConfig` defaults.
- **Output file tagging:** All output filenames use `{station_id}_{start}_{end}` pattern via `RunConfig.file_tag`.
- **Optional dependencies degrade gracefully:** matplotlib (wind roses), openai (LLM narrative), and streamlit are optional extras that don't break the tool when absent.
- **Per-year interval re-inference:** When `interval_change_flag` is set, `build_yearly_summary()` re-infers `dt_minutes` on each year's deduplicated timestamps so that hour-scaled metrics scale correctly.

### Normalized DataFrame Schema

After `normalize_timestamps()` + `filter_window()`, the working DataFrame has:
- `timestamp` (tz-aware), `temp`, `station_id`, `_is_dup` (bool) — always present
- `tmpf`, `dwpf`, `relh`, `sknt`, `drct`, `gust` — IEM fields when requested
- `wetbulb_f` — auto-computed via Stull (2011) when both `tmpf`/`temp` and `relh`/`dwpf` are non-null
- `drct_deg`, `wind_speed_kt`, `wind_speed_mph`, `is_calm` — added by `core/wind.py:normalize_wind()` when direction/speed data are present

All downstream metrics operate on **deduplicated** rows (`_is_dup == False`).

### StationResult

`pipeline.StationResult` is the return type of `run_station_pipeline()`:
- `summary` — per-year DataFrame (the main output)
- `windowed` — normalized, window-filtered time series
- `interval_info` — dict from `infer_interval()`: `dt_minutes`, `p10`, `p90`, `interval_change_flag`, `interval_unknown_flag`, `unique_diff_counts`
- `quality_report`, `metadata` — dicts written to JSON
- `wind_results` — dict with `slices` and `events` sub-dicts (None if `--wind-rose` not set)
- `design_day` — 24-row DataFrame (None if `--design-day` not set)

### Two CLI Commands

- `weather-tool run` — single-station analysis (IEM or CSV mode). Outputs to `outputs/`.
- `weather-tool compare` — multi-station IEM comparison. Outputs to `outputs/{start}_{end}_{stations}/`. Runs `run_station_pipeline()` once per station, then calls `core/compare.py:build_compare_summary()` + `core/compare_scores.py:compute_scores()`.

### Compare Scoring (`core/compare_scores.py`)

All scores are min-max normalized across stations to [0, 100]. Weights (in `config.py:SCORE_WEIGHTS`):
- `heat` 35%: composite of `hours_above_ref` (60%) + `t_p99` (40%)
- `moisture` 35%: `wb_p99_median` (skipped / weight redistributed if wet-bulb unavailable)
- `freeze` 20%: composite of `freeze_hours` (70%) + `tmin_min` (30%, inverted)
- `quality` 10%: composite of missing %, coverage %, wet-bulb availability %, interval change flag

### Core Module Responsibilities

| Module | Responsibility |
|---|---|
| `core/normalize.py` | Timestamp parsing, deduplication, window filtering |
| `core/metrics.py` | `hours_above_ref`, `infer_interval`, `compute_wetbulb_f` |
| `core/aggregate.py` | `build_yearly_summary()` — orchestrates all per-year metric calls |
| `core/freeze.py` | Freeze hours, shoulder-season hours, event detection |
| `core/econ_tower.py` | Air-side econ hours, WEC proxy hours, tower stress hours, LWT proxy, rolling wb maxima |
| `core/extreme.py` | Rolling persistence (`compute_rolling_max`), exceedance hours, design-day profile |
| `core/wind.py` | Wind normalization, wind rose tables, co-occurrence event stats |
| `core/compare.py` | Multi-station aggregation (sums, medians, coverage stats) |
| `core/compare_scores.py` | Scoring logic for `compare` command |
| `core/quality.py` | Per-year quality metrics (missing %, duplicates, NaN counts) |

### Data Source

IEM (Iowa Environmental Mesonet) ASOS API at `mesonet.agron.iastate.edu`. Six fields available: `tmpf`, `dwpf`, `relh`, `sknt`, `drct`, `gust`. When `tmpf` and `relh` are present, wet-bulb is auto-computed via Stull (2011) approximation.

### Test Conventions

Tests live in `tests/` and use synthetic in-memory DataFrames (e.g., `pd.date_range` + sinusoidal temps). No network calls or file fixtures needed. Tests mirror the module they cover (e.g., `test_freeze.py` tests `core/freeze.py`).

### Remaining Roadmap

Phase 4: **UI** — Streamlit dashboard (Pass 1 complete). Entry point: `weather_tool/ui/app.py`. Pages: Compare, Station Detail, Wind, Exports. Service layer: `weather_tool/ui/backend.py`. Smoke validator: `python -m weather_tool.ui.check`. Install: `pip install -e ".[ui]"`.
