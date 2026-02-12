# Weather Analysis Tool

Deterministic historical weather time-series analysis. Computes yearly summaries (Tmax, Tmin, hours above a reference temperature), data quality flags, and generates an insights report — all without relying on an LLM for metric computation.

## Install

```bash
# From the project root
pip install -e ".[dev]"
```

## Quick Start

### CSV Mode

```bash
weather-tool run \
  --mode csv \
  --input data/my_weather.csv \
  --timestamp-col "datetime" \
  --temp-col "temperature" \
  --start 2020-01-01 \
  --end 2023-12-31 \
  --ref-temp 65.0 \
  --units F \
  --tz "America/Chicago" \
  --outdir outputs/
```

### IEM Mode (Iowa Environmental Mesonet)

```bash
weather-tool run \
  --mode iem \
  --station-id KORD \
  --start 2018-01-01 \
  --end 2023-12-31 \
  --ref-temp 65.0 \
  --units F \
  --outdir outputs/
```

### Optional LLM Narrative

Set `OPENAI_API_KEY` and add the `--llm` flag:

```bash
export OPENAI_API_KEY="sk-..."
weather-tool run --mode csv --input data.csv --start 2020-01-01 --end 2023-12-31 --ref-temp 65 --llm
```

## How It Works

### Sampling Interval Inference (`dt_minutes`)

1. Parse and sort all unique timestamps.
2. Compute consecutive differences in minutes.
3. `dt_minutes = median(diffs)` — median is robust to gaps/outliers.
4. If >20% of diffs deviate from the median by >10%, `interval_change_flag` is set.

### Hours Above Reference Temperature

Uses a **step-function** approach:

```
count_above = number of records where (temp > ref_temp) AND temp is not NaN
hours_above_ref = count_above * (dt_minutes / 60.0)
```

NaN temperatures are excluded entirely — they do not count as above or below.

### Missing Data Percentage (Windowed)

Missing% is computed **within the analysis window**, not the full calendar year:

```
For each year-slice [slice_start, slice_end]:
  slice_start = max(window_start, Jan 1 of year)
  slice_end   = min(window_end, Dec 31 23:59:59 of year)
  total_minutes = (slice_end - slice_start) in minutes
  expected_records = floor(total_minutes / dt_minutes) + 1
  missing_pct = clamp(1 - n_unique_timestamps / expected_records, 0, 1)
```

This means a perfect July–December dataset has `missing_pct ≈ 0`, not 50%.

### Coverage

```
coverage_pct = slice_duration_days / year_duration_days
partial_coverage_flag = True if coverage_pct < 0.98
```

## Outputs

All files are written to `--outdir` (default `outputs/`):

| File | Description |
|------|-------------|
| `summary_<tag>.csv` | Per-year summary table with all metrics |
| `raw_clean_<tag>.parquet` | Cleaned, window-filtered time-series |
| `quality_report_<tag>.json` | Data quality flags and diagnostics |
| `metadata_<tag>.json` | Run metadata: units, dt inference, station info |
| `insights_<tag>.md` | Markdown insights report with rankings and trends |

## Units

The `--units` flag controls how temperature values are interpreted:

- **agnostic** (default): values used as-is, no conversion.
- **F / C / K**: records the unit in metadata; no conversion applied.
- **auto**: conservative heuristic detection (always overridable).

All outputs include a `units` field in metadata.

## Running Tests

```bash
pytest -v
```

## Project Structure

```
weather_tool/
  pyproject.toml
  weather_tool/
    __init__.py
    cli.py              # Typer CLI
    config.py           # RunConfig dataclass
    connectors/
      csv_connector.py  # CSV file loading
      iem_connector.py  # IEM/Iowa State download
    core/
      normalize.py      # Timestamp normalization, dedup
      quality.py        # Data quality checks
      metrics.py        # Interval inference, hours_above_ref
      aggregate.py      # Yearly summary builder
    insights/
      deterministic.py  # Rankings, trends, markdown report
      llm.py            # Optional LLM narrative
    storage/
      io.py             # File I/O (CSV, Parquet, JSON, MD)
  tests/
    test_interval.py
    test_hours_above.py
    test_missing_pct.py
    test_yearly_summary.py
```
