# Weather Analysis Tool

Deterministic historical weather time-series analysis. Computes yearly summaries (Tmax, Tmin, wet-bulb percentiles, hours above a reference temperature), data quality flags, and generates an insights report — all without relying on an LLM for metric computation.

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

### IEM Mode — dry-bulb only (default)

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

### IEM Mode — multi-field with wet-bulb

```bash
weather-tool run \
  --mode iem \
  --station-id KORD \
  --fields tmpf,dwpf,relh \
  --start 2018-01-01 \
  --end 2023-12-31 \
  --ref-temp 65.0 \
  --outdir outputs/
```

All six HVAC-relevant fields:

```bash
weather-tool run \
  --mode iem \
  --station-id KORD \
  --fields tmpf,dwpf,relh,sknt,drct,gust \
  --start 2018-01-01 \
  --end 2023-12-31 \
  --ref-temp 65.0 \
  --outdir outputs/
```

### Optional LLM Narrative

Set `OPENAI_API_KEY` and add the `--llm` flag:

```bash
export OPENAI_API_KEY="sk-..."
weather-tool run --mode csv --input data.csv --start 2020-01-01 --end 2023-12-31 --ref-temp 65 --llm
```

## `--fields` — IEM Multi-Field Ingestion

The `--fields` flag is a comma-separated list of IEM ASOS field codes to request.
Default: `tmpf` (backward-compatible, dry-bulb only).

| Field | Description | Unit |
|-------|-------------|------|
| `tmpf` | Dry-bulb temperature | °F |
| `dwpf` | Dew-point temperature | °F |
| `relh` | Relative humidity | % |
| `sknt` | Wind speed | knots |
| `drct` | Wind direction | degrees |
| `gust` | Wind gust | knots |

When `tmpf` is present in the output, it is aliased to `temp` for backward compatibility with all downstream code.

In CSV mode, `--fields` has no effect; use `--temp-col` to specify the temperature column name.

## Wet-Bulb Temperature

When `tmpf` (or `temp`) **and** `relh` are available, `wetbulb_f` is computed automatically using the **Stull (2011) approximation**:

```
Twb_C = T_C · atan(0.151977·(RH+8.313659)^0.5) + atan(T_C+RH)
        − atan(RH−1.676331) + 0.00391838·RH^1.5·atan(0.023101·RH)
        − 4.686035

wetbulb_f = Twb_C × 9/5 + 32
```

Inputs: T in °C (converted from tmpf), RH in %.
Accuracy: ±0.35 °C for T ∈ [−20, 50] °C, RH ∈ [5, 99] % (Stull 2011).
**Intent:** engineering screening and cooling-tower stress analysis, not psychrometric exactness.

**Fallback:** if `relh` is missing but `dwpf` is present, RH is derived via the Magnus approximation before applying Stull.

**Yearly summary columns added when `wetbulb_f` is available:**

| Column | Description |
|--------|-------------|
| `wb_p99` | 99th percentile wet-bulb (°F) |
| `wb_p996` | 99.6th percentile wet-bulb (°F) |
| `wb_max` | Maximum wet-bulb (°F) |
| `wb_mean` | Mean wet-bulb (°F) |
| `hours_wb_above_ref` | Hours where wet-bulb > ref_temp |

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

### Per-Field Quality Metrics

For each field present in the data (tmpf, dwpf, relh, sknt, drct, gust, wetbulb_f), the quality report and yearly summary include:

```
nan_count_<field>         – number of NaN rows for this field in the year-slice
field_missing_pct_<field> – nan_count / n_unique_timestamps (distinct from timestamp missing%)
wetbulb_availability_pct  – % of temp-valid rows where wet-bulb is computable
```

`field_missing_pct` measures data sparsity per variable; `missing_pct` measures timestamp gaps.

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
| `quality_report_<tag>.json` | Data quality flags, per-field NaN counts, wetbulb availability |
| `metadata_<tag>.json` | Run metadata: units, fields, dt inference, station info |
| `insights_<tag>.md` | Markdown report with rankings, trends, Moisture/Tower Stress section |

## Units

IEM fields are always returned in their native units (°F, %, knots, degrees) regardless of `--units`.
For CSV mode, `--units` records the unit in metadata; no conversion is applied.

| Mode | tmpf/dwpf | relh | sknt/gust | drct | wetbulb_f |
|------|-----------|------|-----------|------|-----------|
| IEM  | °F        | %    | knots     | deg  | °F        |
| CSV  | as-is     | —    | —         | —    | °F (if computed) |

## Running Tests

```bash
pytest -q
```

## Project Structure

```
weather_tool/
  pyproject.toml
  weather_tool/
    __init__.py
    cli.py              # Typer CLI (--fields flag added)
    config.py           # RunConfig dataclass (fields, IEM_UNITS)
    connectors/
      csv_connector.py  # CSV file loading
      iem_connector.py  # IEM/Iowa State multi-field download
    core/
      normalize.py      # Timestamp normalization, dedup
      quality.py        # Data quality checks + per-field NaN counts
      metrics.py        # Interval inference, hours_above_ref, compute_wetbulb_f
      aggregate.py      # Yearly summary builder (wb percentiles)
    insights/
      deterministic.py  # Rankings, trends, Moisture/Tower Stress section
      llm.py            # Optional LLM narrative
    storage/
      io.py             # File I/O (CSV, Parquet, JSON, MD)
  tests/
    test_interval.py
    test_hours_above.py
    test_missing_pct.py
    test_yearly_summary.py
    test_wetbulb.py
    test_iem_url.py
```
