# Weather Analysis Tool

Deterministic historical weather time-series analysis. Computes yearly summaries (Tmax, Tmin, wet-bulb percentiles, hours above a reference temperature, freeze risk), data quality flags, and generates an insights report — all without relying on an LLM for metric computation.

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

## Economizer / Tower Decision Metrics

All metrics are **screening-level proxies** computed from dry-bulb and wet-bulb temperatures. They are not full psychrometric simulations; results should be validated against site-specific equipment specs before use in design decisions.

### Airside Economizer Hours

```
air_econ_hours = count(Tdb ≤ air_econ_threshold_f  AND  not NaN) × (dt_minutes / 60)
```

Default threshold: `--air-econ-threshold-f 55` (°F). Step-function; uses deduplicated observations.

### Waterside Economizer (WEC) Proxy

Physical temperature chain (screening-level):

```
Tower delivers:   CWS  ≈ Twb + tower_approach_f
Plate HX delivers: CHWS ≈ CWS  + hx_approach_f

WEC feasible when: Twb + tower_approach_f + hx_approach_f ≤ chw_supply_f

Therefore:
  required_twb_max    = chw_supply_f - tower_approach_f - hx_approach_f
  wec_hours           = count(Twb ≤ required_twb_max AND not NaN) × dt_h
  hours_with_wetbulb  = count(Twb not NaN) × dt_h
  wec_feasible_pct    = wec_hours / hours_with_wetbulb
```

**Default parameters:**

| Parameter | Flag | Default | Meaning |
|-----------|------|---------|---------|
| CHW supply | `--chw-supply-f` | 44 °F | Target chilled-water supply temperature |
| Tower approach | `--tower-approach-f` | 7 °F | Cooling tower approach (LWT − Twb) |
| HX approach | `--hx-approach-f` | 5 °F | Plate heat-exchanger approach delta |

With defaults: `required_twb_max = 44 − 7 − 5 = 32 °F`.

**Degradation:** if wet-bulb data is unavailable (CSV without dewpoint/RH), `wec_hours` and `wec_feasible_pct` are `NaN`; `hours_with_wetbulb` is `0`.

**Compare-mode window metrics:**

| Column | Definition |
|--------|-----------|
| `wec_hours_sum` | Sum of `wec_hours` across years (NaN-safe) |
| `hours_with_wetbulb_sum` | Sum of `hours_with_wetbulb` across years |
| `wec_feasible_pct_over_window` | `wec_hours_sum / hours_with_wetbulb_sum` — comparable across stations with differing wetbulb availability |

### Tower Stress Hours

```
tower_stress_hours_wb_gt_<T> = count(Twb ≥ T  AND  not NaN) × dt_h
```

Computed for each threshold in `--wetbulb-stress-thresholds-f` (default: `75,78,80` °F). Uses `≥` (inclusive).

### Rolling Wet-Bulb Maxima

```
1. Sort and index wetbulb series by timestamp (UTC).
2. Resample to regular round(dt_minutes)-minute grid (mean within bins; NaN for gaps).
3. Integer-step rolling window: n_steps = round(window_hours × 60 / dt_round)
4. Completeness guard: min_periods = max(1, int(0.80 × n_steps))
   — windows with < 80% of expected samples produce NaN.
5. wb_mean_<W>h_max = max of valid rolling means; NaN if none.
```

Resampling to a regular grid before rolling ensures stable `min_periods` counts regardless of timestamp irregularity.

### LWT Proxy

```
lwt_proxy_f     = Twb + tower_approach_f   (per observation)
lwt_proxy_p99   = 99th percentile of lwt_proxy_f
lwt_proxy_max   = max of lwt_proxy_f
```

Actual condenser-water leaving temperature depends on tower specifications, flow rates, and heat load.

### Econ Confidence Flag

`econ_confidence_flag = True` if **any** of the following:
- `timestamp_missing_pct_avg > 2%`
- `wetbulb_availability_pct_avg < 70%`

## Freeze Risk Metrics

All freeze metrics use the same **step-function** convention as other hourly metrics. They are computed from dry-bulb temperature only — no wet-bulb data required.

### Core Freeze Math

**Freeze condition (inclusive):**
```
is_freeze = Tdb ≤ freeze_threshold_f  AND  Tdb not NaN
```

**Hour metrics (per year-slice):**
```
freeze_hours              = count(is_freeze) × (dt_minutes / 60)
total_hours_with_temp     = count(Tdb not NaN) × (dt_minutes / 60)
freeze_hours_pct          = freeze_hours / total_hours_with_temp
                            (NaN if total_hours_with_temp == 0; missing data ≠ 0% freeze)

freeze_hours_shoulder     = freeze_hours restricted to rows where month ∈ shoulder_months
```

**Event detection:**
```
gap_break_minutes = freeze_gap_tolerance_mult × dt_minutes

A run is broken when ANY of:
  - consecutive timestamp gap > gap_break_minutes, OR
  - the observation is NaN (conservative: missing data does not extend freeze events), OR
  - the observation is not freeze-condition

Event: a freeze run with duration ≥ freeze_min_event_hours
duration = run_count × (dt_minutes / 60)   (step function)

freeze_event_count               — integer count of qualifying events per year
freeze_event_max_duration_hours  — max event duration; NaN if count == 0
```

### Freeze CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--freeze-threshold-f` | `32.0` | Dry-bulb freeze threshold °F (Tdb ≤ this counts as freeze) |
| `--freeze-min-event-hours` | `3.0` | Minimum continuous hours below threshold for a freeze event |
| `--freeze-gap-tolerance-mult` | `1.5` | Gap > (mult × dt_minutes) breaks event continuity |
| `--freeze-shoulder-months` | `3,4,10,11` | Comma-separated month numbers for shoulder-season freeze exposure |

Both the `run` and `compare` commands accept all four flags.

**Example — `run` with custom freeze params:**
```bash
weather-tool run \
  --mode iem --station-id KORD \
  --start 2018-01-01 --end 2023-12-31 \
  --ref-temp 65 --fields tmpf,dwpf,relh \
  --freeze-threshold-f 32 \
  --freeze-min-event-hours 3 \
  --freeze-shoulder-months "3,4,10,11"
```

**Example — `compare` with freeze:**
```bash
weather-tool compare KORD KPHX KIAD \
  --start 2018-01-01 --end 2023-12-31 \
  --ref-temps 80,85 --fields tmpf,dwpf,relh \
  --freeze-threshold-f 32
```

### Yearly Summary Freeze Columns

| Column | Description |
|--------|-------------|
| `freeze_threshold_f` | Configured threshold (°F) |
| `freeze_min_event_hours` | Configured minimum event duration |
| `freeze_hours` | Hours with Tdb ≤ threshold |
| `total_hours_with_temp` | Hours with non-NaN Tdb (denominator) |
| `freeze_hours_pct` | `freeze_hours / total_hours_with_temp` (NaN if no temp data) |
| `freeze_hours_shoulder` | Freeze hours in shoulder months only |
| `freeze_event_count` | Number of qualifying freeze events |
| `freeze_event_max_duration_hours` | Longest freeze event (h); NaN if none |

### Compare-Mode Window Freeze Columns

| Column | Aggregation | Description |
|--------|-------------|-------------|
| `freeze_hours_sum` | sum | Total freeze hours over the analysis window |
| `freeze_hours_shoulder_sum` | sum | Shoulder-season freeze hours over the window |
| `total_hours_with_temp_sum` | sum | Total hours with valid Tdb (denominator for pct) |
| `freeze_hours_pct_over_window` | `sum/sum` | Window-level freeze fraction (NaN if denominator = 0) |
| `freeze_event_count_sum` | sum | Total qualifying freeze events across all years |
| `freeze_event_max_duration_hours_max` | max | Longest single freeze event across all years |
| `freeze_confidence_flag` | flag | `True` if temp availability < 90% of actual window hours, or missing% > 2% |

**`freeze_confidence_flag` detail:**

The denominator uses actual window hours derived from the date range in the yearly summary (`window_start.min()` → `window_end.max() + 1 day`), rather than a calendar approximation. A station with `total_hours_with_temp_sum / window_hours < 0.90` gets a low-confidence flag.

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
      aggregate.py      # Yearly summary builder (wb percentiles, freeze)
      econ_tower.py     # Economizer / tower stress metrics
      freeze.py         # Freeze risk metrics (pure functions)
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
    test_econ_tower.py
    test_freeze.py
```
