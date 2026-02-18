"""Wind analytics: direction binning, wind rose tables, co-occurrence analysis.

All functions are deterministic and pure (no I/O, no side effects).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

CALM_THRESHOLD_MPH = 0.5
CALM_THRESHOLD_KT = CALM_THRESHOLD_MPH / 1.15078  # ~0.434 kt
KT_TO_MPH = 1.15078

_LABELS_16 = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]
_LABELS_8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

SLICE_MONTHS: dict[str, list[int] | None] = {
    "annual": None,
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "fall": [9, 10, 11],
}


# ── Wind normalisation ────────────────────────────────────────────────────────

def normalize_wind(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived wind columns to *df* (returns a copy).

    Added columns:
      drct_deg       – direction in degrees [0, 360); 360 mapped to 0; out-of-range → NaN
      wind_speed_kt  – speed in knots (from ``sknt`` column)
      wind_speed_mph – speed in mph
      is_calm        – True when speed_mph < CALM_THRESHOLD_MPH **and** speed is not NaN
    """
    out = df.copy()

    # Direction
    if "drct" in out.columns:
        drct = pd.to_numeric(out["drct"], errors="coerce")
        # 360 → 0
        drct = drct.where(drct != 360, 0.0)
        # Out-of-range → NaN
        drct = drct.where((drct >= 0) & (drct < 360), other=np.nan)
        out["drct_deg"] = drct
    else:
        out["drct_deg"] = np.nan

    # Speed
    if "sknt" in out.columns:
        speed_kt = pd.to_numeric(out["sknt"], errors="coerce")
        out["wind_speed_kt"] = speed_kt
        out["wind_speed_mph"] = speed_kt * KT_TO_MPH
    else:
        out["wind_speed_kt"] = np.nan
        out["wind_speed_mph"] = np.nan

    # Calm: explicitly low speed only; NaN speed → NOT calm (unknown)
    out["is_calm"] = (out["wind_speed_mph"] < CALM_THRESHOLD_MPH) & out["wind_speed_mph"].notna()

    return out


# ── Direction binning ──────────────────────────────────────────────────────────

def direction_bin_index(
    drct_deg: float | np.ndarray,
    n_bins: int = 16,
) -> float | np.ndarray:
    """Map direction(s) in degrees to centered bin indices.

    Bin 0 is centred on 0° (North).  For 16 bins the edges are ±11.25°.
    NaN input → NaN output.  Works on scalars and numpy arrays.
    """
    sector_width = 360.0 / n_bins
    arr = np.asarray(drct_deg, dtype=float)
    result = np.floor(((arr % 360.0) + sector_width / 2.0) / sector_width) % n_bins
    # Preserve NaN
    result = np.where(np.isnan(arr), np.nan, result)
    # Return scalar if input was scalar
    if np.ndim(drct_deg) == 0:
        return float(result)
    return result


def sector_labels(n_bins: int = 16) -> list[str]:
    """Return human-readable sector labels for *n_bins* direction bins."""
    if n_bins == 16:
        return list(_LABELS_16)
    if n_bins == 8:
        return list(_LABELS_8)
    return [str(i) for i in range(n_bins)]


# ── Wind rose table ───────────────────────────────────────────────────────────

def _speed_bin_labels(edges: list[float]) -> list[str]:
    """Build range labels like '0-5', '5-10', '30+' from edge list."""
    labels: list[str] = []
    for i in range(len(edges) - 1):
        labels.append(f"{edges[i]:g}-{edges[i+1]:g}")
    labels.append(f"{edges[-1]:g}+")
    return labels


def wind_rose_table(
    df: pd.DataFrame,
    dt_minutes: float,
    dir_bins: int = 16,
    speed_edges: list[float] | None = None,
    speed_units: str = "mph",
    slice_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a wind-rose hours matrix and metadata dict.

    Parameters
    ----------
    df : DataFrame with ``drct_deg``, ``wind_speed_mph`` (or ``wind_speed_kt``),
         and ``is_calm`` columns (output of :func:`normalize_wind`).
    dt_minutes : sampling interval in minutes.
    dir_bins : number of directional sectors (default 16).
    speed_edges : speed bin edges in *speed_units* (default ``[0,5,10,15,20,30]``).
    speed_units : ``"mph"`` or ``"kt"`` — which speed column to bin on.
    slice_mask : optional boolean Series to subset rows (e.g. seasonal filter).

    Returns
    -------
    (rose_hours, rose_meta)
      rose_hours : DataFrame — rows = sector labels + ``"Calm"``, columns = speed-range labels.
                   Values are hours.
      rose_meta  : dict with ``total_valid_hours``, ``calm_hours``, ``calm_pct``,
                   ``unknown_dir_hours``, ``unknown_dir_pct``, ``obs_count``.
    """
    if speed_edges is None:
        speed_edges = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0]

    speed_col = "wind_speed_mph" if speed_units == "mph" else "wind_speed_kt"
    dt_h = dt_minutes / 60.0

    # Apply slice mask
    work = df if slice_mask is None else df[slice_mask]
    work = work.reset_index(drop=True)

    # Only rows with valid (non-NaN) speed
    has_speed = work[speed_col].notna()
    valid = work[has_speed].reset_index(drop=True)
    obs_count = len(valid)
    total_valid_hours = round(obs_count * dt_h, 2)

    # Calm observations
    calm_mask = valid["is_calm"]
    calm_count = int(calm_mask.sum())
    calm_hours = round(calm_count * dt_h, 2)

    # Non-calm observations with valid speed
    non_calm = valid[~calm_mask].reset_index(drop=True)

    # Unknown direction: non-calm, valid speed, but NaN direction
    unknown_dir_mask = non_calm["drct_deg"].isna()
    unknown_dir_count = int(unknown_dir_mask.sum())
    unknown_dir_hours = round(unknown_dir_count * dt_h, 2)

    # Binnable: non-calm, valid speed, valid direction
    binnable = non_calm[~unknown_dir_mask].reset_index(drop=True)

    # Sector indices
    labels = sector_labels(dir_bins)
    bin_labels = _speed_bin_labels(speed_edges)

    # Initialise matrix (sectors × speed bins) — all zeros
    matrix = pd.DataFrame(0.0, index=labels + ["Calm"], columns=bin_labels)

    if len(binnable) > 0:
        sec_idx = direction_bin_index(binnable["drct_deg"].values, dir_bins).astype(int)
        speeds = binnable[speed_col].values

        # Assign speed bins: digitize gives bin index (1-based into edges+inf)
        full_edges = speed_edges + [np.inf]
        spd_bin_idx = np.digitize(speeds, full_edges) - 1  # 0-based
        spd_bin_idx = np.clip(spd_bin_idx, 0, len(bin_labels) - 1)

        for i in range(len(binnable)):
            sector_label = labels[sec_idx[i]]
            spd_label = bin_labels[spd_bin_idx[i]]
            matrix.loc[sector_label, spd_label] += dt_h

    # Calm row: all hours go in first speed bin column (convention)
    matrix.loc["Calm", bin_labels[0]] = calm_hours

    # Round
    matrix = matrix.round(2)

    calm_pct = round(calm_hours / total_valid_hours * 100, 2) if total_valid_hours > 0 else 0.0
    unknown_dir_pct = round(unknown_dir_hours / total_valid_hours * 100, 2) if total_valid_hours > 0 else 0.0

    rose_meta: dict[str, Any] = {
        "total_valid_hours": total_valid_hours,
        "calm_hours": calm_hours,
        "calm_pct": calm_pct,
        "unknown_dir_hours": unknown_dir_hours,
        "unknown_dir_pct": unknown_dir_pct,
        "obs_count": obs_count,
    }

    return matrix, rose_meta


# ── Prevailing sector ─────────────────────────────────────────────────────────

def prevailing_sector(rose_hours: pd.DataFrame) -> str:
    """Return the sector label (excluding Calm) with the highest total hours."""
    sector_only = rose_hours.drop(index="Calm", errors="ignore")
    row_sums = sector_only.sum(axis=1)
    if row_sums.sum() == 0:
        return "N/A"
    return str(row_sums.idxmax())


# ── Threshold resolution ──────────────────────────────────────────────────────

def resolve_threshold(spec: str, metric_series: pd.Series) -> float:
    """Resolve a threshold specification to a numeric value.

    Parameters
    ----------
    spec : ``"p99"``, ``"p95"``, ``"p996"``, or a numeric string like ``"80"``.
    metric_series : the metric data (caller must pass the correct slice).

    Returns
    -------
    float threshold value.  NaN if metric_series has no valid data.
    """
    pct_map = {"p99": 99, "p95": 95, "p996": 99.6, "p90": 90}
    if spec.lower() in pct_map:
        valid = metric_series.dropna()
        if len(valid) == 0:
            return float("nan")
        return float(np.nanpercentile(valid.values, pct_map[spec.lower()]))
    # Numeric literal
    try:
        return float(spec)
    except ValueError:
        return float("nan")


# ── Exceedance run detection ──────────────────────────────────────────────────

def detect_exceedance_runs(
    timestamps: pd.Series,
    metric: pd.Series,
    threshold: float,
    dt_minutes: float,
    min_hours: float,
    gap_mult: float,
) -> tuple[list[dict[str, Any]], pd.Series]:
    """Detect contiguous exceedance runs (metric >= threshold).

    Mirrors :func:`weather_tool.core.freeze.detect_freeze_events` logic but
    with ``>=`` (exceedance) instead of ``<=`` (freeze).  NaN metric values
    break run continuity (conservative).

    Parameters
    ----------
    timestamps : tz-aware timestamp series aligned to *metric*.
    metric : numeric series (e.g. temp, wetbulb_f).
    threshold : exceedance threshold (inclusive >=).
    dt_minutes : sampling interval in minutes.
    min_hours : minimum run duration to qualify as event.
    gap_mult : gap > ``gap_mult * dt_minutes`` breaks continuity.

    Returns
    -------
    (events, event_mask)
      events : list of dicts with ``start_ts``, ``end_ts``, ``duration_hours``.
      event_mask : boolean Series (same index as input) — True for rows in qualifying events.
    """
    event_mask = pd.Series(False, index=metric.index)

    if math.isnan(dt_minutes) or dt_minutes <= 0 or math.isnan(threshold):
        return [], event_mask

    gap_break_minutes = gap_mult * dt_minutes
    dt_h = dt_minutes / 60.0

    ts_arr = timestamps.reset_index(drop=True)
    m_arr = metric.reset_index(drop=True)
    orig_idx = metric.index

    n = len(m_arr)
    if n == 0:
        return [], event_mask

    events: list[dict[str, Any]] = []
    qualifying_indices: list[int] = []
    run_indices: list[int] = []
    prev_ts = None

    def _flush_run() -> None:
        if not run_indices:
            return
        duration = len(run_indices) * dt_h
        if duration >= min_hours:
            events.append({
                "start_ts": ts_arr.iloc[run_indices[0]],
                "end_ts": ts_arr.iloc[run_indices[-1]],
                "duration_hours": round(duration, 2),
            })
            qualifying_indices.extend(run_indices)

    for i in range(n):
        raw = m_arr.iloc[i]
        is_nan = pd.isna(raw)
        is_exceed = (not is_nan) and (float(raw) >= threshold)

        gap_breaks = False
        if prev_ts is not None:
            gap_min = (ts_arr.iloc[i] - prev_ts).total_seconds() / 60.0
            gap_breaks = gap_min > gap_break_minutes

        if is_exceed and not gap_breaks:
            run_indices.append(i)
        else:
            _flush_run()
            run_indices = [i] if is_exceed else []

        prev_ts = ts_arr.iloc[i]

    _flush_run()

    # Build mask from collected qualifying indices (O(n), not O(n*events))
    mask_values = np.zeros(n, dtype=bool)
    for idx in qualifying_indices:
        mask_values[idx] = True
    event_mask = pd.Series(mask_values, index=orig_idx)

    return events, event_mask


# ── Co-occurrence: event wind stats ───────────────────────────────────────────

def compute_event_wind_stats(
    df: pd.DataFrame,
    metric_col: str,
    threshold_value: float,
    dt_minutes: float,
    dir_bins: int = 16,
    min_event_hours: float = 0.0,
    gap_tolerance_mult: float = 1.5,
) -> dict[str, Any]:
    """Wind statistics during metric-threshold exceedance events.

    Parameters
    ----------
    df : DataFrame with wind columns (from :func:`normalize_wind`) and *metric_col*.
    metric_col : column name for the exceedance metric (e.g. ``"temp"``, ``"wetbulb_f"``).
    threshold_value : exceedance threshold (>=).
    dt_minutes : sampling interval in minutes.
    dir_bins : number of directional bins.
    min_event_hours : minimum event duration (0 = all exceedance rows).
    gap_tolerance_mult : gap tolerance for event detection.

    Returns
    -------
    dict with event_count, event_hours_total, mean_speed_kt, p50_speed_kt,
    p90_speed_kt, calm_pct, unknown_dir_pct, top3_sectors, sector_hours, sector_pct.
    """
    empty_result: dict[str, Any] = {
        "event_count": 0,
        "event_hours_total": 0.0,
        "mean_speed_kt": float("nan"),
        "p50_speed_kt": float("nan"),
        "p90_speed_kt": float("nan"),
        "calm_pct": 0.0,
        "unknown_dir_pct": 0.0,
        "top3_sectors": [],
        "sector_hours": {},
        "sector_pct": {},
    }

    if metric_col not in df.columns:
        return empty_result
    if math.isnan(threshold_value):
        return empty_result

    dt_h = dt_minutes / 60.0

    events, event_mask = detect_exceedance_runs(
        df["timestamp"], df[metric_col],
        threshold_value, dt_minutes, min_event_hours, gap_tolerance_mult,
    )
    event_df = df[event_mask].reset_index(drop=True)
    event_count = len(events)

    if len(event_df) == 0:
        return empty_result

    event_hours_total = round(len(event_df) * dt_h, 2)

    # Speed stats (knots)
    valid_kt = event_df["wind_speed_kt"].dropna()
    mean_kt = round(float(valid_kt.mean()), 2) if len(valid_kt) > 0 else float("nan")
    p50_kt = round(float(np.nanpercentile(valid_kt.values, 50)), 2) if len(valid_kt) > 0 else float("nan")
    p90_kt = round(float(np.nanpercentile(valid_kt.values, 90)), 2) if len(valid_kt) > 0 else float("nan")

    # Calm percentage (of event rows with valid speed)
    has_speed = event_df["wind_speed_mph"].notna()
    valid_speed_count = int(has_speed.sum())
    calm_count = int(event_df.loc[has_speed, "is_calm"].sum()) if valid_speed_count > 0 else 0
    calm_pct = round(calm_count / valid_speed_count * 100, 2) if valid_speed_count > 0 else 0.0

    # Unknown direction (non-calm, valid speed, NaN direction)
    non_calm_valid = event_df[has_speed & ~event_df["is_calm"]]
    unknown_dir_count = int(non_calm_valid["drct_deg"].isna().sum())
    unknown_dir_pct = round(unknown_dir_count / max(len(non_calm_valid), 1) * 100, 2)

    # Sector distribution (non-calm, valid speed, valid direction)
    binnable = non_calm_valid[non_calm_valid["drct_deg"].notna()].reset_index(drop=True)
    labels = sector_labels(dir_bins)
    sector_hrs: dict[str, float] = {lbl: 0.0 for lbl in labels}

    if len(binnable) > 0:
        sec_idx = direction_bin_index(binnable["drct_deg"].values, dir_bins).astype(int)
        for i in range(len(binnable)):
            sector_hrs[labels[sec_idx[i]]] += dt_h

    # Round and compute pct
    total_sector_hrs = sum(sector_hrs.values())
    sector_hrs = {k: round(v, 2) for k, v in sector_hrs.items()}
    sector_pct: dict[str, float] = {}
    for k, v in sector_hrs.items():
        sector_pct[k] = round(v / total_sector_hrs * 100, 2) if total_sector_hrs > 0 else 0.0

    # Top 3 sectors by hours
    sorted_sectors = sorted(sector_hrs.items(), key=lambda x: x[1], reverse=True)
    top3 = [s[0] for s in sorted_sectors[:3] if s[1] > 0]

    return {
        "event_count": event_count,
        "event_hours_total": event_hours_total,
        "mean_speed_kt": mean_kt,
        "p50_speed_kt": p50_kt,
        "p90_speed_kt": p90_kt,
        "calm_pct": calm_pct,
        "unknown_dir_pct": unknown_dir_pct,
        "top3_sectors": top3,
        "sector_hours": sector_hrs,
        "sector_pct": sector_pct,
    }


# ── Sector deltas ─────────────────────────────────────────────────────────────

def compute_sector_deltas(
    baseline_rose_hours: pd.DataFrame,
    event_sector_pcts: dict[str, float],
) -> dict[str, Any]:
    """Compare event sector distribution against baseline wind rose.

    Parameters
    ----------
    baseline_rose_hours : full wind rose hours matrix (from :func:`wind_rose_table`).
    event_sector_pcts : ``sector_pct`` dict from :func:`compute_event_wind_stats`.

    Returns
    -------
    dict with sector_deltas, overrepresented_sector, overrepresented_delta_pct.
    """
    # Baseline sector percentages (exclude Calm row)
    sector_only = baseline_rose_hours.drop(index="Calm", errors="ignore")
    row_sums = sector_only.sum(axis=1)
    total = row_sums.sum()

    baseline_pcts: dict[str, float] = {}
    for lbl in sector_only.index:
        baseline_pcts[lbl] = round(float(row_sums[lbl]) / total * 100, 2) if total > 0 else 0.0

    # Deltas: event_pct - baseline_pct
    deltas: dict[str, float] = {}
    for lbl in baseline_pcts:
        ev_pct = event_sector_pcts.get(lbl, 0.0)
        deltas[lbl] = round(ev_pct - baseline_pcts[lbl], 2)

    # Most overrepresented sector
    if deltas:
        max_sector = max(deltas, key=lambda k: deltas[k])
        max_delta = deltas[max_sector]
    else:
        max_sector = "N/A"
        max_delta = 0.0

    return {
        "sector_deltas": deltas,
        "overrepresented_sector": max_sector,
        "overrepresented_delta_pct": max_delta,
    }
