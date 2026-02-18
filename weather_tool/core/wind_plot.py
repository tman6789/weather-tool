"""Wind rose polar chart generation (requires matplotlib)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def plot_wind_rose(
    rose_hours: pd.DataFrame,
    rose_meta: dict[str, Any],
    speed_edges: list[float],
    speed_units: str,
    title: str,
    filepath: str | Path,
) -> None:
    """Render a polar stacked-bar wind rose and save to *filepath*.

    Parameters
    ----------
    rose_hours : sector × speed-bin hours matrix from :func:`wind_rose_table`.
    rose_meta : metadata dict from :func:`wind_rose_table`.
    speed_edges : speed bin edges (same as passed to wind_rose_table).
    speed_units : ``"mph"`` or ``"kt"``.
    title : chart title string.
    filepath : output image path (PNG recommended).

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for wind rose plots. "
            "Install it with: pip install 'weather-tool[viz]'"
        )

    # Separate sectors from Calm row
    sector_only = rose_hours.drop(index="Calm", errors="ignore")
    n_sectors = len(sector_only)
    total_valid = rose_meta["total_valid_hours"]

    # Convert hours to percentage of total valid hours
    if total_valid > 0:
        pct_matrix = sector_only / total_valid * 100.0
    else:
        pct_matrix = sector_only * 0.0

    # Angles: evenly spaced, starting at North (top), clockwise
    angles = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)
    bar_width = 2 * np.pi / n_sectors * 0.85

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "polar"}, figsize=(8, 8))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise (meteorological convention)

    # Stacked bars per speed bin (use default matplotlib color cycle)
    bottoms = np.zeros(n_sectors)
    speed_bin_labels = list(pct_matrix.columns)

    for i, col in enumerate(speed_bin_labels):
        values = pct_matrix[col].values
        ax.bar(
            angles,
            values,
            width=bar_width,
            bottom=bottoms,
            label=f"{col} {speed_units}",
        )
        bottoms += values

    # Y-axis label
    ax.set_ylabel("% of total hours", labelpad=30)

    # Sector labels
    ax.set_xticks(angles)
    ax.set_xticklabels(sector_only.index.tolist())

    # Legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    # Title
    ax.set_title(title, pad=20, fontsize=11)

    # Annotations below chart
    calm_pct = rose_meta.get("calm_pct", 0.0)
    annotation = f"Calm: {calm_pct:.1f}%"
    unknown_pct = rose_meta.get("unknown_dir_pct", 0.0)
    if unknown_pct > 1.0:
        annotation += f"  |  Unknown dir: {unknown_pct:.1f}%"
    fig.text(0.5, 0.02, annotation, ha="center", fontsize=9)

    # Save
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
