"""Stage 3: integrate a PSD over user-defined frequency bands to produce the
per-animal band-power summary table."""
from __future__ import annotations

import keyword

import numpy as np
import pandas as pd


def validate_band_names(bands: dict[str, tuple[float, float]]) -> None:
    """Band names become column headers in the summary table, so they must
    be valid, non-keyword Python identifiers, and each band's lower edge
    must be strictly less than its upper edge."""
    for name, (low, high) in bands.items():
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"band name {name!r} is not a valid identifier")
        if not (low < high):
            raise ValueError(f"band {name!r} has lower edge {low} >= upper edge {high}")


def validate_band_edges(
    bands: dict[str, tuple[float, float]], window_length: int, fs: float
) -> list[str]:
    """Warn for any band whose lower edge falls below 1 / window_duration,
    since power there is interpolated rather than independently resolved."""
    window_duration_sec = window_length / fs
    min_resolvable_hz = 1.0 / window_duration_sec
    warnings = []
    for name, (low, _high) in bands.items():
        if low < min_resolvable_hz:
            warnings.append(
                f"Band {name!r} lower edge {low} Hz is below the resolvable "
                f"minimum {min_resolvable_hz:.5f} Hz for a {window_length}-sample "
                f"window at {fs} Hz; power there is interpolated."
            )
    return warnings


def integrate_bands(
    psd_table: pd.DataFrame, bands: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    """Integrate power over frequency for each band, per (animal, sex,
    region, condition) group, using the trapezoidal rule. One row per group,
    one column per band name.
    """
    validate_band_names(bands)
    group_cols = ["animal", "sex", "region", "condition"]
    out_rows = []
    for keys, group in psd_table.groupby(group_cols, sort=False):
        animal, sex, region, condition = keys
        group_sorted = group.sort_values("frequency")
        freqs = group_sorted["frequency"].to_numpy()
        power = group_sorted["power"].to_numpy()
        row = {"animal": animal, "sex": sex, "region": region, "condition": condition}
        for name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            if mask.sum() < 2:
                row[name] = float("nan")
            else:
                row[name] = float(np.trapezoid(power[mask], freqs[mask]))
        out_rows.append(row)
    columns = group_cols + list(bands.keys())
    return pd.DataFrame(out_rows, columns=columns)
