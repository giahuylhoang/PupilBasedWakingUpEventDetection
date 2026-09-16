"""Stage 3 outputs: group-average spectra, per-animal small multiples, and
method-comparison overlays. No Streamlit imports -- every function returns a
matplotlib Figure (or DataFrame, for the percent-difference table) that
app.py is responsible for displaying.

`_filtered_or_raise` is the one shared filter-or-fail code path: all three
plotting functions need to select rows for a region (and, for the method
comparison, a condition) below `max_freq`, and none of them should hand
matplotlib an empty frame and let it render a silently blank/confusing axes
-- an unknown region or condition is a caller error and is reported as a
ValueError naming what was requested, before any Figure is created (so
nothing is left open on the error path).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _filtered_or_raise(df: pd.DataFrame, mask: pd.Series, description: str) -> pd.DataFrame:
    """Return df[mask], or raise ValueError naming `description` if that
    selection is empty. Centralises the "no matching data" decision so all
    three plotting functions fail the same way instead of silently drawing
    an empty axes."""
    sub = df[mask]
    if sub.empty:
        raise ValueError(f"no data found for {description}")
    return sub


def plot_group_average(
    psd_table: pd.DataFrame,
    region: str,
    max_freq: float = 0.5,
    xscale: str = "log",
    yscale: str = "log",
) -> plt.Figure:
    """Mean +/- SEM across animals, one line per condition, for one region.

    SEM at each frequency divides by sqrt(number of animals actually
    present at that frequency), not by the condition's total animal count:
    animals can have ragged frequency coverage (e.g. differing trace
    lengths under a method whose frequency grid depends on trace length),
    in which case a fixed denominator would understate SEM at any
    frequency where an animal is missing.
    """
    sub = _filtered_or_raise(
        psd_table,
        (psd_table["region"] == region) & (psd_table["frequency"] <= max_freq),
        f"region={region!r} with frequency <= {max_freq}",
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    for condition, cond_df in sub.groupby("condition", sort=True):
        pivot = cond_df.pivot_table(index="frequency", columns="animal", values="power").sort_index()
        # n_animals: total distinct animals in this condition, for the "n="
        # legend label. n_present: animals actually present at each specific
        # frequency row, for the SEM denominator below. These differ under
        # ragged frequency coverage -- see the SEM note in this function's
        # docstring -- so n_animals must never be substituted into the SEM
        # calculation below.
        n_animals = pivot.shape[1]
        mean = pivot.mean(axis=1)
        n_present = pivot.count(axis=1)
        sem = pivot.std(axis=1, ddof=1) / np.sqrt(n_present)
        ax.plot(mean.index, mean.values, label=f"{condition} (n={n_animals})")
        ax.fill_between(mean.index, (mean - sem).values, (mean + sem).values, alpha=0.3)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(f"Group-average PSD -- {region}")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_per_animal_small_multiples(
    psd_table: pd.DataFrame,
    region: str,
    max_freq: float = 0.5,
    xscale: str = "log",
    yscale: str = "log",
) -> plt.Figure:
    """Small-multiples grid, one subplot per animal, all conditions
    overlaid. Grid slots beyond the animal count are set invisible (not
    merely `axis("off")`, which leaves `get_visible()` True and hides only
    the ticks/spines) so the visible-axes count is exactly the animal
    count.

    `xscale`/`yscale` default to "log" so this signature stays
    backward-compatible with every existing caller, but they exist because
    the app's frequency-axis control previously reached only
    `plot_group_average` -- leaving the per-animal panel locked to log while
    the group figure right above it switched to linear, which reads as a
    rendering bug rather than as a deliberate scope limit. A linear
    frequency axis is the right choice for showing a sharp peak in a narrow
    0-0.1 Hz window; log is right for inspecting 1/f slope.
    """
    sub = _filtered_or_raise(
        psd_table,
        (psd_table["region"] == region) & (psd_table["frequency"] <= max_freq),
        f"region={region!r} with frequency <= {max_freq}",
    )
    animals = sorted(sub["animal"].unique())
    n = len(animals)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for i, animal in enumerate(animals):
        ax = axes[i // ncols][i % ncols]
        animal_df = sub[sub["animal"] == animal]
        for condition, cond_df in animal_df.groupby("condition", sort=True):
            cond_df = cond_df.sort_values("frequency")
            ax.plot(cond_df["frequency"], cond_df["power"], label=str(condition))
        ax.set_title(str(animal), fontsize=10)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.legend(frameon=False, fontsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(f"Per-animal PSD -- {region}")
    fig.tight_layout()
    return fig


def plot_method_comparison(
    psd_tables: dict[str, pd.DataFrame],
    region: str,
    condition: str,
    max_freq: float = 0.5,
    xscale: str = "log",
    yscale: str = "log",
) -> plt.Figure:
    """Overlay group-average spectra from 2-3 methods on one axes, for a
    single region/condition. `psd_tables` maps a method label to its PSD
    table (as produced by psd_methods.compute_psd_table). Every method's
    selection is validated before any Figure is created, so a bad
    region/condition never leaves a half-built Figure open."""
    if not psd_tables:
        raise ValueError("psd_tables must contain at least one method")
    filtered = {
        label: _filtered_or_raise(
            table,
            (table["region"] == region)
            & (table["condition"] == condition)
            & (table["frequency"] <= max_freq),
            f"method={label!r}, region={region!r}, condition={condition!r} "
            f"with frequency <= {max_freq}",
        )
        for label, table in psd_tables.items()
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, sub in filtered.items():
        pivot = sub.pivot_table(index="frequency", columns="animal", values="power").sort_index()
        mean = pivot.mean(axis=1)
        ax.plot(mean.index, mean.values, label=label)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(f"Method comparison -- {region} / {condition}")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def band_power_percent_diff(
    band_tables: dict[str, pd.DataFrame], reference_method: str, band_names: list[str]
) -> pd.DataFrame:
    """Compare mean band power (averaged across all rows) for each method
    against `reference_method`, for each band in `band_names`. Returns one
    row per (method, band) with columns: method, band, mean_power,
    reference_power, percent_diff = (mean_power - reference_power) / reference_power * 100.

    percent_diff is NaN (not raised, not inf) when reference_power is
    exactly 0 -- a real band power of zero is a legitimate value, not an
    error, and NaN reads as "undefined" rather than as a measurement the
    way a divide-by-zero-shaped 0.0 or inf would.
    """
    if reference_method not in band_tables:
        raise ValueError(f"reference_method {reference_method!r} not in band_tables")
    for method, table in band_tables.items():
        missing = [b for b in band_names if b not in table.columns]
        if missing:
            raise ValueError(f"band(s) {missing} not found in band_tables[{method!r}]")

    reference_means = {b: float(band_tables[reference_method][b].mean()) for b in band_names}
    rows = []
    for method, table in band_tables.items():
        for band in band_names:
            mean_power = float(table[band].mean())
            ref_power = reference_means[band]
            percent_diff = (
                (mean_power - ref_power) / ref_power * 100.0 if ref_power != 0 else float("nan")
            )
            rows.append(
                {
                    "method": method,
                    "band": band,
                    "mean_power": mean_power,
                    "reference_power": ref_power,
                    "percent_diff": percent_diff,
                }
            )
    return pd.DataFrame(
        rows, columns=["method", "band", "mean_power", "reference_power", "percent_diff"]
    )
