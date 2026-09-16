"""Tests for vasomotion_app.figures.

Every test asserts the *data* that reached the plot (line/collection
values, axis scales, labels, visibility) rather than merely that a Figure
object came back -- see task-5-brief.md's "test quality bar": a mean
plotted as a median, or an SD plotted as an SEM, must make a test go red.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from vasomotion_app.figures import (
    band_power_percent_diff,
    plot_group_average,
    plot_method_comparison,
    plot_per_animal_small_multiples,
)


def _animal_table(region, condition, animal, power_scale, n_freq=5, max_freq=0.5, sex="M"):
    """One (animal, region, condition) block: flat power across frequency,
    so the group mean/SEM at any frequency is trivial to hand-compute."""
    freqs = np.linspace(0.01, max_freq, n_freq)
    return pd.DataFrame(
        {
            "animal": [animal] * n_freq,
            "sex": [sex] * n_freq,
            "region": [region] * n_freq,
            "condition": [condition] * n_freq,
            "frequency": freqs,
            "power": np.full(n_freq, float(power_scale)),
        }
    )


def _extra_row(region, condition, animal, frequency, power):
    return pd.DataFrame(
        [{"animal": animal, "sex": "M", "region": region, "condition": condition,
          "frequency": frequency, "power": power}]
    )


def _basic_table():
    """Two animals, one region, one condition -- enough to exercise SEM
    without a ddof=1 single-sample NaN."""
    return pd.concat(
        [_animal_table("R1", "pre", 1, 1.0), _animal_table("R1", "pre", 2, 3.0)],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# plot_group_average
# ---------------------------------------------------------------------------


def test_plot_group_average_mean_and_sem_match_hand_computation_not_median_or_sd():
    # Asymmetric powers (1, 2, 6) so mean (3.0) != median (2.0), and SEM !=
    # SD, so a median-instead-of-mean or SD-instead-of-SEM bug goes red.
    table = pd.concat(
        [
            _animal_table("R1", "pre", 1, 1.0),
            _animal_table("R1", "pre", 2, 2.0),
            _animal_table("R1", "pre", 3, 6.0),
        ],
        ignore_index=True,
    )
    values = np.array([1.0, 2.0, 6.0])
    expected_mean = values.mean()
    expected_median = np.median(values)
    expected_sem = values.std(ddof=1) / np.sqrt(3)
    expected_sd = values.std(ddof=1)
    assert expected_mean != expected_median  # sanity: fixture can discriminate
    assert expected_sem != expected_sd

    fig = plot_group_average(table, region="R1", max_freq=0.5)
    ax = fig.axes[0]
    assert len(ax.lines) == 1
    line = ax.lines[0]
    y = line.get_ydata()
    assert np.allclose(y, expected_mean)
    assert not np.allclose(y, expected_median)

    assert len(ax.collections) == 1  # the fill_between SEM band
    verts = ax.collections[0].get_paths()[0].vertices
    half_widths = set(np.round(np.abs(verts[:, 1] - expected_mean), 6))
    assert half_widths == {round(expected_sem, 6)}
    assert round(expected_sd, 6) not in half_widths
    plt.close(fig)


def test_plot_group_average_sem_uses_per_frequency_animal_count_not_total_columns():
    # Animal 3 is missing the middle frequency point (ragged coverage). At
    # that frequency only animals 1 and 2 contribute, so SEM there must
    # divide by sqrt(2), not sqrt(3) (the total distinct-animal count for
    # the condition) -- a stale "pivot.shape[1]" denominator bug.
    freqs = [0.1, 0.2, 0.3]
    rows = []
    for f in freqs:
        rows.append({"animal": 1, "sex": "M", "region": "R1", "condition": "pre",
                      "frequency": f, "power": 1.0})
        rows.append({"animal": 2, "sex": "M", "region": "R1", "condition": "pre",
                      "frequency": f, "power": 2.0})
    rows.append({"animal": 3, "sex": "M", "region": "R1", "condition": "pre",
                 "frequency": 0.1, "power": 6.0})
    rows.append({"animal": 3, "sex": "M", "region": "R1", "condition": "pre",
                 "frequency": 0.3, "power": 6.0})
    table = pd.DataFrame(rows)

    fig = plot_group_average(table, region="R1", max_freq=0.5)
    ax = fig.axes[0]
    verts = ax.collections[0].get_paths()[0].vertices

    mean_at_02 = np.mean([1.0, 2.0])
    correct_sem = np.std([1.0, 2.0], ddof=1) / np.sqrt(2)
    buggy_sem = np.std([1.0, 2.0], ddof=1) / np.sqrt(3)

    ys_at_02 = verts[np.isclose(verts[:, 0], 0.2), 1]
    assert len(ys_at_02) > 0
    half_widths = set(np.round(np.abs(ys_at_02 - mean_at_02), 6))
    assert half_widths == {round(correct_sem, 6)}
    assert round(buggy_sem, 6) not in half_widths
    plt.close(fig)


def test_plot_group_average_two_conditions_each_line_matches_its_own_mean_and_n():
    table = pd.concat(
        [
            _animal_table("R1", "pre", 1, 1.0),
            _animal_table("R1", "pre", 2, 2.0),
            _animal_table("R1", "pre", 3, 6.0),
            _animal_table("R1", "TMX", 1, 10.0),
            _animal_table("R1", "TMX", 2, 20.0),
        ],
        ignore_index=True,
    )
    fig = plot_group_average(table, region="R1", max_freq=0.5)
    ax = fig.axes[0]
    assert len(ax.lines) == 2  # one mean line per condition (pre, TMX)
    by_label = {line.get_label(): line for line in ax.lines}
    pre_label = next(l for l in by_label if l.startswith("pre"))
    tmx_label = next(l for l in by_label if l.startswith("TMX"))
    assert "n=3" in pre_label
    assert "n=2" in tmx_label
    assert np.allclose(by_label[pre_label].get_ydata(), 3.0)
    assert np.allclose(by_label[tmx_label].get_ydata(), 15.0)
    plt.close(fig)


def test_plot_group_average_excludes_other_regions_and_frequencies_above_max():
    table = pd.concat(
        [
            _animal_table("R1", "pre", 1, 1.0),
            _animal_table("R1", "pre", 2, 3.0),
            _animal_table("R2", "pre", 1, 999.0),  # other region: must be excluded
            _extra_row("R1", "pre", 1, 10.0, 999.0),  # above max_freq: must be excluded
        ],
        ignore_index=True,
    )
    fig = plot_group_average(table, region="R1", max_freq=0.5)
    ax = fig.axes[0]
    line = ax.lines[0]
    assert np.allclose(line.get_ydata(), 2.0)  # mean of 1.0 and 3.0 only
    assert line.get_xdata().max() <= 0.5
    plt.close(fig)


def test_plot_group_average_raises_for_region_with_no_data():
    table = _basic_table()
    with pytest.raises(ValueError, match="R2"):
        plot_group_average(table, region="R2")


@pytest.mark.parametrize("xscale,yscale", [("log", "log"), ("linear", "linear")])
def test_plot_group_average_labels_title_and_scales(xscale, yscale):
    fig = plot_group_average(_basic_table(), region="R1", xscale=xscale, yscale=yscale)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Frequency (Hz)"
    assert ax.get_ylabel() == "Power"
    assert ax.get_xscale() == xscale
    assert ax.get_yscale() == yscale
    assert "R1" in ax.get_title()
    plt.close(fig)


def test_plot_group_average_defaults_to_log_log_scale():
    # Fix round 1, Finding 1: test_plot_group_average_labels_title_and_scales
    # above always passes xscale/yscale explicitly, so it exercises the
    # override path but never the *default* parameter values. PSD data
    # spans decades; a silent flip of the default to linear would make the
    # ultra-low band -- the one this tool exists to measure -- unreadable,
    # and nothing would object. This test calls with no xscale/yscale at all.
    fig = plot_group_average(_basic_table(), region="R1")
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_per_animal_small_multiples
# ---------------------------------------------------------------------------


def test_plot_per_animal_small_multiples_one_subplot_per_animal():
    table = pd.concat(
        [_animal_table("R1", "pre", 1, 1.0), _animal_table("R1", "pre", 2, 2.0)],
        ignore_index=True,
    )
    fig = plot_per_animal_small_multiples(table, region="R1", max_freq=0.5)
    assert isinstance(fig, matplotlib.figure.Figure)
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible_axes) == 2  # 2 animals
    plt.close(fig)


def test_plot_per_animal_small_multiples_hides_leftover_grid_slots():
    # 4 animals -> ncols=min(3,4)=3, nrows=ceil(4/3)=2 -> 6 slots, 2 unused.
    # axis("off") alone does NOT change get_visible() (verified empirically
    # against matplotlib 3.10: an Axes stays visible=True after axis("off")),
    # so the leftover slots must be explicitly set_visible(False) for this
    # assertion to mean anything.
    table = pd.concat(
        [_animal_table("R1", "pre", a, float(a)) for a in (1, 2, 3, 4)], ignore_index=True
    )
    fig = plot_per_animal_small_multiples(table, region="R1", max_freq=0.5)
    assert len(fig.axes) == 6
    visible = [ax for ax in fig.axes if ax.get_visible()]
    hidden = [ax for ax in fig.axes if not ax.get_visible()]
    assert len(visible) == 4
    assert len(hidden) == 2
    plt.close(fig)


def test_plot_per_animal_small_multiples_plots_correct_series_per_animal():
    table = pd.concat(
        [
            _animal_table("R1", "pre", 1, 1.0),
            _animal_table("R1", "TMX", 1, 5.0),
            _animal_table("R1", "pre", 2, 2.0),
        ],
        ignore_index=True,
    )
    fig = plot_per_animal_small_multiples(table, region="R1", max_freq=0.5)
    by_title = {ax.get_title(): ax for ax in fig.axes if ax.get_visible()}
    assert set(by_title) == {"1", "2"}

    animal1_lines = {line.get_label(): line for line in by_title["1"].lines}
    assert set(animal1_lines) == {"pre", "TMX"}
    assert np.allclose(animal1_lines["pre"].get_ydata(), 1.0)
    assert np.allclose(animal1_lines["TMX"].get_ydata(), 5.0)

    animal2_lines = {line.get_label(): line for line in by_title["2"].lines}
    assert set(animal2_lines) == {"pre"}
    assert np.allclose(animal2_lines["pre"].get_ydata(), 2.0)
    plt.close(fig)


def test_plot_per_animal_small_multiples_excludes_frequencies_above_max():
    table = pd.concat(
        [_animal_table("R1", "pre", 1, 1.0), _extra_row("R1", "pre", 1, 10.0, 999.0)],
        ignore_index=True,
    )
    fig = plot_per_animal_small_multiples(table, region="R1", max_freq=0.5)
    line = fig.axes[0].lines[0]
    assert line.get_xdata().max() <= 0.5
    assert np.allclose(line.get_ydata(), 1.0)
    plt.close(fig)


def test_plot_per_animal_small_multiples_raises_for_region_with_no_data():
    table = _basic_table()
    with pytest.raises(ValueError, match="R2"):
        plot_per_animal_small_multiples(table, region="R2")


def test_plot_per_animal_small_multiples_uses_log_log_scale():
    # Fix round 1, Finding 1: scale here is hardcoded (not a parameter), so
    # there is no override path to distinguish from a default -- check at
    # least one visible subplot's axes directly.
    table = pd.concat(
        [_animal_table("R1", "pre", 1, 1.0), _animal_table("R1", "pre", 2, 2.0)],
        ignore_index=True,
    )
    fig = plot_per_animal_small_multiples(table, region="R1", max_freq=0.5)
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert visible_axes[0].get_xscale() == "log"
    assert visible_axes[0].get_yscale() == "log"
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_method_comparison
# ---------------------------------------------------------------------------


def test_plot_method_comparison_one_line_per_method_matching_values():
    tables = {
        "periodogram": _animal_table("R1", "pre", 1, 1.0),
        "multitaper_welch": _animal_table("R1", "pre", 1, 0.8),
    }
    fig = plot_method_comparison(tables, region="R1", condition="pre", max_freq=0.5)
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    by_label = {line.get_label(): line for line in ax.lines}
    assert set(by_label) == {"periodogram", "multitaper_welch"}
    assert np.allclose(by_label["periodogram"].get_ydata(), 1.0)
    assert np.allclose(by_label["multitaper_welch"].get_ydata(), 0.8)
    plt.close(fig)


def test_plot_method_comparison_averages_across_animals_within_a_method():
    table_a = pd.concat(
        [_animal_table("R1", "pre", 1, 2.0), _animal_table("R1", "pre", 2, 4.0)],
        ignore_index=True,
    )
    fig = plot_method_comparison({"methodA": table_a}, region="R1", condition="pre", max_freq=0.5)
    line = fig.axes[0].lines[0]
    assert np.allclose(line.get_ydata(), 3.0)  # mean of 2.0 and 4.0
    plt.close(fig)


def test_plot_method_comparison_filters_condition_region_and_max_freq():
    table = pd.concat(
        [
            _animal_table("R1", "pre", 1, 1.0),
            _animal_table("R1", "TMX", 1, 999.0),  # wrong condition
            _animal_table("R2", "pre", 1, 999.0),  # wrong region
            _extra_row("R1", "pre", 1, 10.0, 999.0),  # above max_freq
        ],
        ignore_index=True,
    )
    fig = plot_method_comparison({"methodA": table}, region="R1", condition="pre", max_freq=0.5)
    line = fig.axes[0].lines[0]
    assert np.allclose(line.get_ydata(), 1.0)
    assert line.get_xdata().max() <= 0.5
    plt.close(fig)


def test_plot_method_comparison_raises_for_missing_condition():
    tables = {"A": _animal_table("R1", "pre", 1, 1.0)}
    with pytest.raises(ValueError, match="TMX"):
        plot_method_comparison(tables, region="R1", condition="TMX")


def test_plot_method_comparison_raises_for_empty_psd_tables():
    with pytest.raises(ValueError, match="psd_tables"):
        plot_method_comparison({}, region="R1", condition="pre")


def test_plot_method_comparison_uses_log_log_scale():
    # Fix round 1, Finding 1: scale here is hardcoded (not a parameter).
    tables = {"methodA": _animal_table("R1", "pre", 1, 1.0)}
    fig = plot_method_comparison(tables, region="R1", condition="pre", max_freq=0.5)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    plt.close(fig)


# ---------------------------------------------------------------------------
# band_power_percent_diff
# ---------------------------------------------------------------------------


def test_band_power_percent_diff_known_values():
    band_tables = {
        "A": pd.DataFrame({"ultra_low": [10.0, 10.0]}),
        "B": pd.DataFrame({"ultra_low": [12.0, 12.0]}),
    }
    result = band_power_percent_diff(band_tables, reference_method="A", band_names=["ultra_low"])
    assert set(result.columns) == {"method", "band", "mean_power", "reference_power", "percent_diff"}
    row_a = result[result["method"] == "A"].iloc[0]
    row_b = result[result["method"] == "B"].iloc[0]
    assert row_a["percent_diff"] == pytest.approx(0.0)
    assert row_b["percent_diff"] == pytest.approx(20.0)


def test_band_power_percent_diff_uses_mean_not_sum_across_rows():
    # Different row counts per method: a sum-based implementation would
    # give (36-20)/20*100 = 80.0 instead of the correct (12-10)/10*100 = 20.0.
    band_tables = {
        "A": pd.DataFrame({"ultra_low": [10.0, 10.0]}),
        "B": pd.DataFrame({"ultra_low": [12.0, 12.0, 12.0]}),
    }
    result = band_power_percent_diff(band_tables, reference_method="A", band_names=["ultra_low"])
    row_b = result[result["method"] == "B"].iloc[0]
    assert row_b["mean_power"] == pytest.approx(12.0)
    assert row_b["percent_diff"] == pytest.approx(20.0)


def test_band_power_percent_diff_multiple_bands_and_methods_no_crosstalk():
    band_tables = {
        "A": pd.DataFrame({"low": [10.0], "high": [100.0]}),
        "B": pd.DataFrame({"low": [15.0], "high": [80.0]}),
        "C": pd.DataFrame({"low": [5.0], "high": [120.0]}),
    }
    result = band_power_percent_diff(band_tables, reference_method="A", band_names=["low", "high"])
    assert len(result) == 6  # 3 methods x 2 bands

    def pct(method, band):
        row = result[(result["method"] == method) & (result["band"] == band)]
        assert len(row) == 1
        return row.iloc[0]["percent_diff"]

    assert pct("A", "low") == pytest.approx(0.0)
    assert pct("A", "high") == pytest.approx(0.0)
    assert pct("B", "low") == pytest.approx(50.0)
    assert pct("B", "high") == pytest.approx(-20.0)
    assert pct("C", "low") == pytest.approx(-50.0)
    assert pct("C", "high") == pytest.approx(20.0)


def test_band_power_percent_diff_zero_reference_power_is_nan_not_error_or_inf():
    band_tables = {
        "A": pd.DataFrame({"low": [0.0, 0.0]}),
        "B": pd.DataFrame({"low": [5.0]}),
    }
    result = band_power_percent_diff(band_tables, reference_method="A", band_names=["low"])
    assert np.isnan(result[result["method"] == "A"].iloc[0]["percent_diff"])
    assert np.isnan(result[result["method"] == "B"].iloc[0]["percent_diff"])


def test_band_power_percent_diff_rejects_unknown_reference():
    band_tables = {"A": pd.DataFrame({"x": [1.0]})}
    with pytest.raises(ValueError, match="reference_method"):
        band_power_percent_diff(band_tables, reference_method="Z", band_names=["x"])


def test_band_power_percent_diff_rejects_band_missing_from_a_table():
    band_tables = {
        "A": pd.DataFrame({"low": [10.0]}),
        "B": pd.DataFrame({"other": [5.0]}),  # missing "low"
    }
    with pytest.raises(ValueError, match="low"):
        band_power_percent_diff(band_tables, reference_method="A", band_names=["low"])


# --------------------------------------------------------------------------
# xscale reaches every figure, not just the group average
# --------------------------------------------------------------------------

def _xscale_psd_table():
    rows = []
    for animal in (1, 2):
        for condition in ("pre", "TMX"):
            for freq in (0.01, 0.02, 0.05, 0.1, 0.2):
                rows.append(
                    {
                        "animal": animal, "sex": "M", "region": "R1",
                        "condition": condition, "frequency": freq, "power": 1.0 / freq,
                    }
                )
    return pd.DataFrame(rows)


def test_per_animal_small_multiples_defaults_to_log():
    """The default must not change: every existing caller and the saved
    figures in the results zip rely on it."""
    fig = plot_per_animal_small_multiples(_xscale_psd_table(), "R1")
    try:
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert visible, "precondition: at least one visible axes"
        for ax in visible:
            assert ax.get_xscale() == "log"
            assert ax.get_yscale() == "log"
    finally:
        plt.close(fig)


def test_per_animal_small_multiples_honours_a_linear_frequency_axis():
    """Regression lock on the fix: this figure previously stayed on a log
    frequency axis while the group-average figure directly above it switched
    to linear, which reads as a rendering bug. The y axis must stay log --
    only the frequency axis is user-controlled here."""
    fig = plot_per_animal_small_multiples(
        _xscale_psd_table(), "R1", xscale="linear"
    )
    try:
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert visible
        for ax in visible:
            assert ax.get_xscale() == "linear"
            assert ax.get_yscale() == "log"
    finally:
        plt.close(fig)


def test_per_animal_small_multiples_applies_the_scale_to_every_subplot():
    """Not just the first one: a per-axes loop is easy to write so the scale
    is set once outside it."""
    table = _xscale_psd_table()
    extra = table[table["animal"] == 1].copy()
    for animal in (3, 4, 5):
        clone = extra.copy()
        clone["animal"] = animal
        table = pd.concat([table, clone], ignore_index=True)
    fig = plot_per_animal_small_multiples(table, "R1", xscale="linear")
    try:
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 5
        assert all(ax.get_xscale() == "linear" for ax in visible)
    finally:
        plt.close(fig)


def test_method_comparison_honours_the_frequency_scale():
    table = _xscale_psd_table()
    tables = {"periodogram": table, "multitaper_welch": table}
    fig = plot_method_comparison(tables, "R1", "pre", xscale="linear")
    try:
        assert fig.axes[0].get_xscale() == "linear"
        assert fig.axes[0].get_yscale() == "log"
    finally:
        plt.close(fig)

    fig = plot_method_comparison(tables, "R1", "pre")
    try:
        assert fig.axes[0].get_xscale() == "log"
    finally:
        plt.close(fig)
