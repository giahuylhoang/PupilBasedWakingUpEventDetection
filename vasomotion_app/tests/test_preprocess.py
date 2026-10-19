"""Tests for vasomotion_app.preprocess."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vasomotion_app import preprocess
from vasomotion_app.preprocess import (
    TRACE_STATS_COLUMNS,
    area_to_diameter,
    detrend_signal,
    highpass_filter,
    normalize_signal,
    preprocess_long_dataframe,
    preprocess_signal,
    preprocess_stages,
    trace_stats_table,
    trace_summary,
    trim_signal,
    validate_sampling_rate,
)


def test_trim_signal_drops_correct_count():
    values = np.arange(10.0)
    result = trim_signal(values, fs=2.0, trim_seconds=2.0)
    assert np.array_equal(result, np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))


def test_trim_signal_zero_is_noop():
    values = np.arange(5.0)
    result = trim_signal(values, fs=2.0, trim_seconds=0.0)
    assert np.array_equal(result, values)


def test_trim_signal_negative_raises():
    with pytest.raises(ValueError, match="trim_seconds"):
        trim_signal(np.arange(5.0), fs=2.0, trim_seconds=-1.0)


def test_area_to_diameter_takes_sqrt():
    result = area_to_diameter(np.array([4.0, 9.0, 16.0]), "area")
    assert np.allclose(result, [2.0, 3.0, 4.0])


def test_area_to_diameter_passthrough_for_diameter():
    values = np.array([4.0, 9.0, 16.0])
    result = area_to_diameter(values, "diameter")
    assert np.array_equal(result, values)


def test_area_to_diameter_invalid_type_raises():
    with pytest.raises(ValueError, match="signal_type"):
        area_to_diameter(np.array([1.0]), "bogus")


def test_normalize_dx_x0_median():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = normalize_signal(values, "dx_x0_median")
    assert np.allclose(result, [-2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3])


def test_normalize_dx_x0_mean():
    # Asymmetric fixture (median 3.0, mean 22.0) deliberately differs from
    # test_normalize_dx_x0_median's [1,2,3,4,5], where mean == median == 3.0
    # and the two methods' expected arrays used to be byte-identical -- a
    # swap of the mean/median implementations went undetected. Expected
    # values below are hand-computed from mean=22.0, not derived from the
    # implementation: (1-22)/22, (2-22)/22, (3-22)/22, (4-22)/22, (100-22)/22.
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    result = normalize_signal(values, "dx_x0_mean")
    assert np.allclose(result, [-21 / 22, -20 / 22, -19 / 22, -18 / 22, 78 / 22])


def test_normalize_zscore():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = normalize_signal(values, "zscore")
    expected_std = np.std(values, ddof=1)
    assert np.allclose(result, (values - 3.0) / expected_std)


def test_normalize_none_is_passthrough():
    values = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(normalize_signal(values, "none"), values)


def test_normalize_invalid_method_raises():
    with pytest.raises(ValueError, match="normalize method"):
        normalize_signal(np.array([1.0]), "bogus")


def test_detrend_constant_removes_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = detrend_signal(values, "constant")
    assert np.allclose(result, [-2.0, -1.0, 0.0, 1.0, 2.0])


def test_detrend_linear_removes_perfect_line():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = detrend_signal(values, "linear")
    assert np.allclose(result, 0.0, atol=1e-10)


def test_detrend_none_is_passthrough():
    values = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(detrend_signal(values, "none"), values)


def test_highpass_none_cutoff_is_passthrough():
    values = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(highpass_filter(values, fs=10.0, cutoff_hz=None, order=4), values)


def test_highpass_removes_dc_offset():
    values = np.full(1000, 100.0)
    result = highpass_filter(values, fs=10.0, cutoff_hz=0.5, order=4)
    assert np.allclose(result, 0.0, atol=1e-6)


def test_preprocess_signal_full_pipeline_constant_area():
    values = np.full(10, 100.0)  # constant area
    result = preprocess_signal(
        values, fs=2.0, signal_type="area", normalize="dx_x0_median",
        detrend_method="none", highpass_cutoff=None, highpass_order=4, trim_seconds=0.0,
    )
    # sqrt(100)=10 constant -> (10-10)/10=0 constant -> untouched by off highpass/none detrend
    assert np.allclose(result, 0.0)


def test_preprocess_signal_trims_before_normalizing():
    # Stage order is trim -> area_to_diameter -> normalize -> highpass ->
    # detrend, specifically so a transient at the start of a recording
    # cannot bias the normalisation baseline. trim_seconds=10.0 at fs=2.0
    # drops the first 20 samples (the 1000.0 block), leaving 20 samples of
    # 10.0 whose median is 10.0, so every output is (10-10)/10 = 0.0.
    #
    # If normalise ran BEFORE trim (the wrong order), the median would be
    # computed over all 40 samples: median([1000.0]*20 + [10.0]*20) = 505.0,
    # giving (10.0 - 505.0) / 505.0 = -0.9801980198019802 for every
    # surviving sample instead of 0.0.
    values = np.array([1000.0] * 20 + [10.0] * 20)
    result = preprocess_signal(
        values, fs=2.0, signal_type="diameter", normalize="dx_x0_median",
        detrend_method="none", highpass_cutoff=None, highpass_order=4, trim_seconds=10.0,
    )
    assert len(result) == 20
    assert np.allclose(result, 0.0)


def test_preprocess_signal_converts_area_before_normalizing():
    # Stage order is trim -> area_to_diameter -> normalize -> ..., so a raw
    # area trace is converted to diameter (sqrt) before the normalisation
    # baseline is computed. Diameters are 2.0 (low block) and 4.0 (high
    # block); median diameter is 3.0, giving (2-3)/3 = -1/3 and
    # (4-3)/3 = +1/3.
    #
    # If normalise ran BEFORE area_to_diameter (the wrong order), median
    # area would be median([4.0]*20 + [16.0]*20) = 10.0, giving -0.6 for the
    # low block and +0.6 for the high block; area_to_diameter's
    # sqrt(max(x, 0)) would then clip the negative -0.6 to 0.0 instead of
    # producing -1/3.
    values = np.array([4.0] * 20 + [16.0] * 20)
    result = preprocess_signal(
        values, fs=2.0, signal_type="area", normalize="dx_x0_median",
        detrend_method="none", highpass_cutoff=None, highpass_order=4, trim_seconds=0.0,
    )
    assert np.allclose(result[:20], -1 / 3)
    assert np.allclose(result[20:], 1 / 3)


def test_trace_summary():
    result = trace_summary(n_samples=1000, fs=2.0)
    assert result["n_samples"] == 1000
    assert result["duration_sec"] == pytest.approx(500.0)
    assert result["lowest_freq_hz"] == pytest.approx(0.002)


def test_validate_sampling_rate_rejects_non_positive():
    with pytest.raises(ValueError, match="positive"):
        validate_sampling_rate(fs=-1.0, n_samples=100)
    with pytest.raises(ValueError, match="positive"):
        validate_sampling_rate(fs=0.0, n_samples=100)


def test_validate_sampling_rate_warns_when_too_short():
    warnings = validate_sampling_rate(fs=1000.0, n_samples=5)
    assert len(warnings) == 1
    assert "duration" in warnings[0]


def test_validate_sampling_rate_warns_when_too_long():
    warnings = validate_sampling_rate(fs=0.0001, n_samples=10)
    assert len(warnings) == 1


def test_validate_sampling_rate_no_warning_for_plausible_duration():
    warnings = validate_sampling_rate(fs=1.0, n_samples=1000)
    assert warnings == []


def test_preprocess_long_dataframe_shape_and_trim_renumbering():
    df = pd.DataFrame(
        {
            "animal": [1, 1, 1, 1, 1],
            "sex": ["M"] * 5,
            "region": ["R"] * 5,
            "condition": ["C"] * 5,
            "sample": [1, 2, 3, 4, 5],
            "value": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    result = preprocess_long_dataframe(
        df, fs=1.0, signal_type="diameter", normalize="none",
        detrend_method="none", highpass_cutoff=None, highpass_order=4, trim_seconds=2.0,
    )
    assert list(result.columns) == ["animal", "sex", "region", "condition", "sample", "value"]
    assert len(result) == 3  # 5 samples - 2 dropped
    assert list(result["sample"]) == [1, 2, 3]
    assert np.allclose(result["value"], 10.0)


def test_preprocess_long_dataframe_multiple_groups_independent():
    df = pd.DataFrame(
        {
            "animal": [1, 1, 2, 2],
            "sex": ["M", "M", "F", "F"],
            "region": ["R", "R", "R", "R"],
            "condition": ["C", "C", "C", "C"],
            "sample": [1, 2, 1, 2],
            "value": [4.0, 4.0, 9.0, 9.0],
        }
    )
    result = preprocess_long_dataframe(
        df, fs=1.0, signal_type="area", normalize="none",
        detrend_method="none", highpass_cutoff=None, highpass_order=4, trim_seconds=0.0,
    )
    animal1 = result[result["animal"] == 1]["value"].to_numpy()
    animal2 = result[result["animal"] == 2]["value"].to_numpy()
    assert np.allclose(animal1, 2.0)  # sqrt(4)
    assert np.allclose(animal2, 3.0)  # sqrt(9)


def test_preprocess_long_dataframe_normalize_per_group():
    # normalize="none" in the two tests above proves row/identity
    # bookkeeping but never exercises the normalisation reduction itself.
    # Here animal 1 (~100 scale) and animal 2 (~1000 scale, 10x) each have
    # the same relative shape (-20%, 0%, +20% around their own median).
    # Correct per-group normalisation must reduce BOTH to the identical
    # trace [-0.2, 0.0, 0.2] -- proving each animal is normalised against
    # its own median, landing on a comparable dD/D0 scale despite the 10x
    # difference in raw units.
    #
    # A groupby-leakage bug (one median computed across the flattened
    # dataframe instead of per animal) would instead use
    # median([80,100,120,800,1000,1200]) = 460.0, giving animal 1 roughly
    # [-0.826, -0.783, -0.739] and animal 2 roughly [0.739, 1.174, 1.609] --
    # nothing close to the comparable-scale answer asserted below.
    df = pd.DataFrame(
        {
            "animal": [1, 1, 1, 2, 2, 2],
            "sex": ["M", "M", "M", "F", "F", "F"],
            "region": ["R"] * 6,
            "condition": ["C"] * 6,
            "sample": [1, 2, 3, 1, 2, 3],
            "value": [80.0, 100.0, 120.0, 800.0, 1000.0, 1200.0],
        }
    )
    result = preprocess_long_dataframe(
        df, fs=1.0, signal_type="diameter", normalize="dx_x0_median",
        detrend_method="none", highpass_cutoff=None, highpass_order=4, trim_seconds=0.0,
    )
    animal1 = result[result["animal"] == 1]["value"].to_numpy()
    animal2 = result[result["animal"] == 2]["value"].to_numpy()
    assert np.allclose(animal1, [-0.2, 0.0, 0.2])
    assert np.allclose(animal2, [-0.2, 0.0, 0.2])


# --- preprocess_stages / trace_stats_table -------------------------------
#
# These back the exported trace_stats.csv, whose whole purpose is to let a
# user compute variability outside the app. The risk they guard is a table
# of plausible-looking numbers that describe a different chain than the PSD
# did, so every test below pins a stage to an independently-known value
# rather than to whatever the function happens to return.


def _stats_long_df(values_by_animal: dict[int, list[float]]) -> pd.DataFrame:
    rows = []
    for animal, values in values_by_animal.items():
        for i, value in enumerate(values, start=1):
            rows.append(
                {
                    "animal": animal, "sex": "M", "region": "R", "condition": "C",
                    "sample": i, "value": float(value),
                }
            )
    return pd.DataFrame(rows)


def test_preprocess_stages_analyzed_is_exactly_preprocess_signal():
    # The reason preprocess_signal delegates to preprocess_stages instead of
    # repeating the chain: if these two ever diverge, trace_stats.csv would
    # describe a trace the PSD never saw.
    rng = np.random.default_rng(0)
    values = 100.0 + rng.standard_normal(400) * 5.0
    kwargs = dict(
        fs=2.0, signal_type="diameter", normalize="dx_x0_median",
        detrend_method="linear", highpass_cutoff=0.02, highpass_order=4,
        trim_seconds=1.0,
    )
    stages = preprocess_stages(values, **kwargs)
    assert np.array_equal(stages["analyzed"], preprocess_signal(values, **kwargs))


def test_preprocess_stages_reports_each_step_separately():
    # Area 400 -> diameter 20 by sqrt, and the conversion must land in
    # 'converted', not be folded into 'trimmed'.
    values = np.full(50, 400.0)
    stages = preprocess_stages(
        values, fs=1.0, signal_type="area", normalize="none",
        detrend_method="none", highpass_cutoff=None, highpass_order=4,
        trim_seconds=10.0,
    )
    assert len(stages["trimmed"]) == 40
    assert np.allclose(stages["trimmed"], 400.0)
    assert np.allclose(stages["converted"], 20.0)


def test_trace_stats_table_columns_and_one_row_per_group():
    df = _stats_long_df({1: [90.0, 100.0, 110.0], 2: [190.0, 200.0, 210.0]})
    stats = trace_stats_table(
        df, fs=1.0, signal_type="diameter", normalize="dx_x0_median",
        detrend_method="none", highpass_cutoff=None, highpass_order=4,
        trim_seconds=0.0,
    )
    assert list(stats.columns) == TRACE_STATS_COLUMNS
    assert len(stats) == 2
    assert stats["n_samples"].tolist() == [3, 3]


def test_trace_stats_mean_raw_is_post_conversion_not_raw_area():
    # signal_type='area' means mean_raw is in DIAMETER units. Reporting the
    # raw area mean (400) here would silently mislabel the column's units.
    df = _stats_long_df({1: [400.0] * 20})
    stats = trace_stats_table(
        df, fs=1.0, signal_type="area", normalize="none",
        detrend_method="none", highpass_cutoff=None, highpass_order=4,
        trim_seconds=0.0,
    )
    assert stats.iloc[0]["mean_raw"] == pytest.approx(20.0)
    assert stats.iloc[0]["sd_raw"] == pytest.approx(0.0)


def test_trace_stats_sd_normalized_is_before_highpass_and_detrend():
    # The distinction the two SD columns exist to make. A slow ramp plus a
    # fast wiggle: sd_normalized keeps the ramp's variance, sd_analyzed has
    # had it filtered and detrended away, so sd_analyzed must be clearly
    # smaller. If both columns reported the same stage this would fail.
    fs = 2.0
    t = np.arange(0, 600, 1 / fs)
    values = 100.0 + 20.0 * (t / t[-1]) + 0.5 * np.sin(2 * np.pi * 0.1 * t)
    df = _stats_long_df({1: values.tolist()})
    stats = trace_stats_table(
        df, fs=fs, signal_type="diameter", normalize="dx_x0_median",
        detrend_method="linear", highpass_cutoff=0.02, highpass_order=4,
        trim_seconds=0.0,
    )
    row = stats.iloc[0]
    assert row["sd_normalized"] > row["sd_analyzed"] * 5
    # And sd_normalized really is the normalised trace's own SD, computed
    # independently of trace_stats_table's internals.
    expected = np.std(
        preprocess_stages(
            values, fs=fs, signal_type="diameter", normalize="dx_x0_median",
            detrend_method="linear", highpass_cutoff=0.02, highpass_order=4,
            trim_seconds=0.0,
        )["normalized"],
        ddof=1,
    )
    assert row["sd_normalized"] == pytest.approx(expected)


def test_trace_stats_sd_normalized_is_one_under_zscore():
    # Degenerate by construction -- the app warns about it, so the value
    # itself must actually be 1.0 rather than something near it.
    rng = np.random.default_rng(1)
    df = _stats_long_df({1: (100.0 + rng.standard_normal(300) * 3.0).tolist()})
    stats = trace_stats_table(
        df, fs=1.0, signal_type="diameter", normalize="zscore",
        detrend_method="none", highpass_cutoff=None, highpass_order=4,
        trim_seconds=0.0,
    )
    assert stats.iloc[0]["sd_normalized"] == pytest.approx(1.0)


def test_trace_stats_respects_trim_in_n_samples_and_duration():
    df = _stats_long_df({1: np.linspace(90.0, 110.0, 100).tolist()})
    stats = trace_stats_table(
        df, fs=2.0, signal_type="diameter", normalize="none",
        detrend_method="none", highpass_cutoff=None, highpass_order=4,
        trim_seconds=10.0,
    )
    row = stats.iloc[0]
    assert row["n_samples"] == 80
    assert row["duration_sec"] == pytest.approx(40.0)


# --------------------------------------------------------------------------
# Rolling (local) baseline normalisation
# --------------------------------------------------------------------------

def test_rolling_baseline_tracks_a_drifting_baseline():
    fs = 4.0
    t = np.arange(0, 300, 1 / fs)
    drifting = 1000.0 + np.linspace(0, 500, len(t))
    baseline = preprocess.rolling_baseline(drifting, fs, 50.0)
    # A rolling median of a monotonic ramp follows the ramp.
    assert baseline[0] < baseline[len(baseline) // 2] < baseline[-1]
    assert np.corrcoef(baseline, drifting)[0, 1] > 0.999


def test_rolling_baseline_has_no_nan_at_the_edges():
    """min_periods=1 rather than a NaN edge: on a 300 s recording a 200 s
    centred window would otherwise blank the first and last ~100 s, which is
    most of the data."""
    fs = 4.0
    values = np.arange(1000.0, 2000.0)
    baseline = preprocess.rolling_baseline(values, fs, 100.0)
    assert len(baseline) == len(values)
    assert not np.isnan(baseline).any()


def test_rolling_baseline_window_is_centred_not_lagged():
    """A centred window must not shift the baseline in time. A trailing
    window would lag the ramp and leave a constant positive residual."""
    fs = 4.0
    values = 1000.0 + np.arange(400) * 2.0
    baseline = preprocess.rolling_baseline(values, fs, 20.0)
    mid = slice(100, 300)  # away from the shrinking edge windows
    residual = values[mid] - baseline[mid]
    assert abs(float(np.mean(residual))) < 1e-6, "centred window should leave no net offset"


def test_rolling_baseline_uses_an_odd_window_length():
    fs = 10.0
    values = np.arange(500.0)
    # 2.0 s at 10 Hz is 20 samples -> must be bumped to 21 so it can centre.
    assert np.array_equal(
        preprocess.rolling_baseline(values, fs, 2.0),
        pd.Series(values).rolling(window=21, center=True, min_periods=1).median().to_numpy(),
    )


def test_rolling_baseline_rejects_a_non_positive_window():
    with pytest.raises(ValueError, match="window_seconds must be > 0"):
        preprocess.rolling_baseline(np.arange(10.0), 4.0, 0.0)


def test_normalize_dx_x0_rolling_removes_drift_that_global_normalisation_keeps():
    """The thing a local baseline is actually for: a trace whose baseline
    shifts should come out centred on zero throughout, not drifting."""
    fs = 4.0
    t = np.arange(0, 300, 1 / fs)
    drifting = 1000.0 + np.linspace(0, 600, len(t)) + 20.0 * np.sin(2 * np.pi * 0.08 * t)

    rolled = preprocess.normalize_signal(
        drifting, "dx_x0_rolling", fs=fs, rolling_window_seconds=30.0
    )
    globally = preprocess.normalize_signal(drifting, "dx_x0_median")

    first_half_r, second_half_r = np.mean(rolled[: len(rolled) // 2]), np.mean(rolled[len(rolled) // 2 :])
    first_half_g, second_half_g = np.mean(globally[: len(globally) // 2]), np.mean(globally[len(globally) // 2 :])
    assert abs(first_half_r - second_half_r) < 0.01, "rolling baseline should remove the drift"
    assert abs(first_half_g - second_half_g) > 0.2, "global baseline keeps it (that is the point)"


def test_normalize_dx_x0_rolling_requires_fs():
    with pytest.raises(ValueError, match="fs is required"):
        preprocess.normalize_signal(np.arange(100.0), "dx_x0_rolling")


def test_rolling_is_registered_as_a_normalize_method():
    assert "dx_x0_rolling" in preprocess.NORMALIZE_METHODS
    # And the existing modes are untouched, in order.
    assert preprocess.NORMALIZE_METHODS[:4] == ("none", "dx_x0_median", "dx_x0_mean", "zscore")


@pytest.mark.parametrize(
    "window_sec,min_recovered,max_recovered",
    [
        (10.0, 0.10, 0.25),
        (20.0, 0.25, 0.40),
        (30.0, 0.38, 0.55),
        (50.0, 0.65, 0.80),
        (65.0, 0.80, 0.95),
        (100.0, 0.95, 1.10),
    ],
)
def test_short_rolling_window_suppresses_the_band_it_overlaps(
    window_sec, min_recovered, max_recovered
):
    """The physics behind the guard, pinned against measured values.

    Subtracting a rolling median high-passes at roughly 1/window, so a
    window shorter than the period being measured eats the signal. Measured
    recovery of a pure 0.01 Hz (100 s) oscillation, as a fraction of what
    global dx/x0 recovers: 0.15 at a 10 s baseline, 0.45 at 30 s, 0.87 at
    65 s, and 1.00 once the window reaches the 100 s period itself. This is
    why a local baseline is NOT simply better than a global one, and why
    `validate_rolling_window` demands window >= 1 / lowest_band_edge."""
    fs = 3.91
    t = np.arange(0, 300, 1 / fs)
    signal = 1500.0 + 300.0 * np.sin(2 * np.pi * 0.01 * t)

    rolled = preprocess.normalize_signal(
        signal, "dx_x0_rolling", fs=fs, rolling_window_seconds=window_sec
    )
    globally = preprocess.normalize_signal(signal, "dx_x0_median")
    recovered = (rolled.max() - rolled.min()) / (globally.max() - globally.min())
    assert min_recovered <= recovered <= max_recovered, (
        f"{window_sec} s baseline recovered {recovered:.3f} of the true amplitude, "
        f"expected {min_recovered}-{max_recovered}"
    )


def test_rolling_suppression_increases_monotonically_with_window_length():
    """The ordering is the part that must never regress: a longer baseline
    window always preserves at least as much of a slow oscillation."""
    fs = 3.91
    t = np.arange(0, 300, 1 / fs)
    signal = 1500.0 + 300.0 * np.sin(2 * np.pi * 0.01 * t)
    recovered = [
        (lambda r: r.max() - r.min())(
            preprocess.normalize_signal(
                signal, "dx_x0_rolling", fs=fs, rolling_window_seconds=w
            )
        )
        for w in (10.0, 20.0, 30.0, 50.0, 65.0, 100.0)
    ]
    assert all(a < b for a, b in zip(recovered, recovered[1:])), recovered


def test_validate_rolling_window_warns_when_shorter_than_the_lowest_band_period():
    bands = {"ultra_low": (0.005, 0.04), "myogenic": (0.04, 0.15)}
    warnings = preprocess.validate_rolling_window(65.0, bands, n_samples=1173, fs=3.91)
    assert len(warnings) == 1
    assert "shorter than the longest period" in warnings[0]
    assert "200 s" in warnings[0]  # 1 / 0.005 Hz
    assert "0.005 Hz" in warnings[0]


def test_validate_rolling_window_accepts_a_window_long_enough_for_the_bands():
    bands = {"ultra_low": (0.005, 0.04), "myogenic": (0.04, 0.15)}
    assert preprocess.validate_rolling_window(200.0, bands, n_samples=1173, fs=3.91) == []


def test_validate_rolling_window_scales_its_demand_to_the_bands_requested():
    """A myogenic-only analysis (0.04 Hz = 25 s) is safe with a short window
    that the ultra-low band would reject. The guard must follow the bands,
    not a hardcoded minimum."""
    myogenic_only = {"myogenic": (0.04, 0.15)}
    assert preprocess.validate_rolling_window(65.0, myogenic_only, n_samples=1173, fs=3.91) == []
    assert preprocess.validate_rolling_window(10.0, myogenic_only, n_samples=1173, fs=3.91)


def test_validate_rolling_window_warns_when_longer_than_the_recording():
    bands = {"myogenic": (0.04, 0.15)}
    warnings = preprocess.validate_rolling_window(400.0, bands, n_samples=1173, fs=3.91)
    assert len(warnings) == 1
    assert "effectively the global median" in warnings[0]


def test_validate_rolling_window_rejects_a_non_positive_window():
    warnings = preprocess.validate_rolling_window(0.0, {"b": (0.01, 0.1)}, 1000, 4.0)
    assert warnings and "must be > 0" in warnings[0]


def test_validate_rolling_window_handles_no_bands():
    assert preprocess.validate_rolling_window(100.0, None, n_samples=1173, fs=3.91) == []
    assert preprocess.validate_rolling_window(100.0, {}, n_samples=1173, fs=3.91) == []


def test_preprocess_stages_threads_the_rolling_window_through():
    """The stage chain must actually use the requested window, not the
    module default -- otherwise the control would silently do nothing."""
    fs = 4.0
    t = np.arange(0, 200, 1 / fs)
    values = 1000.0 + np.linspace(0, 400, len(t))
    short = preprocess.preprocess_stages(
        values, fs=fs, signal_type="diameter", normalize="dx_x0_rolling",
        detrend_method="none", highpass_cutoff=None, rolling_window_seconds=20.0,
    )["normalized"]
    long = preprocess.preprocess_stages(
        values, fs=fs, signal_type="diameter", normalize="dx_x0_rolling",
        detrend_method="none", highpass_cutoff=None, rolling_window_seconds=150.0,
    )["normalized"]
    assert not np.allclose(short, long), "window length must change the result"
    assert np.std(short) < np.std(long), "a shorter baseline removes more of the drift"


def test_trace_stats_table_accepts_the_rolling_window():
    df = pd.DataFrame(
        {
            "animal": [1] * 200, "sex": ["M"] * 200, "region": ["R1"] * 200,
            "condition": ["pre"] * 200, "sample": range(1, 201),
            "value": 1000.0 + np.arange(200.0),
        }
    )
    out = preprocess.trace_stats_table(
        df, fs=4.0, signal_type="diameter", normalize="dx_x0_rolling",
        highpass_cutoff=None, rolling_window_seconds=25.0,
    )
    assert len(out) == 1
    assert np.isfinite(out.iloc[0]["sd_normalized"])
