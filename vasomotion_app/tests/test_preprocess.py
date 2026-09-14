"""Tests for vasomotion_app.preprocess."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vasomotion_app.preprocess import (
    area_to_diameter,
    detrend_signal,
    highpass_filter,
    normalize_signal,
    preprocess_long_dataframe,
    preprocess_signal,
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
