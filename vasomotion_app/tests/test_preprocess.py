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
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = normalize_signal(values, "dx_x0_mean")
    assert np.allclose(result, [-2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3])


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
