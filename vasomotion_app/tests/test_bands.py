"""Tests for vasomotion_app.bands."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vasomotion_app.bands import integrate_bands, validate_band_edges, validate_band_names


def test_validate_band_names_accepts_valid_identifiers():
    validate_band_names({"ultra_low": (0.005, 0.04), "myogenic": (0.04, 0.15)})


def test_validate_band_names_rejects_non_identifier():
    with pytest.raises(ValueError, match="not a valid identifier"):
        validate_band_names({"ultra-low": (0.005, 0.04)})


def test_validate_band_names_rejects_keyword():
    with pytest.raises(ValueError, match="not a valid identifier"):
        validate_band_names({"class": (0.005, 0.04)})


def test_validate_band_names_rejects_inverted_edges():
    with pytest.raises(ValueError, match="lower edge"):
        validate_band_names({"bad": (0.1, 0.05)})


def test_validate_band_edges_warns_below_resolution():
    # window_length=256, fs=2.0 -> window_duration=128s -> min resolvable = 1/128 = 0.0078125 Hz
    warnings = validate_band_edges({"ultra_low": (0.005, 0.04)}, window_length=256, fs=2.0)
    assert len(warnings) == 1
    assert "ultra_low" in warnings[0]


def test_validate_band_edges_no_warning_when_above_resolution():
    warnings = validate_band_edges({"myogenic": (0.04, 0.15)}, window_length=256, fs=2.0)
    assert warnings == []


def _flat_psd_table():
    freqs = np.linspace(0.0, 1.0, 101)  # step 0.01 Hz
    power = np.full_like(freqs, 2.0)
    return pd.DataFrame(
        {
            "animal": [1] * 101,
            "sex": ["M"] * 101,
            "region": ["R"] * 101,
            "condition": ["C"] * 101,
            "frequency": freqs,
            "power": power,
        }
    )


def test_integrate_bands_flat_psd_exact_trapezoid():
    table = _flat_psd_table()
    result = integrate_bands(table, {"mid": (0.1, 0.4)})
    assert list(result.columns) == ["animal", "sex", "region", "condition", "mid"]
    assert len(result) == 1
    # trapezoidal rule is exact for a constant function: 2.0 * (0.4 - 0.1) = 0.6
    assert result.iloc[0]["mid"] == pytest.approx(0.6, rel=1e-6)


def test_integrate_bands_two_bands_two_groups():
    table = pd.concat(
        [_flat_psd_table(), _flat_psd_table().assign(animal=2, power=4.0)],
        ignore_index=True,
    )
    result = integrate_bands(table, {"low": (0.0, 0.2), "high": (0.6, 1.0)})
    assert set(result.columns) == {"animal", "sex", "region", "condition", "low", "high"}
    assert len(result) == 2
    row1 = result[result["animal"] == 1].iloc[0]
    row2 = result[result["animal"] == 2].iloc[0]
    assert row1["low"] == pytest.approx(2.0 * 0.2, rel=1e-6)
    assert row1["high"] == pytest.approx(2.0 * 0.4, rel=1e-6)
    assert row2["low"] == pytest.approx(4.0 * 0.2, rel=1e-6)
    assert row2["high"] == pytest.approx(4.0 * 0.4, rel=1e-6)
