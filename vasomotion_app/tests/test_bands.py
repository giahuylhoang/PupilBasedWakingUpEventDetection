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


# --- Finding 1 (CRITICAL): band names must not collide with the reserved
# group-identity output columns. Without this check, a band named e.g.
# "animal" produces a DataFrame with two "animal" columns -- one holding the
# real group identity, one holding a band-power float -- and pandas silently
# keeps only one of them. Each reserved name gets its own test so none of
# the four can regress independently.


def test_validate_band_names_rejects_reserved_column_animal():
    with pytest.raises(ValueError, match="'animal' is reserved"):
        validate_band_names({"animal": (0.1, 0.4)})


def test_validate_band_names_rejects_reserved_column_sex():
    with pytest.raises(ValueError, match="'sex' is reserved"):
        validate_band_names({"sex": (0.1, 0.4)})


def test_validate_band_names_rejects_reserved_column_region():
    with pytest.raises(ValueError, match="'region' is reserved"):
        validate_band_names({"region": (0.1, 0.4)})


def test_validate_band_names_rejects_reserved_column_condition():
    with pytest.raises(ValueError, match="'condition' is reserved"):
        validate_band_names({"condition": (0.1, 0.4)})


def test_integrate_bands_rejects_band_name_colliding_with_group_column():
    # Direct reproduction of the reported defect: integrate_bands must
    # refuse this input rather than silently emitting a duplicate-named
    # "animal" column that clobbers the real group identity.
    table = _flat_psd_table()
    with pytest.raises(ValueError, match="'animal' is reserved"):
        integrate_bands(table, {"animal": (0.1, 0.4)})


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
    # Band names are chosen so insertion order ("zeta" then "alpha") is the
    # reverse of sorted order ("alpha" then "zeta"), so a column-sequence
    # assertion can actually distinguish the two (Finding 4).
    table = pd.concat(
        [_flat_psd_table(), _flat_psd_table().assign(animal=2, power=4.0)],
        ignore_index=True,
    )
    result = integrate_bands(table, {"zeta": (0.0, 0.2), "alpha": (0.6, 1.0)})
    assert list(result.columns) == ["animal", "sex", "region", "condition", "zeta", "alpha"]
    assert len(result) == 2
    row1 = result[result["animal"] == 1].iloc[0]
    row2 = result[result["animal"] == 2].iloc[0]
    assert row1["zeta"] == pytest.approx(2.0 * 0.2, rel=1e-6)
    assert row1["alpha"] == pytest.approx(2.0 * 0.4, rel=1e-6)
    assert row2["zeta"] == pytest.approx(4.0 * 0.2, rel=1e-6)
    assert row2["alpha"] == pytest.approx(4.0 * 0.4, rel=1e-6)


# --- Finding 2 (IMPORTANT): the pre-existing multi-group test varies only
# `animal`, so it cannot detect a groupby key that silently drops `sex`,
# `region`, or `condition`. This fixture holds `animal` fixed at 1 for every
# row and varies each of the other three columns, one at a time, against a
# shared baseline row -- so losing any of them from the groupby key merges
# two groups that must stay separate.


def test_integrate_bands_distinguishes_condition_region_sex_at_fixed_animal():
    baseline = _flat_psd_table()  # animal=1, sex="M", region="R", condition="C", power=2.0
    vary_condition = _flat_psd_table().assign(condition="C2", power=4.0)
    vary_region = _flat_psd_table().assign(region="R2", power=6.0)
    vary_sex = _flat_psd_table().assign(sex="F", power=8.0)
    table = pd.concat(
        [baseline, vary_condition, vary_region, vary_sex], ignore_index=True
    )

    result = integrate_bands(table, {"mid": (0.1, 0.4)})

    assert len(result) == 4
    assert (result["animal"] == 1).all()

    def value_for(sex: str, region: str, condition: str) -> float:
        row = result[
            (result["sex"] == sex)
            & (result["region"] == region)
            & (result["condition"] == condition)
        ]
        assert len(row) == 1
        return row.iloc[0]["mid"]

    assert value_for("M", "R", "C") == pytest.approx(2.0 * 0.3, rel=1e-6)
    assert value_for("M", "R", "C2") == pytest.approx(4.0 * 0.3, rel=1e-6)
    assert value_for("M", "R2", "C") == pytest.approx(6.0 * 0.3, rel=1e-6)
    assert value_for("F", "R", "C") == pytest.approx(8.0 * 0.3, rel=1e-6)


# --- Finding 3 (IMPORTANT): a band lying entirely outside the PSD's
# frequency range selects fewer than 2 points and must resolve to NaN for
# that band only, leaving other bands in the same call unaffected.


def test_integrate_bands_band_entirely_above_range_is_nan():
    table = _flat_psd_table()  # frequency spans 0.0-1.0
    result = integrate_bands(table, {"mid": (0.1, 0.4), "too_high": (1.5, 2.0)})
    assert np.isnan(result.iloc[0]["too_high"])
    assert result.iloc[0]["mid"] == pytest.approx(0.6, rel=1e-6)


def test_integrate_bands_band_entirely_below_range_is_nan():
    table = _flat_psd_table().assign(frequency=lambda d: d["frequency"] + 0.5)  # 0.5-1.5
    result = integrate_bands(table, {"mid": (0.6, 0.9), "too_low": (0.0, 0.3)})
    assert np.isnan(result.iloc[0]["too_low"])
    assert result.iloc[0]["mid"] == pytest.approx(2.0 * 0.3, rel=1e-6)
