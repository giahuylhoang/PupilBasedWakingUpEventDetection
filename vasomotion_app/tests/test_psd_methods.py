"""Tests for vasomotion_app.psd_methods."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vasomotion_app.psd_methods import (
    PSDParams,
    compute_psd,
    compute_psd_table,
    validate_k_nw,
    validate_window_length,
    window_summary,
)


# Shared synthetic signal: fs=10 Hz, f0=1.0 Hz lands exactly on a frequency
# bin both for the whole 2560-sample trace (df=fs/N=0.00390625, f0/df=256)
# and for a 200-sample Welch segment (df=fs/200=0.05, f0/df=20), so there is
# no spectral leakage and the peak power is deterministic to float precision.
FS = 10.0
F0 = 1.0
AMPLITUDE = 2.0
N = 2560


def _sine_signal():
    t = np.arange(N) / FS
    return AMPLITUDE * np.sin(2 * np.pi * F0 * t)


def test_compute_psd_periodogram_peak():
    signal = _sine_signal()
    params = PSDParams(method="periodogram", zero_pad=1, scaling="density")
    freqs, power = compute_psd(signal, FS, params)
    assert freqs.shape == (1281,)
    idx = int(np.argmax(power))
    assert freqs[idx] == pytest.approx(1.0)
    assert power[idx] == pytest.approx(511.9999999999826, rel=1e-4)


def test_compute_psd_multitaper_thomson_peak():
    signal = _sine_signal()
    params = PSDParams(method="multitaper_thomson", nw=3.0, k=5, zero_pad=1, scaling="density")
    freqs, power = compute_psd(signal, FS, params)
    assert freqs.shape == (1281,)
    idx = int(np.argmax(power))
    assert freqs[idx] == pytest.approx(1.0)
    assert power[idx] == pytest.approx(98.74183972652611, rel=1e-4)


def test_compute_psd_welch_peak():
    signal = _sine_signal()
    params = PSDParams(method="welch", nperseg=200, noverlap=100, window="hann", zero_pad=1, scaling="density")
    freqs, power = compute_psd(signal, FS, params)
    assert freqs.shape == (101,)
    idx = int(np.argmax(power))
    assert freqs[idx] == pytest.approx(1.0)
    assert power[idx] == pytest.approx(26.666666666665805, rel=1e-4)


def test_compute_psd_multitaper_welch_peak():
    signal = _sine_signal()
    params = PSDParams(
        method="multitaper_welch", nperseg=200, noverlap=100, nw=3.0, k=5,
        zero_pad=1, scaling="density",
    )
    freqs, power = compute_psd(signal, FS, params)
    assert freqs.shape == (101,)
    idx = int(np.argmax(power))
    assert freqs[idx] == pytest.approx(1.0)
    assert power[idx] == pytest.approx(7.717106260491967, rel=1e-4)


def test_compute_psd_zero_padding_doubles_grid():
    signal = _sine_signal()
    params = PSDParams(method="periodogram", zero_pad=2, scaling="density")
    freqs, _power = compute_psd(signal, FS, params)
    assert freqs.shape == (2561,)


def test_compute_psd_rejects_unknown_method():
    with pytest.raises(ValueError, match="method must be one of"):
        compute_psd(np.zeros(10), 10.0, PSDParams(method="bogus"))


def test_compute_psd_rejects_unknown_scaling():
    # Distinct validation rule from the method check above: scaling is
    # checked independently, so a typo'd scaling must be caught even when
    # method is otherwise valid.
    with pytest.raises(ValueError, match="scaling must be one of"):
        compute_psd(np.zeros(10), 10.0, PSDParams(method="periodogram", scaling="bogus"))


def test_compute_psd_welch_raises_when_window_exceeds_trace():
    signal = np.zeros(50)
    params = PSDParams(method="welch", nperseg=100, noverlap=50, window="hann")
    with pytest.raises(ValueError, match="exceeds trace length"):
        compute_psd(signal, 10.0, params)


def test_compute_psd_welch_rejects_unknown_window():
    # VALID_WINDOWS is a business-rule allowlist narrower than everything
    # scipy.signal.windows.get_window would accept (e.g. "flattop" is a real
    # scipy window but not one of ours) -- this only fires if compute_psd
    # itself enforces the allowlist for the 'welch' method.
    signal = _sine_signal()
    params = PSDParams(method="welch", nperseg=200, noverlap=100, window="bogus")
    with pytest.raises(ValueError, match="window must be one of"):
        compute_psd(signal, FS, params)


def test_compute_psd_multitaper_thomson_rejects_k_above_cap():
    # validate_k_nw is unit-tested below in isolation, but nothing proves
    # compute_psd actually calls it for this method -- this exercises the
    # wiring, not just the standalone function.
    signal = _sine_signal()
    params = PSDParams(method="multitaper_thomson", nw=3.0, k=6)
    with pytest.raises(ValueError, match="exceeds the maximum 5"):
        compute_psd(signal, FS, params)


def test_compute_psd_multitaper_welch_rejects_k_above_cap():
    signal = _sine_signal()
    params = PSDParams(method="multitaper_welch", nperseg=200, noverlap=100, nw=3.0, k=6)
    with pytest.raises(ValueError, match="exceeds the maximum 5"):
        compute_psd(signal, FS, params)


def test_validate_k_nw_accepts_boundary():
    validate_k_nw(nw=3.0, k=5)  # 2*3-1 = 5, exactly at the cap: must not raise


def test_validate_k_nw_rejects_above_cap():
    with pytest.raises(ValueError, match="exceeds the maximum 5"):
        validate_k_nw(nw=3.0, k=6)


def test_validate_k_nw_rejects_k_below_one():
    with pytest.raises(ValueError, match="K must be >= 1"):
        validate_k_nw(nw=3.0, k=0)


def test_validate_window_length_raises_when_too_long():
    with pytest.raises(ValueError, match="exceeds trace length"):
        validate_window_length(300, 200)


def test_validate_window_length_ok_when_equal():
    validate_window_length(200, 200)  # must not raise


def test_window_summary():
    result = window_summary(n_samples=1000, fs=10.0, window_length=200, noverlap=100)
    assert result["resolution_hz"] == pytest.approx(0.05)
    assert result["n_segments"] == 9  # 1 + (1000-200)//100


def test_window_summary_zero_segments_when_window_too_long():
    result = window_summary(n_samples=100, fs=10.0, window_length=200, noverlap=100)
    assert result["n_segments"] == 0


def test_window_summary_zero_segments_when_noverlap_equals_window_length():
    # Distinct branch from "window too long": here window_length <=
    # n_samples, but the step size (window_length - noverlap) is zero, which
    # must also be reported as zero usable segments rather than dividing by
    # zero or looping forever.
    result = window_summary(n_samples=1000, fs=10.0, window_length=200, noverlap=200)
    assert result["n_segments"] == 0


def test_compute_psd_table_groups_correctly():
    fs = 10.0
    n = 100
    t = np.arange(n) / fs
    df = pd.DataFrame(
        {
            "animal": [1] * n + [2] * n,
            "sex": ["M"] * n + ["F"] * n,
            "region": ["R"] * (2 * n),
            "condition": ["C"] * (2 * n),
            "sample": list(range(1, n + 1)) * 2,
            "value": list(np.sin(2 * np.pi * 1.0 * t)) + list(np.sin(2 * np.pi * 2.0 * t)),
        }
    )
    params = PSDParams(method="periodogram", zero_pad=1, scaling="density")
    table = compute_psd_table(df, fs, params)
    assert set(table.columns) == {"animal", "sex", "region", "condition", "frequency", "power"}
    assert set(table["animal"].unique()) == {1, 2}
    n_freqs = n // 2 + 1
    assert len(table[table["animal"] == 1]) == n_freqs
    assert len(table[table["animal"] == 2]) == n_freqs
    t1 = table[table["animal"] == 1]
    t2 = table[table["animal"] == 2]
    assert t1.loc[t1["power"].idxmax(), "frequency"] == pytest.approx(1.0, abs=0.05)
    assert t2.loc[t2["power"].idxmax(), "frequency"] == pytest.approx(2.0, abs=0.05)


def test_compute_psd_table_sorts_by_sample_before_computing():
    # Rows deliberately shuffled out of temporal order (same permutation
    # applied to 'sample' and 'value' so each row still pairs the correct
    # value with its sample index). If compute_psd_table fed the group to
    # compute_psd in raw dataframe row order instead of sorting by 'sample'
    # first, the reconstructed trace would be scrambled noise and its
    # spectral peak would not land at 1.0 Hz.
    fs = 10.0
    n = 100
    t = np.arange(n) / fs
    values = np.sin(2 * np.pi * 1.0 * t)
    samples = np.arange(1, n + 1)
    order = np.random.default_rng(0).permutation(n)
    df = pd.DataFrame(
        {
            "animal": [1] * n,
            "sex": ["M"] * n,
            "region": ["R"] * n,
            "condition": ["C"] * n,
            "sample": samples[order],
            "value": values[order],
        }
    )
    params = PSDParams(method="periodogram", zero_pad=1, scaling="density")
    table = compute_psd_table(df, fs, params)
    peak_freq = table.loc[table["power"].idxmax(), "frequency"]
    assert peak_freq == pytest.approx(1.0, abs=0.05)
