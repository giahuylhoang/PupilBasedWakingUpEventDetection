"""Stage 2: PSD estimation. Four methods behind one interface:
compute_psd(signal, fs, params) -> (freqs, power).

Every method passes detrend=False to scipy: detrending, if wanted, already
happened in Stage 1 (preprocess.py), and scipy's own default ('constant')
would otherwise silently detrend a second time.

Zero-padding is expressed as an integer multiplier of the window length (the
segment length for Welch-family methods, the whole trace for periodogram and
multitaper Thomson) per the spec, rather than scipy_pub_arteriole_psd.py's
next-power-of-2 rounding -- 1x means no padding at all.

All four methods reduce to the same shape: evaluate a one-window (or
one-taper) spectrum estimator once per taper in a bank, then average the
resulting power across tapers. A plain periodogram is that average with a
one-row bank (the boxcar window); a plain Welch estimate is the same with a
one-row bank (the named window, e.g. hann); the two multitaper methods are
that average with a K-row DPSS bank. `_tapered_average` is the one shared
code path -- the four methods differ only in which spectrum function
(periodogram vs. welch) and which taper bank they hand it, not in how the
averaging itself works.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.signal import periodogram, welch
from scipy.signal.windows import dpss, get_window

VALID_METHODS = ("periodogram", "welch", "multitaper_welch", "multitaper_thomson")
VALID_WINDOWS = ("hann", "hamming", "blackman", "boxcar")
VALID_SCALINGS = ("density", "spectrum")


@dataclass
class PSDParams:
    method: str = "multitaper_welch"
    nperseg: int | None = 256
    noverlap: int | None = 128
    window: str = "hann"
    nw: float = 3.0
    k: int = 5
    zero_pad: int = 1
    scaling: str = "density"


def validate_k_nw(nw: float, k: int) -> None:
    """K must satisfy 1 <= K <= 2*NW - 1 (the number of DPSS tapers with
    concentration ratio near 1 for a given time-half-bandwidth product)."""
    k_max = int(2 * nw - 1)
    if k > k_max:
        raise ValueError(f"K={k} exceeds the maximum {k_max} for NW={nw} (K must be <= 2*NW - 1)")
    if k < 1:
        raise ValueError(f"K must be >= 1, got {k}")


def validate_window_length(window_length: int, n_samples: int) -> None:
    """Window length cannot exceed the trace length."""
    if window_length > n_samples:
        raise ValueError(f"window length {window_length} exceeds trace length {n_samples}")


def window_summary(n_samples: int, fs: float, window_length: int, noverlap: int) -> dict:
    """Live readout for the sidebar: frequency resolution and number of
    Welch segments available to average, for a given window length."""
    resolution_hz = fs / window_length
    step = window_length - noverlap
    if window_length > n_samples or step <= 0:
        n_segments = 0
    else:
        n_segments = 1 + (n_samples - window_length) // step
    return {"resolution_hz": resolution_hz, "n_segments": int(n_segments)}


def _nfft(window_length: int, zero_pad: int) -> int:
    return int(window_length * zero_pad)


def _tapered_average(
    spectrum_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    tapers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Call spectrum_fn(taper) -> (freqs, power) once per row of `tapers`
    and average the resulting power across rows.

    This is the single code path shared by all four PSD methods: a one-row
    bank (a named window or boxcar) makes it a plain single-window estimate;
    a K-row DPSS bank makes it a multitaper eigenspectrum average. Averaging
    a one-row bank is a numerical no-op (mean of one row returns that row
    exactly), so plain periodogram/Welch results are unaffected by going
    through this shared path.
    """
    eigenspectra = []
    freqs = None
    for taper in tapers:
        freqs, power = spectrum_fn(taper)
        eigenspectra.append(power)
    return freqs, np.mean(np.vstack(eigenspectra), axis=0)


def compute_psd(signal: np.ndarray, fs: float, params: PSDParams) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the power spectral density of `signal` sampled at `fs` Hz
    using the method and sub-parameters in `params`. Returns (freqs, power),
    one-sided, in ascending frequency order."""
    x = np.asarray(signal, dtype=float)
    n = len(x)

    if params.method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {params.method!r}")
    if params.scaling not in VALID_SCALINGS:
        raise ValueError(f"scaling must be one of {VALID_SCALINGS}, got {params.scaling!r}")

    if params.method in ("periodogram", "multitaper_thomson"):
        nfft = _nfft(n, params.zero_pad)
        if params.method == "multitaper_thomson":
            validate_k_nw(params.nw, params.k)
            tapers = dpss(n, NW=params.nw, Kmax=params.k)
        else:
            tapers = get_window("boxcar", n)[np.newaxis, :]

        def spectrum_fn(taper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return periodogram(
                x, fs=fs, window=taper, nfft=nfft, scaling=params.scaling,
                detrend=False, return_onesided=True,
            )

        return _tapered_average(spectrum_fn, tapers)

    # Welch-family: 'welch' and 'multitaper_welch' both segment the trace
    # via nperseg/noverlap before spectral averaging.
    nperseg = int(params.nperseg)
    noverlap = int(params.noverlap)
    validate_window_length(nperseg, n)
    nfft = _nfft(nperseg, params.zero_pad)

    if params.method == "multitaper_welch":
        validate_k_nw(params.nw, params.k)
        tapers = dpss(nperseg, NW=params.nw, Kmax=params.k)
    else:
        if params.window not in VALID_WINDOWS:
            raise ValueError(f"window must be one of {VALID_WINDOWS}, got {params.window!r}")
        tapers = get_window(params.window, nperseg)[np.newaxis, :]

    def spectrum_fn(taper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return welch(
            x, fs=fs, window=taper, nperseg=nperseg, noverlap=noverlap,
            nfft=nfft, scaling=params.scaling, detrend=False, return_onesided=True,
        )

    return _tapered_average(spectrum_fn, tapers)


def compute_psd_table(long_df: pd.DataFrame, fs: float, params: PSDParams) -> pd.DataFrame:
    """Run compute_psd on every (animal, region, condition) group of an
    already-preprocessed tidy long dataframe (as produced by
    preprocess.preprocess_long_dataframe).

    Returns a tidy long dataframe with columns: animal, sex, region,
    condition, frequency, power.
    """
    out_rows = []
    group_cols = ["animal", "sex", "region", "condition"]
    for keys, group in long_df.sort_values("sample").groupby(group_cols, sort=False):
        animal, sex, region, condition = keys
        freqs, power = compute_psd(group["value"].to_numpy(), fs, params)
        for f_val, p_val in zip(freqs, power):
            out_rows.append(
                {
                    "animal": animal,
                    "sex": sex,
                    "region": region,
                    "condition": condition,
                    "frequency": float(f_val),
                    "power": float(p_val),
                }
            )
    return pd.DataFrame(
        out_rows, columns=["animal", "sex", "region", "condition", "frequency", "power"]
    )
