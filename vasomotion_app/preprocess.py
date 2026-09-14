"""Stage 1: preprocessing a single animal's raw trace before spectral analysis.

Order of operations (fixed, not user-configurable): trim first -- so that the
very samples trim exists to remove cannot bias any statistic computed later
-- then convert area to diameter if needed, then normalise, then high-pass
filter, then detrend. This mirrors scripts/pub_arteriole_psd.py's normalise
-> high-pass -> detrend order (see its `whole_trace_psd`), with trim
prepended and signal-type conversion folded into the normalise step's input.

Unlike scripts/pub_arteriole_psd.py's highpass_filter, this module does not
interpolate over NaNs: io_excel.load_workbook_long guarantees fully numeric
data (it raises NonNumericDataError otherwise), so by the time a trace
reaches this module there is nothing to interpolate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend as scipy_detrend, sosfiltfilt

SIGNAL_TYPES = ("area", "diameter")
NORMALIZE_METHODS = ("none", "dx_x0_median", "dx_x0_mean", "zscore")
DETREND_METHODS = ("none", "constant", "linear")

MIN_PLAUSIBLE_DURATION_SEC = 10.0
MAX_PLAUSIBLE_DURATION_SEC = 6 * 3600.0


def trim_signal(values: np.ndarray, fs: float, trim_seconds: float) -> np.ndarray:
    """Drop the first `trim_seconds` seconds of samples."""
    if trim_seconds < 0:
        raise ValueError(f"trim_seconds must be >= 0, got {trim_seconds}")
    n_drop = int(round(trim_seconds * fs))
    n_drop = min(n_drop, len(values))
    return np.asarray(values, dtype=float)[n_drop:]


def area_to_diameter(values: np.ndarray, signal_type: str) -> np.ndarray:
    """Convert cross-sectional area to diameter by square root. Passes
    diameter data through unchanged."""
    if signal_type == "area":
        return np.sqrt(np.maximum(np.asarray(values, dtype=float), 0.0))
    if signal_type == "diameter":
        return np.asarray(values, dtype=float)
    raise ValueError(f"signal_type must be one of {SIGNAL_TYPES}, got {signal_type!r}")


def normalize_signal(values: np.ndarray, method: str) -> np.ndarray:
    """Normalise a trace. 'dx_x0_median'/'dx_x0_mean' compute (x - x0) / x0
    with x0 the trace median/mean; 'zscore' computes (x - mean) / std."""
    x = np.asarray(values, dtype=float)
    if method == "none":
        return x
    if method == "dx_x0_median":
        x0 = np.median(x)
        return (x - x0) / x0
    if method == "dx_x0_mean":
        x0 = np.mean(x)
        return (x - x0) / x0
    if method == "zscore":
        return (x - np.mean(x)) / np.std(x, ddof=1)
    raise ValueError(f"normalize method must be one of {NORMALIZE_METHODS}, got {method!r}")


def detrend_signal(values: np.ndarray, method: str) -> np.ndarray:
    """Remove a constant or linear trend, or leave the trace unchanged."""
    x = np.asarray(values, dtype=float)
    if method == "none":
        return x
    if method == "constant":
        return scipy_detrend(x, type="constant")
    if method == "linear":
        return scipy_detrend(x, type="linear")
    raise ValueError(f"detrend method must be one of {DETREND_METHODS}, got {method!r}")


def highpass_filter(
    values: np.ndarray,
    fs: float,
    cutoff_hz: float | None,
    order: int,
) -> np.ndarray:
    """Zero-phase Butterworth high-pass filter. Passes the trace through
    unchanged if cutoff_hz is None, <= 0, or at/above the Nyquist frequency."""
    x = np.asarray(values, dtype=float)
    if cutoff_hz is None or cutoff_hz <= 0 or cutoff_hz >= fs / 2:
        return x
    sos = butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def preprocess_signal(
    values: np.ndarray,
    fs: float,
    signal_type: str,
    normalize: str = "dx_x0_median",
    detrend_method: str = "linear",
    highpass_cutoff: float | None = 0.005,
    highpass_order: int = 4,
    trim_seconds: float = 0.0,
) -> np.ndarray:
    """Apply the full Stage 1 pipeline to one animal's raw trace, in order:
    trim, signal-type conversion, normalise, high-pass, detrend."""
    x = trim_signal(values, fs, trim_seconds)
    x = area_to_diameter(x, signal_type)
    x = normalize_signal(x, normalize)
    x = highpass_filter(x, fs, highpass_cutoff, highpass_order)
    x = detrend_signal(x, detrend_method)
    return x


def trace_summary(n_samples: int, fs: float) -> dict:
    """Live readout for the sidebar: sample count, duration, lowest
    resolvable frequency (1 / duration)."""
    duration_sec = n_samples / fs
    return {
        "n_samples": n_samples,
        "duration_sec": duration_sec,
        "lowest_freq_hz": 1.0 / duration_sec if duration_sec > 0 else float("inf"),
    }


def validate_sampling_rate(fs: float, n_samples: int) -> list[str]:
    """Return a list of human-readable warnings about an implausible sampling
    rate. Raises ValueError if fs is not positive (nothing can be computed).

    Plausibility thresholds (10 s to 6 h) are a heuristic for bench vasomotion
    recordings, not a value from the spec; they exist so a badly wrong `fs`
    (e.g. off by 1000x) produces a visible warning instead of a silently
    wrong frequency axis.
    """
    if fs <= 0:
        raise ValueError(f"Sampling rate must be positive, got {fs}")
    warnings = []
    duration_sec = n_samples / fs
    if duration_sec < MIN_PLAUSIBLE_DURATION_SEC:
        warnings.append(
            f"Implied recording duration is only {duration_sec:.2f} s at fs={fs} Hz "
            f"for {n_samples} samples. Check the sampling rate."
        )
    if duration_sec > MAX_PLAUSIBLE_DURATION_SEC:
        warnings.append(
            f"Implied recording duration is {duration_sec / 3600:.1f} h at fs={fs} Hz "
            f"for {n_samples} samples. Check the sampling rate."
        )
    return warnings


def preprocess_long_dataframe(
    df: pd.DataFrame,
    fs: float,
    signal_type: str,
    normalize: str = "dx_x0_median",
    detrend_method: str = "linear",
    highpass_cutoff: float | None = 0.005,
    highpass_order: int = 4,
    trim_seconds: float = 0.0,
) -> pd.DataFrame:
    """Apply preprocess_signal to every (animal, region, condition) group in a
    tidy long dataframe (as produced by io_excel.load_workbook_long).

    Returns a new tidy long dataframe with the same columns; 'sample' is
    renumbered 1..N per group after trimming shortens the trace.
    """
    out_rows = []
    group_cols = ["animal", "sex", "region", "condition"]
    for keys, group in df.sort_values("sample").groupby(group_cols, sort=False):
        animal, sex, region, condition = keys
        processed = preprocess_signal(
            group["value"].to_numpy(),
            fs=fs,
            signal_type=signal_type,
            normalize=normalize,
            detrend_method=detrend_method,
            highpass_cutoff=highpass_cutoff,
            highpass_order=highpass_order,
            trim_seconds=trim_seconds,
        )
        for i, value in enumerate(processed, start=1):
            out_rows.append(
                {
                    "animal": animal,
                    "sex": sex,
                    "region": region,
                    "condition": condition,
                    "sample": i,
                    "value": float(value),
                }
            )
    return pd.DataFrame(
        out_rows, columns=["animal", "sex", "region", "condition", "sample", "value"]
    )
