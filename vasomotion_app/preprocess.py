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
NORMALIZE_METHODS = ("none", "dx_x0_median", "dx_x0_mean", "zscore", "dx_x0_rolling")
DETREND_METHODS = ("none", "constant", "linear")

# Default length of the rolling-baseline window, in seconds, for
# normalize='dx_x0_rolling'. 200 s is 1 / 0.005 Hz -- the period of the
# default ultra_low band's lower edge -- because a rolling baseline shorter
# than the longest period you want to measure subtracts away the very
# fluctuation you are trying to measure (see `rolling_baseline` and
# `validate_rolling_window`). It is a starting point matched to the default
# bands, not a universally right number.
DEFAULT_ROLLING_WINDOW_SEC = 200.0

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


def rolling_baseline(values: np.ndarray, fs: float, window_seconds: float) -> np.ndarray:
    """Centred rolling median of `values`, used as a time-varying baseline.

    The window is `window_seconds` long, converted to an odd sample count so
    it can be centred without introducing a half-sample lag. Near the two
    ends, where a full centred window does not exist, the window shrinks to
    the available samples (pandas' `min_periods=1`) rather than returning
    NaN: a NaN baseline would propagate into the normalised trace and knock
    out the first and last ~100 s of a 5-minute recording, which is a large
    fraction of the data to lose to an edge convention.

    A median rather than a mean, matching `normalize_signal`'s dx_x0_median:
    it is insensitive to the brief dilation spikes that are the signal of
    interest, so the baseline tracks the vessel's resting calibre instead of
    being dragged upward by the events themselves.
    """
    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
    x = np.asarray(values, dtype=float)
    n_window = int(round(window_seconds * fs))
    n_window = max(n_window, 1)
    if n_window % 2 == 0:
        n_window += 1
    return (
        pd.Series(x)
        .rolling(window=n_window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )


def validate_rolling_window(
    window_seconds: float, bands: dict | None, n_samples: int, fs: float
) -> list[str]:
    """Warn when a rolling-baseline window would corrupt the bands being
    measured, or is longer than the recording.

    This is the whole reason a rolling baseline is not simply better than a
    global one. Subtracting a rolling median high-passes the trace at
    roughly 1 / window_seconds: fluctuations slower than the window are
    absorbed into the "baseline" and removed. So a baseline window must be
    LONGER than the longest period you intend to measure, i.e.
    window_seconds >= 1 / lowest_band_edge. With the default ultra_low band
    starting at 0.005 Hz that means >= 200 s, which on a 300 s recording is
    already two-thirds of the whole trace -- at which point a "local"
    baseline is barely distinguishable from the global one.

    That is not a limitation of this implementation; it is what "local
    baseline" and "slow fluctuation" mean at the same timescale. Measured on
    a real 5-minute workbook, normalising each 65.5 s analysis window by its
    own median cut ultra-low band power to a median of 0.49x (worst case
    0.11x) of its global-baseline value, while the myogenic band moved only
    4.5%.

    Returns a list of human-readable warnings; empty means the window is
    safe for the requested bands.
    """
    warnings: list[str] = []
    if window_seconds <= 0:
        return [f"Rolling baseline window must be > 0 s, got {window_seconds}."]

    duration_sec = n_samples / fs if fs > 0 else float("inf")
    if window_seconds >= duration_sec:
        warnings.append(
            f"Rolling baseline window ({window_seconds:g} s) is at least as long as the "
            f"recording ({duration_sec:.1f} s), so the rolling median is effectively the "
            "global median and this behaves the same as dx_x0_median."
        )

    if bands:
        lowest_edge = min(float(low) for low, _high in bands.values())
        if lowest_edge > 0:
            required_sec = 1.0 / lowest_edge
            if window_seconds < required_sec:
                warnings.append(
                    f"Rolling baseline window ({window_seconds:g} s) is shorter than the "
                    f"longest period you are measuring ({required_sec:.0f} s, from the "
                    f"{lowest_edge:g} Hz band edge). Subtracting a rolling median removes "
                    f"fluctuations slower than the window, so power below "
                    f"{1.0 / window_seconds:.4f} Hz will be suppressed -- the baseline "
                    "absorbs the signal you are trying to measure. Use a window of at "
                    f"least {required_sec:.0f} s, or use dx_x0_median instead."
                )
    return warnings


def normalize_signal(
    values: np.ndarray,
    method: str,
    fs: float | None = None,
    rolling_window_seconds: float = DEFAULT_ROLLING_WINDOW_SEC,
) -> np.ndarray:
    """Normalise a trace. 'dx_x0_median'/'dx_x0_mean' compute (x - x0) / x0
    with x0 the trace median/mean; 'zscore' computes (x - mean) / std;
    'dx_x0_rolling' computes (x - m(t)) / m(t) against a centred rolling
    median m(t), i.e. a local rather than a whole-trace baseline.

    `fs` is required only for 'dx_x0_rolling' (the window is specified in
    seconds, so samples cannot be derived without it) and is ignored by
    every other method, which keeps the existing three-argument call sites
    working unchanged.
    """
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
    if method == "dx_x0_rolling":
        if fs is None:
            raise ValueError("fs is required for normalize method 'dx_x0_rolling'")
        baseline = rolling_baseline(x, fs, rolling_window_seconds)
        return (x - baseline) / baseline
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


def validate_highpass_cutoff(cutoff_hz: float | None, fs: float) -> str | None:
    """Return a human-readable warning if `cutoff_hz` would make
    `highpass_filter` above a silent no-op (None, <= 0, or at/above the
    Nyquist frequency fs/2), else None.

    Mirrors `highpass_filter`'s own no-op condition exactly -- the same
    `cutoff_hz <= 0 or cutoff_hz >= fs / 2` test, not a separately
    maintained copy of it -- so a caller can warn about, and refuse to
    record as "applied", a cutoff that would not actually filter
    anything, rather than relying on highpass_filter's own silent
    pass-through to ever surface the mistake. At fs=2 Hz, a cutoff of 1.5
    Hz is exactly this case: >= the 1.0 Hz Nyquist frequency, so
    highpass_filter passes the trace through unchanged, but nothing about
    that call raises or warns on its own.
    """
    if cutoff_hz is None:
        return None
    nyquist = fs / 2
    if cutoff_hz <= 0 or cutoff_hz >= nyquist:
        return (
            f"High-pass cutoff {cutoff_hz} Hz has no effect at fs={fs} Hz: it must be "
            f"> 0 and < the Nyquist frequency (fs/2 = {nyquist:.4f} Hz). The filter will "
            "not be applied, and this run's settings will not record a cutoff."
        )
    return None


def preprocess_stages(
    values: np.ndarray,
    fs: float,
    signal_type: str,
    normalize: str = "dx_x0_median",
    detrend_method: str = "linear",
    highpass_cutoff: float | None = 0.005,
    highpass_order: int = 4,
    trim_seconds: float = 0.0,
    rolling_window_seconds: float = DEFAULT_ROLLING_WINDOW_SEC,
) -> dict[str, np.ndarray]:
    """Every intermediate trace of the Stage 1 chain, keyed by stage name:
    'trimmed', 'converted', 'normalized', 'filtered', 'analyzed'.

    This is the single definition of the stage order -- `preprocess_signal`
    returns this dict's 'analyzed' entry rather than repeating the chain, so
    a caller wanting a statistic from a mid-chain trace (see
    `trace_stats_table`) can never drift out of step with what the PSD was
    actually computed from.
    """
    trimmed = trim_signal(values, fs, trim_seconds)
    converted = area_to_diameter(trimmed, signal_type)
    normalized = normalize_signal(
        converted, normalize, fs=fs, rolling_window_seconds=rolling_window_seconds
    )
    filtered = highpass_filter(normalized, fs, highpass_cutoff, highpass_order)
    analyzed = detrend_signal(filtered, detrend_method)
    return {
        "trimmed": trimmed,
        "converted": converted,
        "normalized": normalized,
        "filtered": filtered,
        "analyzed": analyzed,
    }


def preprocess_signal(
    values: np.ndarray,
    fs: float,
    signal_type: str,
    normalize: str = "dx_x0_median",
    detrend_method: str = "linear",
    highpass_cutoff: float | None = 0.005,
    highpass_order: int = 4,
    trim_seconds: float = 0.0,
    rolling_window_seconds: float = DEFAULT_ROLLING_WINDOW_SEC,
) -> np.ndarray:
    """Apply the full Stage 1 pipeline to one animal's raw trace, in order:
    trim, signal-type conversion, normalise, high-pass, detrend."""
    return preprocess_stages(
        values, fs=fs, signal_type=signal_type, normalize=normalize,
        detrend_method=detrend_method, highpass_cutoff=highpass_cutoff,
        highpass_order=highpass_order, trim_seconds=trim_seconds,
        rolling_window_seconds=rolling_window_seconds,
    )["analyzed"]


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
    rolling_window_seconds: float = DEFAULT_ROLLING_WINDOW_SEC,
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
            rolling_window_seconds=rolling_window_seconds,
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


# Column order of trace_stats_table's output. Kept as a module constant so a
# test can assert the exported table's shape without restating the list.
TRACE_STATS_COLUMNS = [
    "animal", "sex", "region", "condition",
    "n_samples", "duration_sec",
    "mean_raw", "sd_raw", "sd_normalized", "sd_analyzed",
]


def trace_stats_table(
    df: pd.DataFrame,
    fs: float,
    signal_type: str,
    normalize: str = "dx_x0_median",
    detrend_method: str = "linear",
    highpass_cutoff: float | None = 0.005,
    highpass_order: int = 4,
    trim_seconds: float = 0.0,
    rolling_window_seconds: float = DEFAULT_ROLLING_WINDOW_SEC,
) -> pd.DataFrame:
    """Per-trace amplitude statistics, one row per (animal, sex, region,
    condition) group, for export alongside the PSD and band tables.

    Every value comes from `preprocess_stages` under the same settings the
    PSD run used, so these numbers describe exactly the traces that produced
    the spectra -- not a separately recomputed chain.

    Two different standard deviations are reported, because "SD after
    normalisation" is ambiguous and the two answers differ substantially:

      sd_normalized  SD straight after the normalise step, BEFORE the
                     high-pass filter and detrend. This is the variability
                     of the normalised trace as recorded, slow drift
                     included -- usually what is meant by "how much did the
                     vessel fluctuate".
      sd_analyzed    SD of the trace the PSD was actually computed from,
                     i.e. after high-pass and detrend. Always the smaller of
                     the two, by however much those two steps removed: on
                     the real sample workbook at the default 0.005 Hz
                     cutoff the gap is modest (median ratio ~1.2x, because
                     a 0.005 Hz high-pass discards very little), but a
                     higher cutoff widens it sharply. It is also
                     near-redundant with total band power: for a mean-zero
                     trace, variance is the integral of the PSD (measured
                     agreement on that workbook: integral / sd_analyzed^2
                     ~= 0.94, the residual being the taper and
                     segment-averaging normalisation of the estimator
                     rather than a discrepancy in either number).

    Caveats worth knowing before these columns go into a statistics package:

    * Units follow `signal_type`. With signal_type='area' the chain converts
      to diameter (square root) BEFORE normalising, so `mean_raw`/`sd_raw`
      are in diameter units, not area, and the SD columns describe a
      normalised diameter trace. With 'diameter' no conversion happens.
    * `mean_raw`/`sd_raw` are taken after trim and conversion but before
      normalisation, so they carry the absolute scale that the
      dimensionless dx/x0 columns divided out.
    * With normalize='zscore', `sd_normalized` is 1.0 by construction (the
      z-score sets it), so it carries no information about that trace.
    * All SDs use ddof=1 (sample SD), matching normalize_signal's z-score.
    """
    out_rows = []
    group_cols = ["animal", "sex", "region", "condition"]
    for keys, group in df.sort_values("sample").groupby(group_cols, sort=False):
        animal, sex, region, condition = keys
        stages = preprocess_stages(
            group["value"].to_numpy(),
            fs=fs,
            signal_type=signal_type,
            normalize=normalize,
            detrend_method=detrend_method,
            highpass_cutoff=highpass_cutoff,
            highpass_order=highpass_order,
            trim_seconds=trim_seconds,
            rolling_window_seconds=rolling_window_seconds,
        )
        converted = stages["converted"]
        out_rows.append(
            {
                "animal": animal,
                "sex": sex,
                "region": region,
                "condition": condition,
                "n_samples": int(len(converted)),
                "duration_sec": float(len(converted) / fs),
                "mean_raw": float(np.mean(converted)),
                "sd_raw": float(np.std(converted, ddof=1)),
                "sd_normalized": float(np.std(stages["normalized"], ddof=1)),
                "sd_analyzed": float(np.std(stages["analyzed"], ddof=1)),
            }
        )
    return pd.DataFrame(out_rows, columns=TRACE_STATS_COLUMNS)
