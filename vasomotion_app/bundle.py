"""Assemble tables, figures, and parameters.json into a downloadable zip.
Also the home of the three built-in presets and the parameters.json
read/write functions that make every exported result reproducible.

This is the orchestration layer: `run_pipeline` chains preprocess.py ->
psd_methods.py -> bands.py -> figures.py in the fixed order the spec
requires, and does not duplicate any of their internal logic (validation,
NaN handling, filtering) -- it only wires their outputs together and lets
their own errors and edge-case values (e.g. an out-of-range band's NaN)
pass through unchanged.

Not wired here: `bands.validate_band_edges`. Its warnings are advisory
(the data is not wrong; a band's lower edge is just below the resolvable
minimum for the chosen window) rather than a pipeline error or a value
that belongs in a results table. Per the master plan
(docs/superpowers/plans/2026-09-14-vasomotion-psd-tool.md, Task 8's app.py
reference code, ~line 2377), the Streamlit layer calls
`bands_module.validate_band_edges(band_defs, nperseg, fs)` directly, right
before calling `run_pipeline`, and surfaces each warning with
`st.warning(...)`. Task 7's end-to-end test also locks `run_pipeline`'s
return dict to exactly {"full_psd", "band_summary", "figures"}, so adding a
fourth "warnings" key here would break that contract. See task-6-report.md.
"""
from __future__ import annotations

import io
import json
import zipfile

import matplotlib.figure
import pandas as pd

from vasomotion_app import bands as bands_module
from vasomotion_app import figures as figures_module
from vasomotion_app import preprocess as preprocess_module
from vasomotion_app import psd_methods as psd_methods_module
from vasomotion_app.psd_methods import PSDParams

APP_VERSION = "1.0.0"

# Shared by all three presets. Deliberately NOT copied into build_parameters_dict's
# defaults -- there are none there -- this dict only seeds PRESETS below.
_DEFAULT_BANDS = {"ultra_low": [0.005, 0.04], "myogenic": [0.04, 0.15]}

# Exactly three presets, per spec. None of them sets "signal_type": applying
# the area-to-diameter square root to already-converted diameter data is
# silently wrong, so that control deliberately has no default anywhere,
# including here.
PRESETS: dict[str, dict] = {
    "Gordon lab default": {
        "normalize": "dx_x0_median",
        "detrend": "linear",
        "highpass_cutoff": 0.005,
        "highpass_order": 4,
        "trim_seconds": 0.0,
        "psd": {
            "method": "multitaper_welch", "nperseg": 256, "noverlap": 128,
            "window": "hann", "nw": 3.0, "k": 5, "zero_pad": 1, "scaling": "density",
        },
        "max_freq": 0.5,
        "xscale": "log",
        "bands": dict(_DEFAULT_BANDS),
    },
    "Raw periodogram": {
        "normalize": "dx_x0_median",
        "detrend": "linear",
        "highpass_cutoff": 0.005,
        "highpass_order": 4,
        "trim_seconds": 0.0,
        "psd": {
            "method": "periodogram", "nperseg": None, "noverlap": None,
            "window": "boxcar", "nw": 3.0, "k": 5, "zero_pad": 1, "scaling": "density",
        },
        "max_freq": 0.5,
        "xscale": "log",
        "bands": dict(_DEFAULT_BANDS),
    },
    "Long window": {
        "normalize": "dx_x0_median",
        "detrend": "linear",
        "highpass_cutoff": 0.005,
        "highpass_order": 4,
        "trim_seconds": 0.0,
        "psd": {
            "method": "multitaper_welch", "nperseg": 512, "noverlap": 256,
            "window": "hann", "nw": 3.0, "k": 5, "zero_pad": 1, "scaling": "density",
        },
        "max_freq": 0.5,
        "xscale": "log",
        "bands": dict(_DEFAULT_BANDS),
    },
}


def build_parameters_dict(
    signal_type: str | None,
    normalize: str,
    detrend_method: str,
    highpass_cutoff: float | None,
    highpass_order: int,
    trim_seconds: float,
    psd_params: PSDParams,
    max_freq: float,
    xscale: str,
    bands: dict[str, tuple[float, float]],
    fs: float,
) -> dict:
    """Assemble the full settings dict written as parameters.json. This dict
    is both the settings export and the reload format: every field here is
    exactly what `psd_params_from_dict` / a manual reload needs, and nothing
    is dropped by round-tripping through JSON.

    Band edges are converted to lists (`list(edges)`, not left as tuples):
    JSON has no tuple type, so `parameters_from_json` always returns lists
    for them, and a tuple here would make an otherwise-correct round trip
    compare unequal to itself (`(0.005, 0.04) != [0.005, 0.04]` in Python).
    """
    return {
        "app_version": APP_VERSION,
        "fs": fs,
        "signal_type": signal_type,
        "normalize": normalize,
        "detrend": detrend_method,
        "highpass_cutoff": highpass_cutoff,
        "highpass_order": highpass_order,
        "trim_seconds": trim_seconds,
        "psd": {
            "method": psd_params.method,
            "nperseg": psd_params.nperseg,
            "noverlap": psd_params.noverlap,
            "window": psd_params.window,
            "nw": psd_params.nw,
            "k": psd_params.k,
            "zero_pad": psd_params.zero_pad,
            "scaling": psd_params.scaling,
        },
        "max_freq": max_freq,
        "xscale": xscale,
        "bands": {name: list(edges) for name, edges in bands.items()},
    }


def parameters_to_json(params_dict: dict) -> str:
    """Serialise a parameters dict (as built by build_parameters_dict) to a
    JSON string. This is the exact text written as parameters.json."""
    return json.dumps(params_dict, indent=2, sort_keys=False)


def parameters_from_json(json_str: str) -> dict:
    """Parse a parameters.json string back into a plain dict. Paired with
    parameters_to_json so a round trip is just `parameters_from_json(
    parameters_to_json(d)) == d`, given d's leaf values are JSON-safe
    (band edges as lists, no NaN/tuple values)."""
    return json.loads(json_str)


def psd_params_from_dict(psd_dict: dict) -> PSDParams:
    """Reconstruct a PSDParams from the "psd" sub-dict of a parameters dict
    (or a preset's "psd" entry). Every field is read explicitly -- none is
    left to PSDParams' own dataclass defaults -- so a dict that specifies
    every field always reproduces exactly that PSDParams, never a mix of
    the dict's values and leftover defaults."""
    return PSDParams(
        method=psd_dict["method"],
        nperseg=psd_dict["nperseg"],
        noverlap=psd_dict["noverlap"],
        window=psd_dict["window"],
        nw=psd_dict["nw"],
        k=psd_dict["k"],
        zero_pad=psd_dict["zero_pad"],
        scaling=psd_dict["scaling"],
    )


def run_pipeline(
    long_df: pd.DataFrame,
    fs: float,
    signal_type: str,
    normalize: str,
    detrend_method: str,
    highpass_cutoff: float | None,
    highpass_order: int,
    trim_seconds: float,
    psd_params: PSDParams,
    band_defs: dict[str, tuple[float, float]],
    max_freq: float,
) -> dict:
    """Run the full Stage 1 -> Stage 2 -> Stage 3 pipeline on a tidy long
    dataframe (as produced by io_excel.load_workbook_long).

    Returns a dict with keys:
      'full_psd': tidy long PSD dataframe (frequency <= max_freq only)
      'band_summary': one row per animal/region/condition, one column per
          band. bands.integrate_bands (called as-is, with no wrapping
          try/except) already raises ValueError for an invalid band name
          via validate_band_names, and already returns NaN -- never 0.0 --
          for a band whose edges fall entirely outside full_psd's frequency
          range. Neither behaviour is touched, re-caught, or reinterpreted
          here.
      'figures': {'group_average': {region: Figure}, 'per_animal': {region: Figure}}

    `bands.validate_band_edges` is intentionally not called here -- see this
    module's docstring for why and where it is wired instead.
    """
    preprocessed = preprocess_module.preprocess_long_dataframe(
        long_df, fs=fs, signal_type=signal_type, normalize=normalize,
        detrend_method=detrend_method, highpass_cutoff=highpass_cutoff,
        highpass_order=highpass_order, trim_seconds=trim_seconds,
    )
    full_psd = psd_methods_module.compute_psd_table(preprocessed, fs, psd_params)
    full_psd = full_psd[full_psd["frequency"] <= max_freq].reset_index(drop=True)
    band_summary = bands_module.integrate_bands(full_psd, band_defs)

    group_average_figs = {}
    per_animal_figs = {}
    for region in sorted(full_psd["region"].unique()):
        group_average_figs[region] = figures_module.plot_group_average(
            full_psd, region, max_freq=max_freq
        )
        per_animal_figs[region] = figures_module.plot_per_animal_small_multiples(
            full_psd, region, max_freq=max_freq
        )

    return {
        "full_psd": full_psd,
        "band_summary": band_summary,
        "figures": {"group_average": group_average_figs, "per_animal": per_animal_figs},
    }


def build_results_zip(
    band_summary: pd.DataFrame,
    full_psd: pd.DataFrame,
    figures_by_name: dict[str, matplotlib.figure.Figure],
    parameters: dict,
) -> bytes:
    """Bundle the band summary CSV, full PSD CSV, every figure as a PNG, and
    parameters.json into one zip file. Returns the zip file's raw bytes.

    CSV serialisation uses pandas' own `to_csv`/`read_csv` defaults, which
    already round-trip NaN correctly (an empty field, read back as NaN) --
    nothing here fills, rounds, or drops a NaN band value.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("band_summary.csv", band_summary.to_csv(index=False))
        zf.writestr("full_psd.csv", full_psd.to_csv(index=False))
        for name, fig in figures_by_name.items():
            fig_buffer = io.BytesIO()
            fig.savefig(fig_buffer, format="png", dpi=150)
            zf.writestr(f"{name}.png", fig_buffer.getvalue())
        zf.writestr("parameters.json", parameters_to_json(parameters))
    return buffer.getvalue()
