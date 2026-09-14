"""Tests for vasomotion_app.bundle.

bundle.py is the orchestration layer: the first module whose correctness
depends on the other five modules actually agreeing with each other, not
just each satisfying its own contract. Two integration risks get dedicated
tests here beyond the task brief's reference set:

1. `integrate_bands` returns NaN (never 0.0) for a band whose edges fall
   entirely outside the PSD's frequency range (Task 4 finding). 0.0 reads
   as a real zero-power measurement; only NaN says "not measured". Tests
   below prove that NaN survives both `run_pipeline`'s band_summary and a
   full CSV round trip through `build_results_zip`.
2. `bands.validate_band_edges` is never called by anything in this module.
   Per the master plan (docs/superpowers/plans/2026-09-14-vasomotion-psd-tool.md,
   Task 8's app.py reference code), it is wired at the Streamlit UI layer,
   called directly before `run_pipeline` -- not inside `run_pipeline` itself,
   and Task 7's end-to-end test locks `run_pipeline`'s return dict to
   exactly {"full_psd", "band_summary", "figures"}. See task-6-report.md for
   the full citation; this file does not call or test validate_band_edges.

Also, per the "Test quality bar": the brief's own `test_psd_params_from_dict_roundtrip`
checks only 3 of PSDParams' 8 fields, using the "Gordon lab default" preset
-- whose psd sub-dict happens to equal every PSDParams dataclass default
field-for-field. A `psd_params_from_dict` that ignored its argument entirely
and returned `PSDParams()` would pass that test. A second round-trip test
below uses values that disagree with every dataclass default to close that
gap.
"""
from __future__ import annotations

import ast
import inspect
import io
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import vasomotion_app.bundle as bundle_module
from vasomotion_app.bundle import (
    PRESETS,
    build_parameters_dict,
    build_results_zip,
    parameters_from_json,
    parameters_to_json,
    psd_params_from_dict,
    run_pipeline,
)
from vasomotion_app.psd_methods import PSDParams


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_presets_have_exactly_three_named_entries():
    assert set(PRESETS.keys()) == {"Gordon lab default", "Raw periodogram", "Long window"}


def test_gordon_lab_default_preset_values():
    preset = PRESETS["Gordon lab default"]
    assert preset["highpass_cutoff"] == 0.005
    assert preset["highpass_order"] == 4
    assert preset["psd"]["method"] == "multitaper_welch"
    assert preset["psd"]["nperseg"] == 256
    assert preset["psd"]["noverlap"] == 128
    assert preset["psd"]["nw"] == 3.0
    assert preset["psd"]["k"] == 5
    assert preset["max_freq"] == 0.5


def test_raw_periodogram_preset_uses_periodogram_method():
    preset = PRESETS["Raw periodogram"]
    assert preset["psd"]["method"] == "periodogram"


def test_long_window_preset_uses_512_sample_window():
    preset = PRESETS["Long window"]
    assert preset["psd"]["nperseg"] == 512


def test_presets_never_set_signal_type():
    # Spec constraint: applying the area-to-diameter square root to already-
    # converted diameter data is silently wrong, so signal_type deliberately
    # has no default anywhere. A preset that filled it in would defeat that
    # safeguard for anyone who picks a preset without checking every field.
    for name, preset in PRESETS.items():
        assert "signal_type" not in preset, f"preset {name!r} must not set signal_type"


# ---------------------------------------------------------------------------
# psd_params_from_dict
# ---------------------------------------------------------------------------


def test_psd_params_from_dict_roundtrip():
    psd_dict = PRESETS["Gordon lab default"]["psd"]
    params = psd_params_from_dict(psd_dict)
    assert isinstance(params, PSDParams)
    assert params.method == "multitaper_welch"
    assert params.nperseg == 256
    assert params.nw == 3.0


def test_psd_params_from_dict_roundtrip_all_fields_non_default():
    # Every field here disagrees with PSDParams' own dataclass default
    # (method="multitaper_welch", nperseg=256, noverlap=128, window="hann",
    # nw=3.0, k=5, zero_pad=1, scaling="density") -- which are, not by
    # coincidence, also the "Gordon lab default" preset's numbers. The test
    # above therefore cannot tell psd_params_from_dict apart from a stub
    # that ignores its argument and returns PSDParams(); this one can.
    psd_dict = {
        "method": "welch", "nperseg": 64, "noverlap": 16, "window": "hamming",
        "nw": 2.5, "k": 3, "zero_pad": 2, "scaling": "spectrum",
    }
    params = psd_params_from_dict(psd_dict)
    assert params.method == "welch"
    assert params.nperseg == 64
    assert params.noverlap == 16
    assert params.window == "hamming"
    assert params.nw == 2.5
    assert params.k == 3
    assert params.zero_pad == 2
    assert params.scaling == "spectrum"


# ---------------------------------------------------------------------------
# build_parameters_dict / parameters_to_json / parameters_from_json
# ---------------------------------------------------------------------------


def test_build_parameters_dict_and_json_roundtrip():
    psd_params = PSDParams(method="multitaper_welch", nperseg=256, noverlap=128, nw=3.0, k=5)
    params_dict = build_parameters_dict(
        signal_type="area", normalize="dx_x0_median", detrend_method="linear",
        highpass_cutoff=0.005, highpass_order=4, trim_seconds=0.0,
        psd_params=psd_params, max_freq=0.5, xscale="log",
        bands={"ultra_low": (0.005, 0.04)}, fs=2.0,
    )
    assert params_dict["app_version"] == "1.0.0"
    assert params_dict["fs"] == 2.0
    assert params_dict["psd"]["nperseg"] == 256

    json_str = parameters_to_json(params_dict)
    reloaded = parameters_from_json(json_str)
    assert reloaded == params_dict


def test_build_parameters_dict_and_json_roundtrip_modified_values():
    # Every value disagrees with the Gordon lab default preset, so a round
    # trip (or a build_parameters_dict) that silently substituted preset/
    # default values for the ones actually passed in is caught field by
    # field, not just by incidental whole-dict equality. method="welch" is
    # the one method that reads `window` (asserted below as "blackman",
    # unmolested) but never reads `nw`/`k` -- fix-wave item 3 nulls those
    # two in the export precisely because this method ignores them, so
    # their expected value here is None, not the 2.0/2 passed in. That is
    # not "silently substituting a default" (PSDParams' own defaults are
    # 3.0/5, not None): a regression back to reporting *any* specific
    # number here -- 2.0/2, or 3.0/5 -- would still fail this assertion.
    psd_params = PSDParams(
        method="welch", nperseg=128, noverlap=32, window="blackman",
        nw=2.0, k=2, zero_pad=4, scaling="spectrum",
    )
    params_dict = build_parameters_dict(
        signal_type="diameter", normalize="zscore", detrend_method="constant",
        highpass_cutoff=None, highpass_order=6, trim_seconds=12.5,
        psd_params=psd_params, max_freq=0.35, xscale="linear",
        bands={"custom_band": (0.02, 0.09)}, fs=7.5,
    )

    json_str = parameters_to_json(params_dict)
    reloaded = parameters_from_json(json_str)

    assert reloaded == params_dict
    assert reloaded["signal_type"] == "diameter"
    assert reloaded["normalize"] == "zscore"
    assert reloaded["detrend"] == "constant"
    assert reloaded["highpass_cutoff"] is None
    assert reloaded["highpass_order"] == 6
    assert reloaded["trim_seconds"] == 12.5
    assert reloaded["max_freq"] == 0.35
    assert reloaded["xscale"] == "linear"
    assert reloaded["bands"] == {"custom_band": [0.02, 0.09]}
    assert reloaded["fs"] == 7.5
    assert reloaded["psd"] == {
        "method": "welch", "nperseg": 128, "noverlap": 32, "window": "blackman",
        "nw": None, "k": None, "zero_pad": 4, "scaling": "spectrum",
    }


def test_build_parameters_dict_nulls_fields_the_method_does_not_consume():
    """IMPORTANT fix-wave item 3. `compute_psd` only reads `window` for
    method="welch" (periodogram hardcodes a boxcar; the two multitaper
    methods use a DPSS taper bank, not any single named window), and only
    reads `nw`/`k` for the two multitaper methods (confirmed by reading
    psd_methods.compute_psd) -- a field the selected method ignores must
    not be exported as though it were used. One case per method, each
    with every one of window/nw/k set to an unmistakable non-default
    value, so a field that should be nulled but isn't would be caught
    immediately rather than by coincidence."""
    base_kwargs = dict(
        signal_type="area", normalize="dx_x0_median", detrend_method="linear",
        highpass_cutoff=0.005, highpass_order=4, trim_seconds=0.0,
        max_freq=0.5, xscale="log", bands={"x": (0.01, 0.1)}, fs=2.0,
    )

    periodogram_psd = build_parameters_dict(
        psd_params=PSDParams(
            method="periodogram", nperseg=None, noverlap=None, window="blackman",
            nw=7.0, k=9, zero_pad=1, scaling="density",
        ),
        **base_kwargs,
    )["psd"]
    assert periodogram_psd["window"] is None
    assert periodogram_psd["nw"] is None
    assert periodogram_psd["k"] is None

    welch_psd = build_parameters_dict(
        psd_params=PSDParams(
            method="welch", nperseg=200, noverlap=100, window="hamming",
            nw=7.0, k=9, zero_pad=1, scaling="density",
        ),
        **base_kwargs,
    )["psd"]
    assert welch_psd["window"] == "hamming"  # the one method that DOES read window
    assert welch_psd["nw"] is None
    assert welch_psd["k"] is None

    multitaper_welch_psd = build_parameters_dict(
        psd_params=PSDParams(
            method="multitaper_welch", nperseg=200, noverlap=100, window="hamming",
            nw=4.0, k=7, zero_pad=1, scaling="density",
        ),
        **base_kwargs,
    )["psd"]
    assert multitaper_welch_psd["window"] is None
    assert multitaper_welch_psd["nw"] == 4.0
    assert multitaper_welch_psd["k"] == 7

    multitaper_thomson_psd = build_parameters_dict(
        psd_params=PSDParams(
            method="multitaper_thomson", nperseg=None, noverlap=None, window="hamming",
            nw=4.0, k=7, zero_pad=1, scaling="density",
        ),
        **base_kwargs,
    )["psd"]
    assert multitaper_thomson_psd["window"] is None
    assert multitaper_thomson_psd["nw"] == 4.0
    assert multitaper_thomson_psd["k"] == 7


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


def _synthetic_long_df():
    fs = 4.0
    n = 400
    t = np.arange(n) / fs
    rows = []
    for animal in (1, 2):
        for condition in ("pre", "TMX"):
            values = 1000.0 + 5.0 * np.sin(2 * np.pi * 0.1 * t) + animal
            for i, v in enumerate(values, start=1):
                rows.append(
                    {
                        "animal": animal, "sex": "M", "region": "R1",
                        "condition": condition, "sample": i, "value": float(v),
                    }
                )
    return pd.DataFrame(rows, columns=["animal", "sex", "region", "condition", "sample", "value"]), fs


def _close_pipeline_figures(result):
    for fig in result["figures"]["group_average"].values():
        plt.close(fig)
    for fig in result["figures"]["per_animal"].values():
        plt.close(fig)


def test_run_pipeline_returns_expected_keys_and_shapes():
    long_df, fs = _synthetic_long_df()
    psd_params = PSDParams(method="welch", nperseg=100, noverlap=50, window="hann")
    result = run_pipeline(
        long_df, fs=fs, signal_type="diameter", normalize="dx_x0_median",
        detrend_method="linear", highpass_cutoff=None, highpass_order=4, trim_seconds=0.0,
        psd_params=psd_params, band_defs={"low": (0.05, 0.4)}, max_freq=0.5,
    )
    try:
        assert set(result.keys()) == {"full_psd", "band_summary", "figures"}
        assert set(result["full_psd"]["region"].unique()) == {"R1"}
        assert (result["full_psd"]["frequency"] <= 0.5).all()
        assert len(result["band_summary"]) == 4  # 2 animals x 2 conditions
        assert "low" in result["band_summary"].columns
        assert set(result["figures"]["group_average"].keys()) == {"R1"}
        assert set(result["figures"]["per_animal"].keys()) == {"R1"}
    finally:
        _close_pipeline_figures(result)


def test_run_pipeline_band_entirely_outside_range_is_nan_not_zero():
    # np.trapezoid over an empty frequency mask returns 0.0, not NaN --
    # bands.integrate_bands guards this, but nothing here may launder that
    # NaN back into 0.0 (a real, wrong "measurement") on its way through
    # run_pipeline's band_summary.
    long_df, fs = _synthetic_long_df()
    psd_params = PSDParams(method="welch", nperseg=100, noverlap=50, window="hann")
    result = run_pipeline(
        long_df, fs=fs, signal_type="diameter", normalize="dx_x0_median",
        detrend_method="linear", highpass_cutoff=None, highpass_order=4, trim_seconds=0.0,
        psd_params=psd_params,
        band_defs={"low": (0.05, 0.4), "out_of_range": (100.0, 200.0)},
        max_freq=0.5,
    )
    try:
        band_summary = result["band_summary"]
        assert len(band_summary) == 4
        assert "out_of_range" in band_summary.columns
        assert band_summary["out_of_range"].isna().all()
        assert (band_summary["out_of_range"] != 0.0).all()
        assert not band_summary["low"].isna().any()
    finally:
        _close_pipeline_figures(result)


def test_run_pipeline_propagates_reserved_band_name_error():
    # Integration check: a band named after a group-identity column is
    # Task 4's Critical defect (it silently overwrote the real animal id
    # with a band-power float). run_pipeline must let bands.integrate_bands'
    # guard raise through unchanged, not swallow or reword it.
    long_df, fs = _synthetic_long_df()
    psd_params = PSDParams(method="welch", nperseg=100, noverlap=50, window="hann")
    with pytest.raises(ValueError, match="'animal' is reserved"):
        run_pipeline(
            long_df, fs=fs, signal_type="diameter", normalize="dx_x0_median",
            detrend_method="linear", highpass_cutoff=None, highpass_order=4, trim_seconds=0.0,
            psd_params=psd_params, band_defs={"animal": (0.05, 0.4)}, max_freq=0.5,
        )


# ---------------------------------------------------------------------------
# build_results_zip
# ---------------------------------------------------------------------------


def test_build_results_zip_contains_expected_files():
    band_summary = pd.DataFrame({"animal": [1], "low": [0.5]})
    full_psd = pd.DataFrame({"frequency": [0.1], "power": [1.0]})
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    parameters = {"app_version": "1.0.0", "fs": 2.0}

    zip_bytes = build_results_zip(band_summary, full_psd, {"demo_figure": fig}, parameters)
    plt.close(fig)
    assert isinstance(zip_bytes, bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
        assert names == {"band_summary.csv", "full_psd.csv", "demo_figure.png", "parameters.json"}
        for member in names:
            assert zf.getinfo(member).file_size > 0

        roundtripped = pd.read_csv(io.BytesIO(zf.read("band_summary.csv")))
        assert roundtripped.iloc[0]["low"] == pytest.approx(0.5)

        full_psd_roundtrip = pd.read_csv(io.BytesIO(zf.read("full_psd.csv")))
        assert full_psd_roundtrip.iloc[0]["frequency"] == pytest.approx(0.1)
        assert full_psd_roundtrip.iloc[0]["power"] == pytest.approx(1.0)

        params_roundtrip = parameters_from_json(zf.read("parameters.json").decode("utf-8"))
        assert params_roundtrip == parameters


def test_build_results_zip_preserves_nan_band_value_through_csv_roundtrip():
    # Companion to the run_pipeline NaN test: the CSV write/read path inside
    # build_results_zip must not coerce an out-of-range band's NaN to 0.0,
    # nor drop the row or column.
    band_summary = pd.DataFrame(
        {"animal": [1, 2], "low": [0.5, 0.6], "out_of_range": [float("nan"), float("nan")]}
    )
    full_psd = pd.DataFrame({"frequency": [0.1], "power": [1.0]})
    parameters = {"app_version": "1.0.0", "fs": 2.0}

    zip_bytes = build_results_zip(band_summary, full_psd, {}, parameters)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        assert set(zf.namelist()) == {"band_summary.csv", "full_psd.csv", "parameters.json"}
        roundtripped = pd.read_csv(io.BytesIO(zf.read("band_summary.csv")))
        assert list(roundtripped.columns) == ["animal", "low", "out_of_range"]
        assert roundtripped["out_of_range"].isna().all()
        assert (roundtripped["out_of_range"] != 0.0).all()
        assert roundtripped.iloc[0]["low"] == pytest.approx(0.5)
        assert roundtripped.iloc[1]["low"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Global constraints (explicit, regression-proof guards)
# ---------------------------------------------------------------------------


def _imported_module_names(module) -> set[str]:
    """Every top-level module name this module's source actually imports
    (via `import x` or `from x import ...`), ignoring docstrings and
    comments -- so a docstring that merely mentions a module by name (e.g.
    explaining where something is wired in the Streamlit UI layer) cannot
    make this check fail or pass vacuously."""
    tree = ast.parse(inspect.getsource(module))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_bundle_module_has_no_streamlit_imports():
    assert "streamlit" not in _imported_module_names(bundle_module)


def test_bundle_module_has_no_statistics_imports():
    imported = _imported_module_names(bundle_module)
    assert "scipy" not in imported  # scipy.stats specifically is the concern
    assert "statsmodels" not in imported
