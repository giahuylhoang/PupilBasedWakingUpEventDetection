"""Tests for vasomotion_app.app using Streamlit's AppTest harness.

The five `test_app_*` functions below are the widget-scaffold smoke tests
from task-8-brief.md, kept verbatim: they cover the "no file uploaded yet"
state, which is all AppTest can drive for a real `st.file_uploader` in some
Streamlit versions -- but streamlit 1.63.0 (pinned in requirements.txt) does
support `FileUploader.set_value(...)`, confirmed directly against this
version's `streamlit.testing.v1.element_tree` source before relying on it
(rather than assumed from the brief's more cautious note). Every test past
those five therefore drives real file uploads and asserts on behaviour, not
just the absence of an exception:

- uploading the real sample workbook and producing full results
- the frequency-axis radio actually changing a rendered figure's scale
  (Defect 1's regression guard)
- the band-edge warning appearing, informatively, for the default settings
  (Defect 3)
- a NaN band value being explained rather than left as a silent blank
- the live sampling-rate readout showing all four numbers the spec requires
- a settings-JSON round trip genuinely restoring modified widget values
  (not merely displaying the file back at the user), including the one
  widget (the band-edge data_editor) that cannot be restored via
  st.session_state and is instead handled with a rotating widget key
- two crash-shaped inputs (a malformed settings file, an out-of-range NW)
  degrading to an error message or a clamp instead of an uncaught exception

Fix round 1 (review findings) added four more:

- selecting a preset after a manual edit updates the Advanced widgets'
  *displayed* values, not just the numbers computed underneath them
  (Finding 1)
- no module except app.py imports streamlit, discovered dynamically so a
  module added later is covered automatically (Finding 2)
- periodogram/multitaper_thomson (no `nperseg`) still get a band-edge
  warning when one is due, via the trace-length fallback (Finding 3)
- the NaN-band explanation does not appear when nothing is actually NaN

Fix round 2 (a defect introduced by fix round 1's own preset refactor)
added two more: `window`/`nw`/`k` used to fall back to a hardcoded literal
whenever their widget wasn't instantiated for the current method (e.g.
"Raw periodogram" sets window="boxcar", but periodogram never shows the
Window function widget, so parameters.json reported "hann"). Covered by a
test selecting that preset and a second test covering the latent form
(non-default nw/k surviving export under a method that hides their
widget too).

`st.pyplot` renders a static image with no queryable scale, and
`streamlit.testing.v1` has no typed accessor for `st.data_editor`, so
app.py stashes the actual Figure objects, the resolved band dict, and the
resolved PSDParams into `st.session_state["_test_figures"]` /
`["_test_band_defs"]` / `["_test_psd_params"]` purely so these tests can
inspect real objects rather than rendered pixels or a downloaded file's
bytes (which `st.download_button` does not expose to AppTest either).
"""
from __future__ import annotations

import os

import pytest
from streamlit.testing.v1 import AppTest

from vasomotion_app import bundle
from vasomotion_app.psd_methods import PSDParams

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app.py")
SAMPLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "Tim",
    "Vasomotion analysis compilation_Bal K.xlsx",
)
# The workbook has no sampling rate of its own -- 2.0 Hz is this suite's own
# fixed choice, matching test_end_to_end.py, not a fact read from the file.
FS = 2.0

requires_sample_file = pytest.mark.skipif(
    not os.path.exists(SAMPLE_PATH), reason="sample workbook not present in this checkout",
)


# --- small widget-lookup helpers --------------------------------------
# AppTest exposes widgets as flat, type-grouped lists (at.sidebar.number_input,
# at.sidebar.selectbox, ...) regardless of nesting inside st.expander, so
# every widget here is found by its exact label rather than by position.

def _one(matches, description):
    assert len(matches) == 1, f"expected exactly one {description}, found {len(matches)}"
    return matches[0]


def _number_input(at, label):
    return _one([ni for ni in at.sidebar.number_input if ni.label == label], f"number_input {label!r}")


def _selectbox(at, label):
    return _one([sb for sb in at.sidebar.selectbox if sb.label == label], f"selectbox {label!r}")


def _radio(at, options):
    return _one(
        [r for r in at.sidebar.radio if list(r.options) == options],
        f"radio with options {options!r}",
    )


def _checkbox(at, label):
    return _one([c for c in at.sidebar.checkbox if c.label == label], f"checkbox {label!r}")


def _excel_uploader(at):
    return _one(
        [fu for fu in at.sidebar.file_uploader if "Excel" in fu.label], "the Excel file_uploader"
    )


def _settings_uploader(at):
    return _one(
        [fu for fu in at.sidebar.file_uploader if "parameters.json" in fu.label],
        "the settings file_uploader",
    )


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def _upload_sample_and_configure(at: AppTest) -> AppTest:
    """Upload the real sample workbook and fill in the two required fields
    (fs=2.0 Hz, signal is Area), in one rerun, matching how a user staging
    several changes before Streamlit next reruns the script would land."""
    with open(SAMPLE_PATH, "rb") as f:
        content = f.read()
    _excel_uploader(at).set_value((
        "Vasomotion analysis compilation_Bal K.xlsx", content,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ))
    _number_input(at, "Sampling rate (Hz)").set_value(FS)
    _radio(at, ["Area", "Diameter"]).set_value("Area")
    at.run()
    return at


# --- widget-scaffold smoke tests (task-8-brief.md, verbatim) ------------

def test_app_runs_without_exception():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception


def test_app_title():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert at.title[0].value == "Vasomotion PSD Tool"


def test_app_shows_upload_prompt_when_no_file():
    at = AppTest.from_file(APP_PATH)
    at.run()
    info_texts = [i.value for i in at.info]
    assert any("Upload a vasomotion Excel file" in t for t in info_texts)


def test_app_signal_type_radio_has_no_default():
    at = AppTest.from_file(APP_PATH)
    at.run()
    radios = [r for r in at.sidebar.radio if list(r.options) == ["Area", "Diameter"]]
    assert len(radios) == 1
    assert radios[0].value is None


def test_app_preset_dropdown_has_three_presets_plus_custom():
    at = AppTest.from_file(APP_PATH)
    at.run()
    preset_boxes = [sb for sb in at.sidebar.selectbox if "Gordon lab default" in list(sb.options)]
    assert len(preset_boxes) == 1
    assert set(preset_boxes[0].options) == {
        "(custom)", "Gordon lab default", "Raw periodogram", "Long window",
    }


# --- behavioural coverage ------------------------------------------------

@requires_sample_file
def test_uploading_sample_workbook_produces_results():
    at = _upload_sample_and_configure(_fresh_app())
    assert not at.exception
    assert "Group-average spectra" in [h.value for h in at.header]
    assert "Band summary" in [h.value for h in at.header]
    assert "Per-animal spectra" in [h.value for h in at.header]

    figures = at.session_state["_test_figures"]
    assert set(figures["group_average"].keys()) == {"SmFP", "SpVB"}
    assert set(figures["per_animal"].keys()) == {"SmFP", "SpVB"}

    band_summary_tables = [
        d.value for d in at.dataframe
        if list(d.value.columns) == ["animal", "sex", "region", "condition", "ultra_low", "myogenic"]
    ]
    assert len(band_summary_tables) == 1
    assert len(band_summary_tables[0]) == 24  # 4 sheets x 6 animals -- test_end_to_end.py

    download_labels = {b.label for b in at.download_button}
    assert {
        "Download band summary (CSV)", "Download full PSD (CSV)",
        "Download parameters.json", "Download everything (.zip)",
    } <= download_labels


@requires_sample_file
def test_changing_frequency_axis_radio_changes_rendered_figure_scale():
    """Regression guard for Defect 1. figures.plot_group_average defaults to
    xscale="log", and bundle.run_pipeline has no xscale parameter (its
    return contract is locked to exactly {full_psd, band_summary, figures}
    by Task 7's end-to-end test) -- so without app.py re-rendering after
    run_pipeline, this radio changes parameters.json but no figure ever
    moves off "log", however it is set."""
    at = _upload_sample_and_configure(_fresh_app())

    figures_before = at.session_state["_test_figures"]["group_average"]
    assert len(figures_before) == 2
    assert all(fig.axes[0].get_xscale() == "log" for fig in figures_before.values())

    _radio(at, ["log", "linear"]).set_value("linear")
    at.run()
    assert not at.exception

    figures_after = at.session_state["_test_figures"]["group_average"]
    assert len(figures_after) == 2
    assert all(fig.axes[0].get_xscale() == "linear" for fig in figures_after.values())
    # No control exposes yscale; the fix must not touch it.
    assert all(fig.axes[0].get_yscale() == "log" for fig in figures_after.values())


@requires_sample_file
def test_band_edge_warning_appears_with_default_preset():
    """Defect 3. At fs=2.0 with nperseg=256 the grid spacing is exactly
    0.0078125 Hz, so the Gordon lab default preset's own ultra_low band
    (lower edge 0.005 Hz) sits below the resolvable minimum -- confirmed by
    Task 7 as correct behaviour, not a bug, and by test_bands.py's own
    test_validate_band_edges_warns_below_resolution with these exact
    numbers. It must fire and must read as informative, not alarming."""
    at = _upload_sample_and_configure(_fresh_app())
    _selectbox(at, "Preset").set_value("Gordon lab default")
    at.run()
    assert not at.exception

    warnings = [w.value for w in at.warning]
    assert any("ultra_low" in w and "resolvable minimum" in w for w in warnings)

    info_texts = [i.value for i in at.info]
    assert any("still computed" in t for t in info_texts)


@requires_sample_file
def test_nan_band_value_is_explained_in_ui():
    at = _upload_sample_and_configure(_fresh_app())
    # Push Max frequency below the myogenic band (0.04-0.15 Hz) entirely, so
    # fewer than two frequency points fall inside it and integrate_bands
    # (bands.py) returns NaN for that column.
    _number_input(at, "Max frequency (Hz)").set_value(0.03)
    at.run()
    assert not at.exception

    band_summary_tables = [
        d.value for d in at.dataframe if "myogenic" in getattr(d.value, "columns", [])
    ]
    assert len(band_summary_tables) == 1
    assert band_summary_tables[0]["myogenic"].isna().all()

    info_texts = [i.value for i in at.info]
    assert any("'myogenic'" in t and "blank" in t for t in info_texts)


@requires_sample_file
def test_live_readout_shows_sample_count_duration_and_window_count():
    at = _upload_sample_and_configure(_fresh_app())
    readout = [
        i.value for i in at.info
        if "samples" in i.value and "duration" in i.value
    ]
    assert len(readout) == 1
    text = readout[0]
    assert "1173" in text  # sample count, from the real workbook
    assert "Lowest resolvable frequency" in text
    assert "windows to average" in text


def test_settings_round_trip_restores_modified_values():
    at = _fresh_app()

    # Move several controls away from their defaults first, so restoring
    # can be shown to actually change something rather than coincide with
    # values already in place.
    _number_input(at, "Sampling rate (Hz)").set_value(9.0)
    _radio(at, ["Area", "Diameter"]).set_value("Diameter")
    at.run()
    assert _number_input(at, "Sampling rate (Hz)").value == 9.0
    assert _radio(at, ["Area", "Diameter"]).value == "Diameter"

    params = bundle.build_parameters_dict(
        signal_type="area", normalize="zscore", detrend_method="constant",
        highpass_cutoff=None, highpass_order=4, trim_seconds=3.0,
        psd_params=PSDParams(
            method="welch", nperseg=128, noverlap=64, window="hamming",
            nw=3.0, k=5, zero_pad=2, scaling="spectrum",
        ),
        max_freq=0.35, xscale="linear", bands={"lowband": (0.01, 0.2)}, fs=5.5,
    )
    payload = bundle.parameters_to_json(params).encode("utf-8")
    _settings_uploader(at).set_value(("mysettings.json", payload, "application/json"))
    at.run()
    assert not at.exception
    assert any("loaded and applied" in s.value for s in at.success)

    assert _number_input(at, "Sampling rate (Hz)").value == 5.5
    assert _radio(at, ["Area", "Diameter"]).value == "Area"
    assert _selectbox(at, "Preset").value == "(custom)"
    assert _selectbox(at, "Normalise").value == "zscore"
    assert _selectbox(at, "Detrend").value == "constant"
    assert _checkbox(at, "High-pass filter").value is False
    assert _number_input(at, "Trim first N seconds").value == 3.0
    assert _selectbox(at, "Method").value == "welch"
    assert _number_input(at, "Window length (samples)").value == 128
    assert _number_input(at, "Overlap (samples)").value == 64
    assert _selectbox(at, "Window function").value == "hamming"
    assert _selectbox(at, "Zero-padding").value == 2
    assert _selectbox(at, "Scaling").value == "spectrum"
    assert _number_input(at, "Max frequency (Hz)").value == 0.35
    assert _radio(at, ["log", "linear"]).value == "linear"
    # The band-edge table (st.data_editor) has no typed AppTest accessor and
    # refuses session_state writes outright, so app.py restores it via a
    # rotating widget key instead; this is the only way to confirm that
    # path actually worked rather than silently no-opping.
    assert at.session_state["_test_band_defs"] == {"lowband": (0.01, 0.2)}

    # A settings file stays "uploaded" across reruns until cleared; a later,
    # unrelated manual edit must not be silently overwritten by it again.
    _number_input(at, "Sampling rate (Hz)").set_value(42.0)
    at.run()
    assert not at.exception
    assert _number_input(at, "Sampling rate (Hz)").value == 42.0


def test_malformed_settings_file_shows_error_not_crash():
    at = _fresh_app()
    _settings_uploader(at).set_value(("bad.json", b"{not valid json", "application/json"))
    at.run()
    assert not at.exception
    assert any("Could not load settings file" in e.value for e in at.error)


def test_nw_minimum_prevents_invalid_k_range_crash():
    """K's range is 1..2*NW-1; any NW below 1.0 makes that range empty,
    which crashed the whole app with an uncaught StreamlitAPIException
    (reproduced while building app.py, since the brief's reference code
    allows NW down to 0.5). NW's minimum is now 1.0, the true lower bound
    for a non-empty K range, so this must degrade to a clamp, not a crash."""
    at = _fresh_app()
    _number_input(at, "NW").set_value(0.5)
    at.run()
    assert not at.exception
    assert _number_input(at, "NW").value >= 1.0


# --- fix round 1 (review findings) ---------------------------------------

def test_selecting_preset_updates_method_selectbox_after_manual_edit():
    """Regression guard for Finding 1. Reproduced by the reviewer: set
    Method to "welch" by hand, then select "Gordon lab default" -- before
    this fix the Method selectbox still read "welch" while the results
    demonstrably came from multitaper_welch (parameters.json was always
    correct; only the sidebar's displayed value was stale). Selecting a
    preset must make the widget redraw showing what will actually run."""
    at = _fresh_app()
    _selectbox(at, "Method").set_value("welch")
    at.run()
    assert _selectbox(at, "Method").value == "welch"

    _selectbox(at, "Preset").set_value("Gordon lab default")
    at.run()
    assert not at.exception
    assert _selectbox(at, "Method").value == "multitaper_welch"
    assert _number_input(at, "Window length (samples)").value == 256
    assert _number_input(at, "Overlap (samples)").value == 128
    assert _number_input(at, "NW").value == 3.0


def test_no_module_except_app_imports_streamlit():
    """Architectural guard for Finding 2. Design spec: "The analysis core
    does not import Streamlit." Discovers vasomotion_app/*.py dynamically
    (not a hardcoded list) so a module added later is covered
    automatically. Scoped to the top-level package files -- not
    vasomotion_app/tests/, which legitimately imports streamlit.testing to
    drive AppTest; that is test infrastructure, not the analysis core this
    constraint protects. Reproduced against the actual bug: adding `import
    streamlit` to a scratch copy of figures.py makes this fail, naming
    figures.py, while the rest of the suite stays green (see task-8-report.md,
    Fix round 1)."""
    import ast
    import pathlib

    package_dir = pathlib.Path(__file__).resolve().parent.parent
    assert package_dir.name == "vasomotion_app"

    def _imports_streamlit(path: pathlib.Path) -> bool:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name == "streamlit" or a.name.startswith("streamlit.") for a in node.names):
                    return True
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "streamlit" or node.module.startswith("streamlit.")):
                    return True
        return False

    offenders = [
        path.name for path in sorted(package_dir.glob("*.py"))
        if path.name != "app.py" and _imports_streamlit(path)
    ]
    assert offenders == [], f"streamlit imported outside app.py in: {offenders}"


@requires_sample_file
def test_band_edge_warning_fires_under_periodogram_for_out_of_range_band():
    """Regression guard for Finding 3. periodogram/multitaper_thomson have
    no `nperseg` (they use the whole trace as their one window), so
    app.py's window_length fallback (trace length instead of nperseg) must
    still be consulted for those methods, not silently skipped -- reverting
    to the brief's original `if psd_params.nperseg is not None:` gate
    around the whole validate_band_edges call would make this go silent.
    At fs=2.0 Hz with the real sample file's 1173 samples, the whole-trace
    resolvable minimum is 2/1173 ~= 0.0017 Hz; a band edge below that must
    still warn under periodogram."""
    at = _upload_sample_and_configure(_fresh_app())
    params = bundle.build_parameters_dict(
        signal_type="area", normalize="dx_x0_median", detrend_method="linear",
        highpass_cutoff=0.005, highpass_order=4, trim_seconds=0.0,
        psd_params=PSDParams(
            method="periodogram", nperseg=None, noverlap=None, window="boxcar",
            nw=3.0, k=5, zero_pad=1, scaling="density",
        ),
        max_freq=0.5, xscale="log", bands={"tiny": (0.0005, 0.001)}, fs=FS,
    )
    payload = bundle.parameters_to_json(params).encode("utf-8")
    _settings_uploader(at).set_value(("periodogram_settings.json", payload, "application/json"))
    at.run()
    assert not at.exception
    assert _selectbox(at, "Method").value == "periodogram"
    assert at.session_state["_test_band_defs"] == {"tiny": (0.0005, 0.001)}

    warnings = [w.value for w in at.warning]
    assert any("tiny" in w and "resolvable minimum" in w for w in warnings)


@requires_sample_file
def test_nan_explanation_does_not_appear_when_no_band_is_nan():
    """Mutation guard: the NaN-band explanation is gated on `if nan_bands:`
    (app.py) -- a mutation to `if True:` would leave the rest of the suite
    green, since every other NaN-related test only checks the explanation
    DOES appear when a band genuinely is NaN. Gordon lab default's own two
    bands are both fully resolvable on the real sample file (confirmed by
    test_end_to_end.py -- zero NaNs), so no explanation should appear."""
    at = _upload_sample_and_configure(_fresh_app())
    _selectbox(at, "Preset").set_value("Gordon lab default")
    at.run()
    assert not at.exception
    info_texts = [i.value for i in at.info]
    assert not any("blank cells" in t for t in info_texts)


# --- fix round 2 (a defect introduced by fix round 1's own refactor) ----

def test_raw_periodogram_preset_exports_true_window_not_hardcoded_fallback():
    """Regression guard for the Fix-round-2 defect. The Window function
    widget only exists `if method == "welch"`; before this fix, `window`
    fell back to a hardcoded "hann" literal whenever that widget wasn't
    instantiated, silently discarding whatever a preset or a loaded
    settings file actually declared. "Raw periodogram" declares
    window="boxcar" (PRESETS["Raw periodogram"]["psd"]["window"]) while
    method="periodogram" never shows that widget -- parameters.json used
    to report "hann" regardless. compute_psd ignores `window` for every
    method except "welch" (confirmed by reading psd_methods.py), so this
    is a metadata-fidelity fix only; no PSD value changes."""
    at = _fresh_app()
    _selectbox(at, "Preset").set_value("Raw periodogram")
    at.run()
    assert not at.exception

    psd_params = at.session_state["_test_psd_params"]
    assert psd_params.method == "periodogram"
    assert psd_params.window == "boxcar"

    # The literal check the reviewer asked for: round-trip through the
    # actual parameters.json machinery, not just the PSDParams object.
    parameters = bundle.build_parameters_dict(
        signal_type="area", normalize="dx_x0_median", detrend_method="linear",
        highpass_cutoff=0.005, highpass_order=4, trim_seconds=0.0,
        psd_params=psd_params, max_freq=0.5, xscale="log",
        bands={"x": (0.01, 0.1)}, fs=FS,
    )
    reloaded = bundle.parameters_from_json(bundle.parameters_to_json(parameters))
    assert reloaded["psd"]["window"] == "boxcar"


def test_non_default_nw_k_survive_export_under_method_without_that_widget():
    """The latent form of the same defect: the NW/K widgets only exist
    `if method in ("multitaper_welch", "multitaper_thomson")`. A loaded
    settings file specifying method="welch" with non-default nw=7.0/k=9
    (moot for welch's own computation, but still real configured metadata
    -- exactly the shape "Raw periodogram" itself uses for nw/k, just with
    non-default numbers no built-in preset happens to exercise) must
    round-trip through parameters.json unchanged rather than silently
    reporting the hardcoded nw=3.0/k=5 fallback."""
    at = _fresh_app()
    params_in = bundle.build_parameters_dict(
        signal_type="area", normalize="dx_x0_median", detrend_method="linear",
        highpass_cutoff=0.005, highpass_order=4, trim_seconds=0.0,
        psd_params=PSDParams(
            method="welch", nperseg=256, noverlap=128, window="hamming",
            nw=7.0, k=9, zero_pad=1, scaling="density",
        ),
        max_freq=0.5, xscale="log", bands={"x": (0.01, 0.1)}, fs=FS,
    )
    payload = bundle.parameters_to_json(params_in).encode("utf-8")
    _settings_uploader(at).set_value(("nwk_settings.json", payload, "application/json"))
    at.run()
    assert not at.exception
    assert _selectbox(at, "Method").value == "welch"

    psd_params = at.session_state["_test_psd_params"]
    assert psd_params.nw == 7.0
    assert psd_params.k == 9
