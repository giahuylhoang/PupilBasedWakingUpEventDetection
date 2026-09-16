"""Streamlit UI for the vasomotion PSD tool. Layout, widgets, and calls into
the analysis modules only -- no analysis logic lives here.

Three defects found by reading ahead into this file's own reference
implementation (task-8-brief.md) are fixed here rather than transcribed:

1. The frequency-axis ("log"/"linear") radio previously had no effect: it was
   captured and written into parameters.json, but never passed to a figure.
   `bundle.run_pipeline`'s return contract is locked by Task 7's end-to-end
   test to exactly {full_psd, band_summary, figures} and takes no xscale
   argument, so the group-average figures it returns are always rendered at
   figures.py's own default (xscale="log"). Fixed by re-rendering the
   group-average figures at the user's chosen scale immediately after
   run_pipeline returns, closing the ones run_pipeline handed back first so
   nothing leaks (see the "Defect 1" comment below). plot_per_animal_small_
   multiples and plot_method_comparison take no xscale parameter at all, so
   there is nothing to wire for those two.
2. `bands.validate_band_edges` is still called right before `run_pipeline`
   and surfaced with `st.warning`, exactly as designed: Task 6 deliberately
   left this call out of `bundle.py` (it is advisory, not a pipeline error)
   specifically so this file would be the one place it is wired in.
3. That warning fires on the Gordon lab DEFAULT preset itself (fs=2.0 Hz,
   nperseg=256 -> grid spacing 0.0078125 Hz, and the default ultra_low band's
   0.005 Hz lower edge sits below that) -- this is correct, not a bug (Task 7
   confirmed it by direct computation), so most users will see it on their
   first run. It is framed with an introductory sentence explaining that the
   numbers are still computed, so it reads as informative rather than as an
   error, and it is never hidden inside a collapsed expander.

Two more fixes beyond those three, both cheap and directly motivated by the
"a control that silently does nothing is worse than no control" principle
Defect 1 was found under:

4. Restoring settings from a downloaded parameters.json actually restores
   every widget's value (via st.session_state, set before each widget is
   instantiated) rather than only displaying the JSON for the user to
   "manually match" -- the design spec is explicit that a settings file
   "can be loaded back into the app to restore every control to its
   previous state." st.data_editor (used for the band-edge table) refuses
   direct session_state writes at any time (a hard Streamlit policy, not a
   version quirk -- confirmed empirically), so its restoration instead gives
   it a fresh widget key with the restored rows as that key's first-ever
   value, which is not subject to the same restriction.
5. The NW number input previously allowed values as low as 0.5, but K's
   range is 1..2*NW-1: any NW below 1.0 makes that range empty (max < min),
   which crashes the whole app with an uncaught StreamlitAPIException the
   moment such an NW is chosen (confirmed against streamlit 1.63.0: the
   brief's original code raises StreamlitValueBelowMinError). Fixed with
   two independent, overlapping protections, either of which alone already
   prevents the crash: NW's widget minimum raised to 1.0 (the true lower
   bound for a non-empty K range -- at NW=1.0, 2*NW-1=1, so K=1 is always
   valid), and K's own `max(int(2*NW-1), 1)` floor, which guards the same
   case if NW ever reached the K-range computation below 1.0 by some other
   path (e.g. a hand-edited parameters.json). K is also actively clamped
   down to that ceiling (`if k_input > k_max: k_input = k_max`, not merely
   left to Streamlit's own default handling) whenever a still-larger K
   survives a drop in NW, so it never resets to 1 except when 1 truly is
   the only valid value (NW=1.0).

Fix round 1 (see the report for the full review) added a sixth fix, in the
same "a control that silently does nothing" family as Defect 1:

6. Selecting a preset used to change the *computed* PSD method/window/
   bands correctly while the Advanced widgets (Method, Window length, NW,
   ...) kept showing whatever had been set by hand before the preset was
   picked -- a bench scientist reading the sidebar, not the downloaded
   parameters.json, would record the wrong method in a methods section.
   Fixed by writing a selected preset's values into the same widget
   session_state keys `_apply_loaded_settings` already uses (extracted
   into the shared `_push_settings_into_widgets`), the first time that
   preset is selected, and by removing the separate "use the preset dict
   instead of the widgets" branch entirely: `psd_params`/`band_defs` are
   now always built from the widgets, so a preset can only ever act by
   seeding them, never by silently overriding them behind their backs.

Also: every matplotlib Figure this file creates or receives from another
module is closed once it has been displayed and/or written into the results
zip, so a long-running server process does not accumulate open figures
across reruns (Streamlit reruns this whole script on every widget
interaction). A NaN band-summary value (bands.integrate_bands returns NaN,
never 0.0, when a band has fewer than two resolvable frequency points) is
explained in the UI rather than left as an unexplained blank cell. The
sampling-rate live readout combines sample count, implied duration, lowest
resolvable frequency, and the number of windows available to average into
one always-visible panel right under the Sampling rate field (using a
placeholder container filled in once the PSD method further down the
sidebar is known), rather than splitting the window-count part into a
collapsed "Advanced" expander where a wrong fs would be easy to miss.
"""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from vasomotion_app import bands as bands_module
from vasomotion_app import bundle as bundle_module
from vasomotion_app import figures as figures_module
from vasomotion_app import io_excel
from vasomotion_app import preprocess
from vasomotion_app import psd_methods
from vasomotion_app import stats as stats_module
from vasomotion_app.psd_methods import PSDParams

# Imported after figures_module (which calls matplotlib.use("Agg") as the
# first thing it does) so pyplot never gets a chance to pick a different,
# possibly GUI-requiring backend first.
import matplotlib.pyplot as plt

st.set_page_config(page_title="Vasomotion PSD Tool", layout="wide")
st.title("Vasomotion PSD Tool")

DEFAULT_BANDS = {"ultra_low": (0.005, 0.04), "myogenic": (0.04, 0.15)}
DEFAULT_BAND_ROWS = [
    {"band": name, "lower_hz": lo, "upper_hz": hi} for name, (lo, hi) in DEFAULT_BANDS.items()
]


def _push_settings_into_widgets(settings: dict) -> None:
    """Push the preprocessing/PSD/output fields shared by a parameters dict
    and a PRESETS[name] entry (both have the same "normalize", "detrend",
    "highpass_cutoff", "highpass_order", "trim_seconds", "psd", "max_freq",
    "xscale", "bands" sub-shape -- a preset entry simply omits "fs",
    "signal_type", and "app_version") into the session_state keys the
    corresponding widgets below are keyed on.

    Must run before any of those widgets are instantiated in this script
    run -- Streamlit widgets read `st.session_state[key]` as their current
    value if present, and take it as the initial value on first creation
    otherwise, so setting it earlier in the same run is enough; no rerun is
    needed. `bands` is handled separately (see `_pending_band_rows` below)
    because st.data_editor refuses session_state writes outright.

    Shared by `_apply_loaded_settings` (a full parameters.json reload) and
    `_apply_preset_to_widgets` (Finding 1's fix: selecting a preset must
    make these same widgets redraw showing what will actually run, not
    leave them showing whatever was set by hand before the preset was
    picked while the preset's values are what actually get computed).
    Neither touches "preset_select" or "fs"/"signal_type" -- callers own
    those directly, since a preset must never touch fs or signal_type (no
    preset sets signal_type -- squaring diameter data is silently wrong)
    and only a full settings reload should force the dropdown back to
    "(custom)".
    """
    ss = st.session_state
    if settings.get("normalize") is not None:
        ss["normalize_select"] = settings["normalize"]
    if settings.get("detrend") is not None:
        ss["detrend_select"] = settings["detrend"]
    cutoff = settings.get("highpass_cutoff")
    ss["highpass_on_checkbox"] = cutoff is not None
    if cutoff is not None:
        ss["highpass_cutoff_input"] = float(cutoff)
    if settings.get("highpass_order") is not None:
        ss["highpass_order_input"] = int(settings["highpass_order"])
    if settings.get("trim_seconds") is not None:
        ss["trim_seconds_input"] = float(settings["trim_seconds"])

    psd = settings.get("psd") or {}
    if psd.get("method") is not None:
        ss["method_select"] = psd["method"]
    if psd.get("nperseg") is not None:
        ss["nperseg_input"] = int(psd["nperseg"])
    if psd.get("noverlap") is not None:
        ss["noverlap_input"] = int(psd["noverlap"])
    if psd.get("window") is not None:
        ss["window_select"] = psd["window"]
    nw = psd.get("nw")
    if nw is not None:
        ss["nw_input"] = float(nw)
    if psd.get("k") is not None:
        # K's own widget caps at 2*NW-1 for whatever NW this dict also
        # restores; clamp here so a stale/edited (nw, k) pair from an
        # older file, or an inconsistent preset, can never hand that
        # widget an out-of-range value.
        k_max = max(int(2 * float(nw if nw is not None else 3.0) - 1), 1)
        ss["k_input"] = min(int(psd["k"]), k_max)
    if psd.get("zero_pad") is not None:
        ss["zero_pad_select"] = int(psd["zero_pad"])
    if psd.get("scaling") is not None:
        ss["scaling_select"] = psd["scaling"]

    if settings.get("max_freq") is not None:
        ss["max_freq_input"] = float(settings["max_freq"])
    if settings.get("xscale") is not None:
        ss["xscale_radio"] = settings["xscale"]

    bands = settings.get("bands")
    if bands:
        ss["_pending_band_rows"] = [
            {"band": name, "lower_hz": float(edges[0]), "upper_hz": float(edges[1])}
            for name, edges in bands.items()
        ]
        # data_editor cannot be written via session_state at all (Streamlit
        # raises StreamlitValueAssignmentNotAllowedError unconditionally, not
        # only once already instantiated -- confirmed empirically). Giving it
        # a fresh key instead makes this a brand-new widget, whose "value="
        # argument (the restored rows) is honoured because the key has never
        # been seen before.
        ss["band_editor_gen"] = ss.get("band_editor_gen", 0) + 1


def _apply_loaded_settings(loaded: dict) -> None:
    """Restore a full parameters.json reload: forces the preset dropdown
    back to "(custom)" (a raw settings file is not a named preset) and
    restores fs/signal_type directly (never touched by a preset), then
    delegates the rest to `_push_settings_into_widgets`."""
    ss = st.session_state
    ss["preset_select"] = "(custom)"
    if loaded.get("fs") is not None:
        ss["fs_input"] = float(loaded["fs"])
    signal_type = loaded.get("signal_type")
    if signal_type == "area":
        ss["signal_type_radio"] = "Area"
    elif signal_type == "diameter":
        ss["signal_type_radio"] = "Diameter"
    _push_settings_into_widgets(loaded)


def _apply_preset_to_widgets(preset_name: str) -> None:
    """Finding 1 (fix round 1): make a just-selected preset's Advanced
    widgets redraw showing what will actually run. Before this fix,
    selecting a preset changed the *computed* PSD method/window/bands
    (correctly) while the Method/Window length/NW/... widgets kept
    showing whatever had been set by hand before the preset was picked --
    reproduced by the reviewer: set Method to "welch" by hand, select
    "Gordon lab default", and the Method selectbox still read "welch"
    while the results demonstrably came from multitaper_welch.
    parameters.json was always correct (it records the resolved values,
    not the widgets), so this was a display bug, not a reproducibility
    bug -- but a bench scientist reads the sidebar, not the downloaded
    JSON, before writing a methods section.

    Does not touch "fs" or "signal_type" (no preset ever sets either) or
    "preset_select" (the widget that called this already reflects the
    just-chosen preset by the normal selectbox mechanism)."""
    _push_settings_into_widgets(bundle_module.PRESETS[preset_name])


with st.sidebar:
    st.subheader("Load previous settings (optional)")
    settings_file = st.file_uploader(
        "Load a previously downloaded parameters.json", type=["json"], key="settings_upload"
    )
    if settings_file is not None:
        if st.session_state.get("_settings_file_id_applied") != settings_file.file_id:
            try:
                loaded_settings = bundle_module.parameters_from_json(
                    settings_file.getvalue().decode("utf-8")
                )
                _apply_loaded_settings(loaded_settings)
                st.session_state["_settings_file_id_applied"] = settings_file.file_id
                st.session_state["_settings_loaded_dict"] = loaded_settings
            except Exception as exc:  # noqa: BLE001 - arbitrary user-supplied file
                st.session_state["_settings_file_id_applied"] = None
                st.session_state["_settings_loaded_dict"] = None
                st.error(f"Could not load settings file: {exc}")

        loaded_dict = st.session_state.get("_settings_loaded_dict")
        applied_id = st.session_state.get("_settings_file_id_applied")
        if loaded_dict is not None and applied_id == settings_file.file_id:
            message = "Settings loaded and applied to the controls below."
            loaded_version = loaded_dict.get("app_version")
            if loaded_version != bundle_module.APP_VERSION:
                message += (
                    f" (Saved with app version {loaded_version!r}; this app is "
                    f"{bundle_module.APP_VERSION!r} -- double-check the values.)"
                )
            st.success(message)

    st.header("1. Upload")
    uploaded_file = st.file_uploader("Vasomotion Excel file", type=["xlsx"])

    st.header("2. Required settings")
    st.session_state.setdefault("fs_input", 0.0)
    fs = st.number_input(
        "Sampling rate (Hz)", min_value=0.0, step=0.1, format="%.4f", key="fs_input"
    )
    # Filled in further down, once the PSD method (and therefore the window
    # length that "number of windows" depends on) is known -- but it visibly
    # renders right here, directly below the Sampling rate field, because a
    # container's position is fixed at creation time regardless of when
    # content is written into it.
    readout_slot = st.container()

    st.session_state.setdefault("signal_type_radio", None)
    signal_type_label = st.radio("Signal is", options=["Area", "Diameter"], key="signal_type_radio")
    signal_type = signal_type_label.lower() if signal_type_label else None

    st.session_state.setdefault("preset_select", "(custom)")
    preset_name = st.selectbox(
        "Preset", options=["(custom)"] + list(bundle_module.PRESETS.keys()), key="preset_select"
    )
    # Finding 1: applied once per distinct preset selection (guarded by
    # _last_applied_preset), not on every rerun -- otherwise a later,
    # deliberate hand-edit of e.g. Method while the same preset name stays
    # selected would be silently overwritten back on the very next rerun.
    if preset_name != "(custom)":
        if st.session_state.get("_last_applied_preset") != preset_name:
            _apply_preset_to_widgets(preset_name)
            st.session_state["_last_applied_preset"] = preset_name
    else:
        st.session_state["_last_applied_preset"] = None

    with st.expander("Advanced: preprocessing"):
        st.session_state.setdefault("normalize_select", "dx_x0_median")
        normalize = st.selectbox(
            "Normalise", options=list(preprocess.NORMALIZE_METHODS), key="normalize_select"
        )
        st.session_state.setdefault("detrend_select", "linear")
        detrend_method = st.selectbox(
            "Detrend", options=list(preprocess.DETREND_METHODS), key="detrend_select"
        )
        st.session_state.setdefault("highpass_on_checkbox", True)
        highpass_on = st.checkbox("High-pass filter", key="highpass_on_checkbox")
        if highpass_on:
            st.session_state.setdefault("highpass_cutoff_input", 0.005)
            # Item 2 (fix wave): preprocess.highpass_filter silently passes
            # the trace through unchanged for any cutoff at or above the
            # Nyquist frequency (fs/2) -- by design, so it never raises or
            # warns on its own. Left unguarded here, a user could tick this
            # box, enter e.g. 1.5 Hz at fs=2 Hz, and get an untouched trace
            # while the checkbox stays ticked, the field still shows 1.5,
            # and parameters.json records highpass_cutoff: 1.5 as though it
            # had been applied -- the same "control that silently does
            # nothing" defect this branch has already fixed twice for other
            # controls. Two independent layers, neither alone sufficient:
            # `nyquist` bounds the widget itself (None -- no bound -- while
            # fs is still unset, so this is never clamped to a meaningless
            # max_value before fs is known; confirmed empirically against
            # streamlit 1.63.0 that a stale value above a newly-lowered
            # max_value degrades to min_value on its own, unlike the empty
            # min>max range that crashed the NW/K widget, so no manual
            # pre-clamp is needed here).
            nyquist = fs / 2 if fs > 0 else None
            highpass_cutoff = st.number_input(
                "High-pass cutoff (Hz)", format="%.4f",
                min_value=0.0, max_value=nyquist, key="highpass_cutoff_input",
            )
            st.session_state.setdefault("highpass_order_input", 4)
            highpass_order = st.number_input(
                "High-pass order", min_value=1, step=1, key="highpass_order_input"
            )
            # The widget bound above still leaves its own exact boundary
            # (cutoff == nyquist) selectable, which is just as inert as
            # exceeding it (highpass_filter's own condition is `>=`) -- so
            # the effective value is validated again here, against that
            # same condition, regardless of what the widget allowed. An
            # inert filter is never passed on as though it were applied:
            # highpass_cutoff becomes None -- not merely flagged with a
            # warning while the stale, ineffective number keeps flowing
            # into run_pipeline and parameters.json.
            if fs > 0:
                cutoff_warning = preprocess.validate_highpass_cutoff(highpass_cutoff, fs)
                if cutoff_warning:
                    st.warning(cutoff_warning)
                    highpass_cutoff = None
        else:
            highpass_cutoff = None
            highpass_order = 4
        st.session_state.setdefault("trim_seconds_input", 0.0)
        trim_seconds = st.number_input(
            "Trim first N seconds", min_value=0.0, step=1.0, key="trim_seconds_input"
        )

    with st.expander("Advanced: PSD method"):
        st.session_state.setdefault("method_select", "multitaper_welch")
        method = st.selectbox("Method", options=list(psd_methods.VALID_METHODS), key="method_select")

        # Fix round 2: seed every conditional widget's session_state
        # unconditionally, before knowing whether the current method will
        # render it, and read window/nw/k FROM session_state -- never from
        # a hardcoded literal -- when their widget isn't shown this run.
        # Before this, `window`/`nw`/`k` fell back to a plain local literal
        # ("hann"/3.0/5) whenever their widget didn't render for the
        # current method, silently discarding a preset's or a loaded
        # settings file's real value for that field (e.g. selecting "Raw
        # periodogram" -- window="boxcar" -- exported psd.window: "hann"
        # into parameters.json, since periodogram never shows the Window
        # function widget). compute_psd ignores `window` for every method
        # but "welch", and ignores `nw`/`k` for both "welch" and
        # "periodogram" (confirmed by reading psd_methods.compute_psd), so
        # this cannot change any computed PSD value -- it only fixes what
        # parameters.json, the settings export/reload format, reports as
        # having been configured. `nperseg`/`noverlap` are deliberately NOT
        # given this treatment: None is their genuinely correct value for
        # periodogram/multitaper_thomson (neither method has a window
        # length at all), which is exactly what every preset's own "psd"
        # dict already encodes for those two methods.
        st.session_state.setdefault("nperseg_input", 256)
        st.session_state.setdefault("noverlap_input", 128)
        st.session_state.setdefault("window_select", "hann")
        st.session_state.setdefault("nw_input", 3.0)
        st.session_state.setdefault(
            "k_input", min(5, max(int(2 * float(st.session_state["nw_input"]) - 1), 1))
        )

        nperseg = None
        noverlap = None
        if method in ("welch", "multitaper_welch"):
            nperseg = st.number_input(
                "Window length (samples)", min_value=2, step=1, key="nperseg_input"
            )
            noverlap = st.number_input("Overlap (samples)", min_value=0, step=1, key="noverlap_input")

        if method == "welch":
            window = st.selectbox(
                "Window function", options=list(psd_methods.VALID_WINDOWS), key="window_select"
            )
        else:
            window = st.session_state["window_select"]

        if method in ("multitaper_welch", "multitaper_thomson"):
            # Defect found alongside the three named in the brief: NW's
            # minimum used to be 0.5, but K must satisfy 1 <= K <= 2*NW-1,
            # and that range is empty for any NW below 1.0 -- picking such
            # an NW crashed the whole app (StreamlitAPIException: min_value
            # 1 > max_value 0), reproduced while building this file. 1.0 is
            # the true lower bound, not a defensive guess.
            nw = st.number_input("NW", min_value=1.0, step=0.5, key="nw_input")
            k_max = max(int(2 * nw - 1), 1)
            if st.session_state["k_input"] > k_max:
                st.session_state["k_input"] = k_max
            k = st.number_input(
                f"K (max {k_max})", min_value=1, max_value=k_max, step=1, key="k_input"
            )
        else:
            nw = st.session_state["nw_input"]
            k = st.session_state["k_input"]
        st.session_state.setdefault("zero_pad_select", 1)
        zero_pad = st.selectbox(
            "Zero-padding", options=[1, 2, 4], key="zero_pad_select",
            format_func=lambda x: f"{x}x" if x > 1 else "1x (none)",
        )
        st.session_state.setdefault("scaling_select", "density")
        scaling = st.selectbox("Scaling", options=list(psd_methods.VALID_SCALINGS), key="scaling_select")

    with st.expander("Advanced: output"):
        st.session_state.setdefault("max_freq_input", 0.5)
        max_freq = st.number_input("Max frequency (Hz)", min_value=0.0, step=0.05, key="max_freq_input")
        st.session_state.setdefault("xscale_radio", "log")
        xscale = st.radio("Frequency axis", options=["log", "linear"], key="xscale_radio")
        pending_band_rows = st.session_state.pop("_pending_band_rows", None)
        band_gen = st.session_state.setdefault("band_editor_gen", 0)
        band_rows = st.data_editor(
            pd.DataFrame(pending_band_rows if pending_band_rows is not None else DEFAULT_BAND_ROWS),
            num_rows="dynamic",
            key=f"band_editor_{band_gen}",
        )

    compare_mode = st.checkbox("Compare methods")
    compare_methods = []
    if compare_mode:
        compare_methods = st.multiselect(
            "Methods to compare", options=list(psd_methods.VALID_METHODS),
            default=["periodogram", "multitaper_welch"],
        )

    # Finding 1: psd_params/band_defs (and normalize/detrend_method/
    # highpass_*/trim_seconds/max_freq/xscale, already assigned directly by
    # the widgets above) are now ALWAYS built from the widgets -- there is
    # no separate "use the preset dict instead" branch any more. A preset
    # only seeds those widgets' session_state, just above, the first time
    # it is selected; it never bypasses them for the actual computation.
    # This is what makes the sidebar's displayed values and what
    # run_pipeline actually uses identical by construction, rather than by
    # convention that a future edit could silently break again.
    psd_params = PSDParams(
        method=method, nperseg=nperseg, noverlap=noverlap, window=window,
        nw=nw, k=k, zero_pad=zero_pad, scaling=scaling,
    )
    band_defs = {
        row["band"]: (row["lower_hz"], row["upper_hz"])
        for _, row in band_rows.iterrows()
        if row["band"]
    }
    # Exposed purely so tests can confirm which bands/PSD settings are
    # actually in effect (a data_editor's resulting DataFrame has no other
    # introspection path under streamlit.testing.v1 in this version, and
    # parameters.json itself is only built much later, inside the
    # uploaded-file branch, so this is also the only way to check the
    # window/nw/k fields Fix round 2 addresses without uploading a file).
    st.session_state["_test_band_defs"] = dict(band_defs)
    st.session_state["_test_psd_params"] = psd_params
    # Item 2 (fix wave): same rationale -- st.download_button exposes no
    # more to AppTest than whether it was clicked, so this is the only way
    # a test can confirm the actual value that will flow into run_pipeline
    # and parameters.json is None whenever the widget's raw number would
    # be an inert (<=0 or >= Nyquist) high-pass cutoff, not merely that a
    # warning was shown alongside an unchanged, still-ineffective number.
    st.session_state["_test_highpass_cutoff"] = highpass_cutoff

    with readout_slot:
        if uploaded_file is not None and fs > 0:
            try:
                preview_df = io_excel.load_workbook_long(io.BytesIO(uploaded_file.getvalue()))
                n_samples_preview = int(
                    preview_df.groupby(["animal", "region", "condition"]).size().max()
                )
                summary = preprocess.trace_summary(n_samples_preview, fs)
                eff_nperseg = (
                    int(psd_params.nperseg) if psd_params.nperseg is not None else n_samples_preview
                )
                eff_noverlap = int(psd_params.noverlap) if psd_params.noverlap is not None else 0
                window_stats = psd_methods.window_summary(n_samples_preview, fs, eff_nperseg, eff_noverlap)
                st.info(
                    f"**{summary['n_samples']}** samples → **{summary['duration_sec']:.1f} s** duration  \n"
                    f"Lowest resolvable frequency: **{summary['lowest_freq_hz']:.5f} Hz**  \n"
                    f"Window resolution: **{window_stats['resolution_hz']:.5f} Hz**, "
                    f"windows to average: **{window_stats['n_segments']}**"
                )
                for warning in preprocess.validate_sampling_rate(fs, n_samples_preview):
                    st.warning(warning)
            except (ValueError, io_excel.NonNumericDataError) as exc:
                st.caption(f"Cannot compute a live readout yet: {exc}")
        else:
            st.caption(
                "Upload a file and enter a sampling rate above to see sample count, "
                "duration, and resolution here."
            )

if uploaded_file is None:
    st.info("Upload a vasomotion Excel file to begin.")
elif fs <= 0:
    st.info("Enter a sampling rate to begin.")
elif signal_type is None:
    st.info("Choose whether the signal is Area or Diameter to begin.")
else:
    with pd.ExcelFile(io.BytesIO(uploaded_file.getvalue()), engine="openpyxl") as xls:
        sheet_names = list(xls.sheet_names)
    structure = io_excel.detect_sheet_structure(sheet_names)

    sheet_regions_override = {}
    if not structure["all_parsed"]:
        st.warning(
            "Some sheet names did not match the '<region> <condition>' convention. "
            "Assign region/condition manually below."
        )
        for name, parsed in structure["parsed"].items():
            if parsed is None:
                col1, col2 = st.columns(2)
                region_in = col1.text_input(f"Region for sheet '{name}'", key=f"region_{name}")
                condition_in = col2.text_input(f"Condition for sheet '{name}'", key=f"condition_{name}")
                if region_in and condition_in:
                    sheet_regions_override[name] = (region_in, condition_in)
        still_unassigned = [
            name for name, parsed in structure["parsed"].items()
            if parsed is None and name not in sheet_regions_override
        ]
        if still_unassigned:
            st.info(f"Assign region/condition for: {', '.join(still_unassigned)}")
            st.stop()

    try:
        long_df = io_excel.load_workbook_long(
            io.BytesIO(uploaded_file.getvalue()), sheet_regions=sheet_regions_override or None
        )
    except io_excel.NonNumericDataError as exc:
        st.error(str(exc))
        st.stop()

    n_regions = len(set(structure["regions"]) | {r for r, _ in sheet_regions_override.values()})
    n_conditions = len(set(structure["conditions"]) | {c for _, c in sheet_regions_override.values()})
    # Item 1 (fix wave): a naive `groupby("region")["animal"].nunique()` counts
    # DISTINCT animals per region, which says nothing about whether those are
    # the SAME animals across that region's conditions -- a region with
    # condition="pre" on {A,B,C} and condition="TMX" on an entirely different
    # {D,E,F} has 6 unique animals and 0 actually paired, but nunique()
    # reported 6 either way. io_excel.check_animal_pairing computes the true
    # per-region intersection instead; the smallest such intersection across
    # regions is shown here (same single-number caption format as before),
    # and every region whose conditions disagree gets its own plain warning
    # naming the mismatch.
    pairing = io_excel.check_animal_pairing(long_df)
    n_animals = min(pairing["n_paired_per_region"].values()) if pairing["n_paired_per_region"] else 0
    st.caption(
        f"Detected {n_regions} region(s) x {n_conditions} condition(s), "
        f"{n_animals} paired animal(s) per region."
    )
    for pairing_warning in pairing["warnings"]:
        st.warning(pairing_warning)

    # A duplicated column passes every check above: it is fully numeric, the
    # right length, and leaves the animal correctly present in both
    # conditions, so check_animal_pairing calls it perfectly paired. The one
    # real workbook this tool was built against had exactly this defect
    # (animal 461672's SpVB TMX column was a byte-for-byte copy of its SpVB
    # pre column, all 1173 cells), and it was invisible on screen because
    # the two identical PSD curves overplotted into what looked like a
    # single condition. Surfaced with st.error rather than st.warning
    # because, unlike a band-edge notice, the affected numbers are wrong
    # rather than merely approximate -- but deliberately NOT st.stop(), so
    # the rest of the run still produces its tables and the user can see the
    # scale of the problem before deciding whether to re-export.
    duplicates = io_excel.check_duplicate_traces(long_df)
    if duplicates["warnings"]:
        st.error(
            "**Identical traces detected.** Two vessels cannot produce identical "
            "floating-point values, so the columns below are duplicates in the source "
            "workbook, not measurements. Results are still computed so you can see the "
            "effect, but any statistic involving these traces is invalid until the "
            "workbook is fixed:"
        )
        for duplicate_warning in duplicates["warnings"]:
            st.warning(duplicate_warning)

    try:
        bands_module.validate_band_names(band_defs)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    n_samples_each = int(long_df.groupby(["animal", "region", "condition"]).size().max())
    if psd_params.nperseg is not None:
        try:
            psd_methods.validate_window_length(int(psd_params.nperseg), n_samples_each)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    # scaling="spectrum" is undefined for a DPSS taper bank (psd_methods.
    # validate_scaling) -- compute_psd already raises this same ValueError,
    # but checking it here too means the message appears immediately and
    # on its own, the same way an over-long window does above, rather than
    # only after run_pipeline has already preprocessed every trace.
    try:
        psd_methods.validate_scaling(psd_params.method, psd_params.scaling)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    # Defect 2 & 3: validate_band_edges is deliberately not called inside
    # bundle.run_pipeline (see bundle.py's module docstring) -- this is the
    # one place its warnings reach a user, so the call stays here, right
    # before run_pipeline, exactly as designed. periodogram/multitaper_
    # thomson have no nperseg (they use the whole trace as their one
    # window), so window_length falls back to the trace length for those
    # two methods rather than skipping the check for them entirely.
    effective_window_length = (
        int(psd_params.nperseg) if psd_params.nperseg is not None else n_samples_each
    )
    band_edge_warnings = bands_module.validate_band_edges(band_defs, effective_window_length, fs)
    if band_edge_warnings:
        # Framed as informative rather than alarming (Defect 3): this fires
        # on the Gordon lab DEFAULT preset itself (fs=2.0, nperseg=256 puts
        # the default ultra_low band's lower edge below the resolvable
        # minimum), so most users will see it on their very first run.
        st.info(
            "One or more bands reach below what this window can resolve on its own. "
            "Power there is interpolated rather than independently measured -- the "
            "numbers below are still computed using it, this is just letting you know:"
        )
        for warning in band_edge_warnings:
            st.warning(warning)

    try:
        result = bundle_module.run_pipeline(
            long_df, fs=fs, signal_type=signal_type, normalize=normalize,
            detrend_method=detrend_method, highpass_cutoff=highpass_cutoff,
            highpass_order=highpass_order, trim_seconds=trim_seconds,
            psd_params=psd_params, band_defs=band_defs, max_freq=max_freq,
        )
    except ValueError as exc:
        st.error(f"Could not run the analysis with these settings: {exc}")
        st.stop()

    # Defect 1: figures.plot_group_average defaults to xscale="log", and
    # run_pipeline has no xscale parameter (its return contract is locked to
    # exactly {full_psd, band_summary, figures} by Task 7's end-to-end
    # test), so the frequency-axis radio would otherwise have no visible
    # effect. Re-render each group-average figure at the chosen scale,
    # closing the one run_pipeline returned first so it does not leak.
    rerendered_group_average = {}
    for region, fig in result["figures"]["group_average"].items():
        plt.close(fig)
        rerendered_group_average[region] = figures_module.plot_group_average(
            result["full_psd"], region, max_freq=max_freq, xscale=xscale, yscale="log",
        )
    result["figures"]["group_average"] = rerendered_group_average
    # Same treatment for the per-animal small multiples. These previously
    # stayed locked to figures.py's log default while the group figure
    # directly above them switched to linear, which reads as a rendering bug
    # rather than a deliberate limit -- and the lab asked specifically for
    # both variants, since log is what you want for a 1/f slope and linear
    # is what you want to show a sharp peak in a narrow 0-0.1 Hz window.
    rerendered_per_animal = {}
    for region, fig in result["figures"]["per_animal"].items():
        plt.close(fig)
        rerendered_per_animal[region] = figures_module.plot_per_animal_small_multiples(
            result["full_psd"], region, max_freq=max_freq, xscale=xscale, yscale="log",
        )
    result["figures"]["per_animal"] = rerendered_per_animal
    # Exposed purely for the xscale regression test: st.pyplot renders a
    # static image with no queryable scale, so this is the only way a test
    # can confirm which scale was actually applied to the real Figure.
    st.session_state["_test_figures"] = result["figures"]

    st.header("Group-average spectra")
    for region, fig in result["figures"]["group_average"].items():
        st.pyplot(fig)

    st.header("Band summary")
    band_summary = result["band_summary"]
    nan_bands = [name for name in band_defs if band_summary[name].isna().any()]
    if nan_bands:
        st.info(
            f"Band(s) {', '.join(repr(b) for b in nan_bands)} show blank cells for some "
            "or all rows below (and in the downloaded CSV). A blank means fewer than two "
            "frequency points fell inside that band's edges under the current settings -- "
            "for example the band sits above your Max frequency cutoff, or is narrower "
            "than this window's frequency resolution. It is not a computed zero and not "
            "missing data; widen the band, raise Max frequency, or use a longer window if "
            "you need a value there."
        )
    st.dataframe(band_summary)
    st.download_button(
        "Download band summary (CSV)", band_summary.to_csv(index=False), file_name="band_summary.csv",
    )

    with st.expander("Full PSD table"):
        st.dataframe(result["full_psd"])
        st.download_button(
            "Download full PSD (CSV)", result["full_psd"].to_csv(index=False), file_name="full_psd.csv",
        )

    # Per-trace amplitude statistics. Computed here rather than inside
    # bundle.run_pipeline because Task 7's end-to-end test locks that
    # function's return dict to exactly {full_psd, band_summary, figures};
    # the settings passed are the same ones handed to run_pipeline above, so
    # these numbers describe the very traces that produced the spectra.
    trace_stats = preprocess.trace_stats_table(
        long_df, fs=fs, signal_type=signal_type, normalize=normalize,
        detrend_method=detrend_method, highpass_cutoff=highpass_cutoff,
        highpass_order=highpass_order, trim_seconds=trim_seconds,
    )
    with st.expander("Per-trace amplitude statistics"):
        st.caption(
            "Two standard deviations, because they answer different questions. "
            "**sd_normalized** is the normalised trace before the high-pass filter "
            "and detrend -- slow drift included, usually what 'how much did the vessel "
            "fluctuate' means. **sd_analyzed** is the trace the PSD was computed from, "
            "after high-pass and detrend, so it is the smaller number and is close to "
            "total band power by construction. **mean_raw** / **sd_raw** carry the "
            "absolute scale that dx/x0 divides out"
            + (
                " -- in diameter units, since Area was selected and the tool takes the "
                "square root before normalising."
                if signal_type == "area"
                else " -- in the diameter units of your input."
            )
        )
        if normalize == "zscore":
            st.info(
                "Normalisation is z-score, which sets the standard deviation to 1 by "
                "definition, so sd_normalized is 1.0 for every trace and carries no "
                "information. Use sd_raw for absolute variability, or switch "
                "normalisation to dx/x0 if you want a comparable normalised SD."
            )
        st.dataframe(trace_stats)
        st.download_button(
            "Download per-trace statistics (CSV)", trace_stats.to_csv(index=False),
            file_name="trace_stats.csv",
        )

    # Statistics. Added at the lab's request so the paired pre/post tests do
    # not have to be redone by hand in another package. Kept out of
    # bundle.run_pipeline for the same reason trace_stats is: Task 7's
    # end-to-end test locks that function's return dict to exactly
    # {full_psd, band_summary, figures}.
    stats_paired = None
    stats_region = None
    st.header("Statistics")
    try:
        baseline, treatment = stats_module.resolve_condition_order(
            long_df["condition"].unique()
        )
    except ValueError as exc:
        # Not an error state: a one-condition or three-condition workbook is
        # perfectly valid input for the spectra and band tables above, it
        # just has no single paired contrast to test.
        st.info(f"No paired comparison available. {exc}")
    else:
        stats_paired = stats_module.paired_band_tests(
            band_summary, band_names=list(band_defs.keys()),
            condition_order=(baseline, treatment),
        )
        stats_region = stats_module.region_change_tests(
            band_summary, band_names=list(band_defs.keys()),
            condition_order=(baseline, treatment),
        )
        st.caption(
            f"Within-animal change from **{baseline}** to **{treatment}**, tested on log10 "
            "band power. A difference of logs is a fold change, which is the quantity that "
            "generalises across animals with different baseline vessel sizes -- a test on "
            "raw power would be dominated by whichever animals happen to have the largest "
            "vessels. Both a parametric (paired t) and a non-parametric (Wilcoxon "
            "signed-rank) p-value are reported for every row rather than one being chosen "
            "for you: at these sample sizes a normality test has almost no power to justify "
            "such a choice, so the two agreeing is the reassurance and the two disagreeing "
            "is itself the finding. `t_p_fdr` / `wilcoxon_p_fdr` are Benjamini-Hochberg "
            "corrected across every row of this table."
        )
        if normalize == "none":
            st.warning(
                "Normalisation is off, so band power still carries each animal's absolute "
                "vessel size. The paired tests below are largely protected from that -- "
                "each animal is its own control -- but any comparison BETWEEN animals or "
                "regions is confounded by baseline vessel diameter. `mean_raw` in the "
                "per-trace statistics above is that baseline; either normalise, or carry "
                "it into your model as a covariate."
            )
        if not stats_paired.empty:
            floor = stats_paired["wilcoxon_min_p"].max()
            if pd.notna(floor) and floor > 0.01:
                st.info(
                    f"At this sample size the Wilcoxon signed-rank test cannot return a "
                    f"p-value below {floor:.3f} however large the effect (column "
                    "`wilcoxon_min_p`, shown per row because a dropped zero-difference "
                    "pair raises it further). A non-significant non-parametric result here "
                    "is a statement about n, not evidence of no effect."
                )
            if (stats_paired["n_zero_diff"] > 0).any():
                st.error(
                    "One or more rows below include an animal whose change is exactly zero "
                    "(`n_zero_diff`). That is the duplicated-column defect flagged at the "
                    "top of this page, and it pulls the estimated effect toward no change "
                    "while still counting toward n. Fix the workbook before reporting "
                    "these numbers."
                )
        st.subheader(f"Paired: {baseline} vs {treatment}, per region")
        st.dataframe(stats_paired)
        st.download_button(
            "Download paired statistics (CSV)", stats_paired.to_csv(index=False),
            file_name="stats_paired.csv",
        )
        if stats_region.empty:
            st.caption(
                "The between-region comparison needs exactly two regions; with a different "
                "number, which pair to compare is a question for you rather than a default "
                "this tool should pick."
            )
        else:
            st.subheader("Between regions: is the change different?")
            st.caption(
                "Each animal's own log10 fold change, compared between the two regions. "
                "The regions contain **different animals**, so this comparison is unpaired "
                "(Welch's t-test, which does not assume equal variances, plus Mann-Whitney "
                "U). `hedges_g` is the effect size with the small-sample correction applied."
            )
            st.dataframe(stats_region)
            st.download_button(
                "Download between-region statistics (CSV)", stats_region.to_csv(index=False),
                file_name="stats_region.csv",
            )

    st.header("Per-animal spectra")
    for region, fig in result["figures"]["per_animal"].items():
        st.pyplot(fig)

    if compare_mode and len(compare_methods) >= 2:
        st.header("Compare methods")
        preprocessed = preprocess.preprocess_long_dataframe(
            long_df, fs=fs, signal_type=signal_type, normalize=normalize,
            detrend_method=detrend_method, highpass_cutoff=highpass_cutoff,
            highpass_order=highpass_order, trim_seconds=trim_seconds,
        )
        psd_tables = {}
        band_tables = {}
        comparison_ok = True
        for m in compare_methods:
            m_nperseg = psd_params.nperseg
            m_noverlap = psd_params.noverlap
            if m in ("welch", "multitaper_welch") and m_nperseg is None:
                # The currently-selected method has no window length of its
                # own (e.g. "Raw periodogram": method="periodogram",
                # nperseg=None) -- psd_params.nperseg is that method's
                # value, not this compared method's, so it must not be
                # carried over as-is. compute_psd now rejects nperseg=None
                # for every Welch-family method with a ValueError (instead
                # of a bare TypeError from int(None)), so without this,
                # ticking "Compare methods" with the shipped default
                # selection (["periodogram", "multitaper_welch"]) under the
                # "Raw periodogram" preset would still fail here -- just
                # with a caught, readable error instead of a crash. A
                # window length valid for THIS method is substituted
                # instead: PSDParams' own dataclass default (256), capped
                # to the trace length so it is never invalid on a short
                # trace, with a matching 50%-overlap noverlap.
                m_nperseg = min(PSDParams().nperseg, n_samples_each)
                m_noverlap = m_nperseg // 2
            p = PSDParams(
                method=m, nperseg=m_nperseg, noverlap=m_noverlap,
                window=psd_params.window, nw=psd_params.nw, k=psd_params.k,
                zero_pad=psd_params.zero_pad, scaling=psd_params.scaling,
            )
            try:
                table = psd_methods.compute_psd_table(preprocessed, fs, p)
            except ValueError as exc:
                st.error(f"Could not compute {m!r} for comparison: {exc}")
                comparison_ok = False
                break
            table = table[table["frequency"] <= max_freq].reset_index(drop=True)
            psd_tables[m] = table
            band_tables[m] = bands_module.integrate_bands(table, band_defs)

        if comparison_ok:
            first_region = sorted(long_df["region"].unique())[0]
            first_condition = sorted(long_df["condition"].unique())[0]
            comparison_fig = figures_module.plot_method_comparison(
                psd_tables, first_region, first_condition, max_freq=max_freq,
                xscale=xscale, yscale="log",
            )
            st.pyplot(comparison_fig)
            plt.close(comparison_fig)

            diff_table = figures_module.band_power_percent_diff(
                band_tables, compare_methods[0], list(band_defs.keys())
            )
            if diff_table["percent_diff"].isna().any():
                st.info(
                    "Some 'percent_diff' rows are blank because the reference method's "
                    "mean power for that band is exactly 0, which makes a percent "
                    "difference undefined rather than a real number."
                )
            st.dataframe(diff_table)

    parameters = bundle_module.build_parameters_dict(
        signal_type=signal_type, normalize=normalize, detrend_method=detrend_method,
        highpass_cutoff=highpass_cutoff, highpass_order=highpass_order, trim_seconds=trim_seconds,
        psd_params=psd_params, max_freq=max_freq, xscale=xscale, bands=band_defs, fs=fs,
    )
    st.download_button(
        "Download parameters.json", bundle_module.parameters_to_json(parameters),
        file_name="parameters.json",
    )

    figures_by_name = {f"group_average_{r}": f for r, f in result["figures"]["group_average"].items()}
    figures_by_name.update({f"per_animal_{r}": f for r, f in result["figures"]["per_animal"].items()})
    zip_bytes = bundle_module.build_results_zip(
        result["band_summary"], result["full_psd"], figures_by_name, parameters,
        trace_stats=trace_stats, stats_paired=stats_paired, stats_region=stats_region,
    )
    st.download_button("Download everything (.zip)", zip_bytes, file_name="vasomotion_psd_results.zip")

    # Every Figure this run created or received (group-average originals
    # were already closed above, right after being re-rendered) is closed
    # once display and zip-building are both done with it, so a long-running
    # server process does not accumulate open figures across reruns.
    for fig in result["figures"]["group_average"].values():
        plt.close(fig)
    for fig in result["figures"]["per_animal"].values():
        plt.close(fig)
