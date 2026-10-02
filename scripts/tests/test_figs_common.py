"""Tests for scripts/figs_prelim/common.py -- the definitions every figure uses.

    ./venv/bin/python -B -m pytest scripts/tests/test_figs_common.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.figs_prelim import common as C


def lab(*runs):
    """lab(('NREM',6),('Awake',2)) -> per-bin label array."""
    out = []
    for s, n in runs:
        out += [s] * n
    return np.array(out, dtype=object)


# ---------------------------------------------------------------- events ----
def test_micro_arousal_needs_nrem_both_sides_and_max_15s():
    e = C.events(lab(("NREM", 4), ("Awake", 3), ("NREM", 4)))
    assert list(e.type) == ["MA"]
    assert e.t0.iloc[0] == 4 * C.BIN

    # 4 awake bins = 20 s: too long for an MA
    e = C.events(lab(("NREM", 4), ("Awake", 4), ("NREM", 4)))
    assert "MA" not in set(e.type)

    # NREM after is only 1 bin (5 s): sleep did not resume for >= 10 s
    e = C.events(lab(("NREM", 4), ("Awake", 2), ("NREM", 1), ("Awake", 6)))
    assert "MA" not in set(e.type)


def test_nrem_to_wake_requires_30s_nrem_and_20s_wake():
    e = C.events(lab(("NREM", 6), ("Awake", 4)))
    assert list(e.type) == ["N2W"] and e.t0.iloc[0] == 30.0
    assert C.events(lab(("NREM", 5), ("Awake", 6))).empty        # NREM too short
    assert "N2W" not in set(C.events(lab(("NREM", 6), ("Awake", 3), ("NREM", 3))).type)


def test_ma_and_n2w_are_mutually_exclusive():
    # a 15-s awake bout inside sleep is an MA, never also an N2W
    e = C.events(lab(("NREM", 8), ("Awake", 3), ("NREM", 8)))
    assert list(e.type) == ["MA"]


def test_unclear_breaks_a_bout():
    e = C.events(lab(("NREM", 3), ("Unclear", 1), ("NREM", 3), ("Awake", 6)))
    assert "N2W" not in set(e.type)     # neither NREM run reaches 30 s


def test_wake_to_nrem_and_rem():
    e = C.events(lab(("Awake", 6), ("NREM", 6), ("REM", 2)))
    assert set(e.type) == {"W2N", "REM"}


# ---------------------------------------------------------------- epochs ----
def test_epoch_baseline_and_alignment():
    fs = 2.0
    t = np.arange(0, 200, 1 / fs)
    y = np.where(t >= 100, 1.0, 0.0)               # step at 100 s
    rg = C.rel_grid(fs)
    v = C.epoch(t, y, 100.0, fs)
    assert np.nanmean(v[(rg >= -5) & (rg < 0)]) == pytest.approx(0.0)
    assert np.nanmean(v[(rg >= 1) & (rg <= 10)]) == pytest.approx(1.0)


def test_epoch_masks_out_of_range_and_gaps():
    fs = 2.0
    t = np.arange(0, 60, 1 / fs)
    y = np.ones_like(t)
    y[(t > 20) & (t < 30)] = np.nan               # 10-s gap
    rg = C.rel_grid(fs)
    v = C.epoch(t, y, 10.0, fs, baseline=False)
    assert np.isnan(v[rg < -10]).all()            # before t=0 of the record
    assert np.isnan(v[(rg > 12) & (rg < 18)]).all()   # inside the gap
    assert np.isfinite(v[(rg > 0) & (rg < 9)]).all()


def test_response_metrics_peak_and_latency():
    rg = C.rel_grid(2.0)
    M = np.zeros((1, len(rg)))
    M[0, np.argmin(np.abs(rg - 6.0))] = 3.0
    peak, auc, tpk = C.response_metrics(rg, M)
    assert peak[0] == 3.0 and tpk[0] == pytest.approx(6.0)


# ---------------------------------------------------------------- spectra ---
def test_band_power_recovers_injected_oscillation_band():
    fs = 1.955
    t = np.arange(0, 900, 1 / fs)
    rng = np.random.default_rng(0)
    slow = np.sin(2 * np.pi * 0.02 * t) + 0.05 * rng.standard_normal(len(t))
    fast = np.sin(2 * np.pi * 0.08 * t) + 0.05 * rng.standard_normal(len(t))
    for y, big, small in ((slow, "ultra_slow", "vasomotion"), (fast, "vasomotion", "ultra_slow")):
        f, P = C.psd_whole(y, fs)
        assert C.band_power(f, P, C.BANDS[big]) > 10 * C.band_power(f, P, C.BANDS[small])


def test_state_windows_purity():
    labels = lab(("NREM", 12), ("Awake", 12))
    starts = C.state_windows(labels, "NREM")
    assert starts == [0.0]                          # only the pure 60-s window
    assert C.state_windows(lab(("NREM", 11), ("Awake", 1)), "NREM") == [0.0]   # 11/12 >= 90%


# ---------------------------------------------------------------- stats -----
def test_bh_matches_statsmodels_values():
    # hand-checked: raw 0.04, 0.06, 0.0533, 0.2 -> monotone step-up gives
    # 0.04, 0.0533, 0.0533, 0.2 in input order (agrees with statsmodels fdr_bh)
    q = C.bh([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(q, [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.2])


def test_paired_test_wilcoxon_floor_and_noise_guard():
    a = np.ones(6)
    r = C.paired_test(a, a * 2)
    assert r["wilcoxon_min_p"] == pytest.approx(2 / 64)
    assert np.isnan(r["dz"])                        # identical diffs: no t on FP noise


def test_group_test_direction():
    r = C.group_test([1, 2, 3, 4], [5, 6, 7, 8])
    assert r["hedges_g"] > 0 and r["mw_p"] < 0.05


# ---------------------------------------------------------------- data ------
requires_data = pytest.mark.skipif(not C.LABEL_DIR.is_dir(), reason="label snapshot absent")


@requires_data
def test_real_cycle_loads_with_expected_signals():
    k = C.keys_for("WT")[0]
    cyc = C.load_cycle(k)
    assert {"pupil", "whisk", "art"} <= set(cyc.sig)
    t, y = cyc.sig["art"]
    assert abs(np.nanmedian(y)) < 1e-9             # dD/D0 is zero at the median
    assert 800 <= cyc.duration <= 910


@requires_data
def test_every_labelled_key_parses_and_has_a_group():
    for k in C.labeled_keys():
        assert C.parse_key(k).group in {"WT", "KD", "SCR", "NE"}


# ---------------------------------------------------------------- label modes
requires_pred = pytest.mark.skipif(not C.LABEL_PRED_DIR.is_dir(), reason="no ML predictions")


@requires_pred
def test_manual_labels_always_win_over_predictions():
    key = "IP3R2_KD_2025.11.01_cycle9"            # 29 manual bins, rest predicted
    manual = C.load_labels(key, "manual")
    mixed = C.load_labels(key, "manual+ml")
    assert len(mixed) >= len(manual)
    for i, v in enumerate(manual):
        if v is not None:
            assert mixed[i] == v
    assert sum(v is not None for v in mixed) > sum(v is not None for v in manual)


@requires_pred
def test_excluded_day_never_uses_predictions():
    key = "IP3R2_KD_2023.08.26_cycle6"            # model failed validation on this day
    assert list(C.load_labels(key, "manual+ml")) == list(C.load_labels(key, "manual"))
    assert C.label_source(key, "manual+ml") == "manual"


@requires_pred
def test_label_mode_switch_changes_kd_coverage():
    try:
        C.set_label_mode("manual")
        n_man = len(C.keys_for("KD"))
        C.set_label_mode("manual+ml")
        n_mix = len(C.keys_for("KD"))
    finally:
        C.set_label_mode("manual+ml")
    assert n_man == 7 and n_mix > n_man


# ---------------------------------------------------------------- refinements
def _ramp(onset, rise=2.0, noise=0.0, seed=0):
    t = np.arange(0, 120, 1 / C.FS_BEH)
    y = np.where(t < onset, 0.0, np.minimum((t - onset) / 3.0, 1.0) * rise)
    y = y + noise * np.random.default_rng(seed).standard_normal(len(t))
    return t, y


def test_pupil_onset_recovers_true_onset_within_a_second():
    for true_on, label in ((57.0, 60.0), (62.0, 60.0), (60.0, 60.0)):
        t, y = _ramp(true_on, noise=0.05)
        on = C.pupil_onset(t, y, label)
        assert on is not None and abs(on - true_on) < 1.0, (true_on, on)


def test_pupil_onset_rejects_flat_pupil():
    t, y = _ramp(60.0, rise=0.2)                  # below ANCHOR_MIN_RISE
    assert C.pupil_onset(t, y, 60.0) is None


def test_pupil_onset_stays_inside_search_window():
    t, y = _ramp(40.0)                            # rise 20 s before the label
    on = C.pupil_onset(t, y, 60.0)
    assert on is None or on >= 60.0 + C.ANCHOR_SEARCH[0]


def test_drop_overlapping_uses_window_and_is_per_cycle():
    import pandas as pd
    ev = pd.DataFrame(dict(type=["MA", "MA", "N2W", "N2W"], t0=[100.0, 110.0, 300.0, 100.0],
                           key=["a", "a", "a", "b"]))
    kept = C.drop_overlapping(ev)
    # the two cycle-a MAs 10 s apart drop each other; 300 s and the other cycle stay
    assert sorted(zip(kept.key, kept.t0)) == [("a", 300.0), ("b", 100.0)]


def test_lmm_flags_collapsed_mouse_variance():
    import pandas as pd
    rng = np.random.default_rng(3)
    rows = []
    for mi in range(8):                                    # no true mouse effect
        for _ in range(15):
            g = mi % 2
            rows.append(dict(mouse=f"m{mi}", g=g, y=0.1 * g + rng.normal(0, 1)))
    t = C.lmm(pd.DataFrame(rows), "y", "g")
    assert "degenerate_fit" in t and bool(t["degenerate_fit"].iloc[0])

    rows = []
    for mi in range(8):                                    # strong mouse effect
        off = rng.normal(0, 3)
        for _ in range(15):
            rows.append(dict(mouse=f"m{mi}", g=mi % 2, y=off + rng.normal(0, 1)))
    t = C.lmm(pd.DataFrame(rows), "y", "g")
    assert not bool(t["degenerate_fit"].iloc[0])
