"""Figure 4 — ultra-slow and vasomotion arteriole oscillations across states
(WT only), per the shared plan in scripts/figs_prelim/AGENT_BRIEF.md and
common.py. Produces:

  fig4_A_example.png/pdf   - panel A, representative stable NREM-like segment
  fig4_B-E_spectra.png/pdf - panels B-E, whole/state spectra, band power,
                             Ca2+ spectra/coherence/cross-correlation, and the
                             frequency-resolved Awake-vs-NREM test strip
  tables/fig4_*.csv        - all underlying numeric tables
  notes/fig4.md            - tight write-up

Run (from repo root):
  ./venv/bin/python -B -m scripts.figs_prelim.fig4
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, coherence, correlate, sosfiltfilt

sys.path.insert(0, ".")
from scripts.figs_prelim import common as C  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GROUP = "WT"
TRANS_EXCL_S = 15.0  # +/- window around any state change / MA, excluded


# --------------------------------------------------------------------------- #
# helpers local to this figure (per brief rule 1: define, don't edit common.py)
# --------------------------------------------------------------------------- #
def bandpass_filter(y, fs, lo, hi, order=4):
    """Zero-phase Butterworth band-pass; NaNs interpolated then restored."""
    y = np.asarray(y, float)
    fin = np.isfinite(y)
    if fin.sum() < 10:
        return np.full_like(y, np.nan)
    work = y.copy()
    if not fin.all():
        idx = np.arange(len(work))
        work[~fin] = np.interp(idx[~fin], idx[fin], work[fin])
    nyq = fs / 2.0
    hi_c = min(hi, 0.98 * nyq)
    sos = butter(order, [lo, hi_c], btype="bandpass", fs=fs, output="sos")
    out = sosfiltfilt(sos, work)
    out[~fin] = np.nan
    return out


def exclude_transitions(labels, excl_s=TRANS_EXCL_S):
    """Copy of per-bin labels with bins within +/-excl_s of any state change
    (bout boundary) set to None. A boundary is any bin where the label at
    bin i differs from bin i-1 (so this also captures MA onsets/offsets,
    which are themselves short Awake bouts bounded by state changes)."""
    lab = np.asarray(labels, dtype=object).copy()
    n = len(lab)
    nb = int(round(excl_s / C.BIN))
    bad = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if lab[i] != lab[i - 1]:
            bad[max(0, i - nb):min(n, i + nb)] = True
    out = lab.copy()
    out[bad] = None
    return out


def state_window_count(labels, state):
    return len(C.state_windows(labels, state))


def freq_mask(f, band):
    return (f >= band[0]) & (f < band[1])


def mean_sem(mat, axis=0):
    """Mean +/- SEM across rows that are all-finite is too strict for ragged
    per-mouse spectra on a common grid; use nan-aware stats, dof via finite
    count per column."""
    m = np.nanmean(mat, axis=axis)
    n = np.sum(np.isfinite(mat), axis=axis)
    sd = np.nanstd(mat, axis=axis, ddof=1) if mat.shape[axis] > 1 else np.zeros_like(m)
    sem = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), np.nan)
    return m, sem


# --------------------------------------------------------------------------- #
# data assembly
# --------------------------------------------------------------------------- #
def load_wt():
    keys = C.keys_for(GROUP)
    cycles = {k: C.load_cycle(k) for k in keys}
    mice = sorted({c.info.mouse for c in cycles.values()})
    return keys, cycles, mice


def panel_a_segment(cycles):
    """Longest NREM run across WT cycles -> (key, cyc, t_start, dur_s)."""
    best = None
    for k, cyc in cycles.items():
        b = C.bouts(cyc.labels)
        nrem = b[b.state == "NREM"]
        if nrem.empty:
            continue
        row = nrem.loc[nrem.dur_s.idxmax()]
        if best is None or row.dur_s > best[3]:
            best = (k, cyc, float(row.t_start), float(row.dur_s))
    return best


# --------------------------------------------------------------------------- #
# Figure A: representative stable NREM-like segment
# --------------------------------------------------------------------------- #
def make_fig_a(cycles):
    key, cyc, t0, dur = panel_a_segment(cycles)
    # pad a little on each side so the bandpass filter has run-in and the
    # reader can see the segment sits inside a longer NREM bout
    pad = 20.0
    win = (t0 - pad, t0 + dur + pad)

    sigs = ["pupil", "whisk", "art", "ca"]
    present = [s for s in sigs if s in cyc.sig]

    fs_art = cyc.fs["art"]
    t_art, y_art = cyc.sig["art"]
    m_art = (t_art >= win[0]) & (t_art <= win[1])
    bands_filt = {}
    for name, (lo, hi) in C.BANDS.items():
        bands_filt[name] = bandpass_filter(y_art[m_art], fs_art, lo, hi)

    n_rows = len(present) + len(C.BANDS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 1.5 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    row = 0
    for name in present:
        t, y = cyc.sig[name]
        m = (t >= win[0]) & (t <= win[1])
        ax = axes[row]
        C.shade_states(ax, cyc.labels, t_offset=0.0, states=("Awake", "NREM", "REM"))
        ax.plot(t[m], y[m], color=C.SIG_COL[name], lw=0.8)
        ax.set_ylabel(C.SIG_LABEL[name], fontsize=9)
        ax.set_xlim(win)
        row += 1

    for name, (lo, hi) in C.BANDS.items():
        ax = axes[row]
        C.shade_states(ax, cyc.labels, t_offset=0.0, states=("Awake", "NREM", "REM"))
        ax.plot(t_art[m_art], bands_filt[name], color=C.SIG_COL["art"], lw=0.9)
        ax.set_ylabel(f"art {name.replace('_', ' ')}\n({lo:g}-{hi:g} Hz)", fontsize=8)
        ax.set_xlim(win)
        row += 1

    # mark the exact NREM bout extent used to select the segment
    for ax in axes:
        ax.axvspan(t0, t0 + dur, facecolor="none", edgecolor="k", lw=1.0, ls="--")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        f"Fig 4A — longest WT NREM-like run: {key}  (dur={dur:.0f} s)\n"
        "shading = manual state (blue Awake / red NREM / yellow REM); "
        "dashed box = NREM bout extent\n" + C.PRELIM_STAMP,
        fontsize=9,
    )
    fig.tight_layout(h_pad=0.4)
    return fig, pd.DataFrame([dict(key=key, mouse=cyc.info.mouse, t_start=t0, dur_s=dur)])


# --------------------------------------------------------------------------- #
# Figure B: whole-record arteriole PSD, mouse-averaged
# --------------------------------------------------------------------------- #
def whole_spectra(cycles, signal="art"):
    """Per-cycle whole-record PSD for `signal` -> rows (key, mouse, f, P)."""
    rows = []
    f_common = None
    for k, cyc in cycles.items():
        if signal not in cyc.sig:
            continue
        t, y = cyc.sig[signal]
        fs = cyc.fs[signal]
        f, P = C.psd_whole(y, fs)
        if f is None:
            continue
        if f_common is None:
            f_common = f
        Pi = np.interp(f_common, f, P) if len(f) != len(f_common) or not np.allclose(f, f_common) else P
        rows.append(dict(key=k, mouse=cyc.info.mouse, P=Pi))
    return f_common, rows


def mouse_avg_spectra(f_common, rows):
    """rows (key, mouse, P) -> per-mouse mean P, then grand mean+SEM."""
    by_mouse = {}
    for r in rows:
        by_mouse.setdefault(r["mouse"], []).append(r["P"])
    mouse_mat = np.array([np.mean(v, axis=0) for v in by_mouse.values()])
    mice = list(by_mouse.keys())
    mean, sem = mean_sem(mouse_mat, axis=0)
    return mice, mouse_mat, mean, sem


def make_panel_b(ax, cycles):
    f, rows = whole_spectra(cycles, "art")
    mice, mouse_mat, mean, sem = mouse_avg_spectra(f, rows)
    for name, (lo, hi) in C.BANDS.items():
        ax.axvspan(lo, hi, color="grey", alpha=0.15, lw=0)
    for row in mouse_mat:
        ax.plot(f, row, color=C.GROUP_COL["WT"], alpha=0.15, lw=0.7)
    ax.plot(f, mean, color=C.GROUP_COL["WT"], lw=1.8, label=f"WT mean (n={len(mice)} days)")
    ax.fill_between(f, mean - sem, mean + sem, color=C.GROUP_COL["WT"], alpha=0.3, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.005, C.F_MAX)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("arteriole PSD\n(ΔD/D₀² / Hz)")
    ax.set_title("4B. whole-record arteriole spectrum")
    ax.legend(fontsize=7, loc="lower left")
    return pd.DataFrame({"f": f, "mean": mean, "sem": sem,
                         **{f"mouse_{i}": mouse_mat[i] for i in range(len(mice))}})


# --------------------------------------------------------------------------- #
# Figure C: state-specific arteriole spectra, with/without transition excl.
# --------------------------------------------------------------------------- #
def state_spectra_all_mice(cycles, state, excl=False, min_windows=2):
    """Per-mouse mean state PSD for `state`, averaging cycles within a mouse
    weighted equally per cycle. Returns dict mouse -> (P, n_windows_total),
    excluding mice with < min_windows windows total."""
    per_mouse_cycle = {}
    win_counts = {}
    for k, cyc in cycles.items():
        if "art" not in cyc.sig:
            continue
        t, y = cyc.sig["art"]
        fs = cyc.fs["art"]
        labels = exclude_transitions(cyc.labels) if excl else cyc.labels
        f, P, nwin = C.psd_state(t, y, fs, labels, state)
        win_counts.setdefault(cyc.info.mouse, 0)
        win_counts[cyc.info.mouse] += nwin
        if P is None:
            continue
        per_mouse_cycle.setdefault(cyc.info.mouse, []).append(P)
    out = {}
    for mouse, plist in per_mouse_cycle.items():
        if win_counts.get(mouse, 0) < min_windows:
            continue
        out[mouse] = np.mean(plist, axis=0)
    return C.STATE_F, out, win_counts


def make_panel_c(fig, gs_slice, cycles):
    """Two spectra axes (raw vs transition-excluded) + a BH-FDR bar strip."""
    axes = [fig.add_subplot(gs_slice[0]), fig.add_subplot(gs_slice[1])]
    strip_ax = fig.add_subplot(gs_slice[2])

    tables = {}
    window_counts = {}
    for col, (excl, title) in enumerate([(False, "4C raw state spectra"),
                                          (True, "4C transitions excluded (±15 s)")]):
        ax = axes[col]
        for name, (lo, hi) in C.BANDS.items():
            ax.axvspan(lo, hi, color="grey", alpha=0.12, lw=0)
        f = None
        state_mats = {}
        for state in ("Awake", "NREM"):
            f, by_mouse, wc = state_spectra_all_mice(cycles, state, excl=excl)
            window_counts[(excl, state)] = wc
            mat = np.array(list(by_mouse.values())) if by_mouse else np.empty((0, len(C.STATE_F)))
            state_mats[state] = (mat, list(by_mouse.keys()))
            if mat.shape[0] == 0:
                continue
            mean, sem = mean_sem(mat, axis=0)
            ax.plot(f, mean, color=C.STATE_COL[state], lw=1.8,
                    label=f"{state} (n={mat.shape[0]} days)")
            ax.fill_between(f, mean - sem, mean + sem, color=C.STATE_COL[state], alpha=0.25, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.01, 0.3)
        ax.set_xlabel("frequency (Hz)")
        if col == 0:
            ax.set_ylabel("arteriole PSD\n(60-s windows)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
        tables[excl] = state_mats

    # frequency-resolved Awake vs NREM test (on the raw, non-excluded spectra,
    # restricted to mice present in both states), BH-FDR across bins, shown
    # as a bar strip under panel C
    mat_a, mice_a = tables[False]["Awake"]
    mat_n, mice_n = tables[False]["NREM"]
    common_mice = [m for m in mice_a if m in mice_n]
    f = C.STATE_F
    pvals = np.full(len(f), np.nan)
    if len(common_mice) >= 3:
        ia = [mice_a.index(m) for m in common_mice]
        in_ = [mice_n.index(m) for m in common_mice]
        A = np.log10(np.clip(mat_a[ia], 1e-12, None))
        N = np.log10(np.clip(mat_n[in_], 1e-12, None))
        from scipy import stats as sstats
        for j in range(len(f)):
            a, n = A[:, j], N[:, j]
            ok = np.isfinite(a) & np.isfinite(n)
            if ok.sum() >= 3 and np.any((n - a)[ok] != 0):
                try:
                    pvals[j] = sstats.wilcoxon(n[ok], a[ok]).pvalue
                except ValueError:
                    pass
    qvals = C.bh(pvals)
    sig = np.where(np.isfinite(qvals) & (qvals < 0.05), 1.0, np.nan)
    n_sig = int(np.isfinite(sig).sum())
    strip_ax.scatter(f, np.ones_like(f), s=0)  # keep axis scale aligned to f
    strip_ax.bar(f, sig, width=np.gradient(f), color="black", align="center")
    strip_ax.set_xscale("log")
    strip_ax.set_xlim(0.01, 0.3)
    strip_ax.set_ylim(0, 1.3)
    strip_ax.set_yticks([])
    strip_ax.set_xlabel("frequency (Hz)")
    strip_ax.set_title(f"Awake vs NREM per-bin Wilcoxon, BH-FDR<0.05 (n={len(common_mice)} days)",
                       fontsize=8)
    if n_sig == 0:
        min_q = np.nanmin(qvals) if np.isfinite(qvals).any() else np.nan
        strip_ax.text(0.5, 0.5, f"no bins survive BH-FDR<0.05 (min q={min_q:.2g}; "
                                f"n={len(common_mice)} days limits attainable p, see wilcoxon_min_p)",
                      transform=strip_ax.transAxes, ha="center", va="center", fontsize=8, color="0.3")

    freq_table = pd.DataFrame(dict(f=f, p=pvals, q=qvals))

    # window-count table
    rows = []
    for excl in (False, True):
        for state in ("Awake", "NREM"):
            wc = window_counts[(excl, state)]
            for mouse, n in wc.items():
                rows.append(dict(transitions_excluded=excl, state=state, mouse=mouse, n_windows=n))
    win_table = pd.DataFrame(rows)
    return win_table, freq_table, tables


# --------------------------------------------------------------------------- #
# Figure D: band power by state, paired per mouse
# --------------------------------------------------------------------------- #
def cycle_band_power_by_state(cycles):
    """Per-cycle, per-state band power (vasomotion reliable; ultra_slow shown
    greyed/labelled). Returns a long dataframe: key, mouse, state, band, power."""
    rows = []
    for k, cyc in cycles.items():
        if "art" not in cyc.sig:
            continue
        t, y = cyc.sig["art"]
        fs = cyc.fs["art"]
        for state in ("Awake", "NREM"):
            f, P, nwin = C.psd_state(t, y, fs, cyc.labels, state)
            if P is None:
                continue
            for band_name, band in C.BANDS.items():
                bp = C.band_power(f, P, band)
                rows.append(dict(key=k, mouse=cyc.info.mouse, group=cyc.info.group,
                                 state=state, band=band_name, power=bp, n_windows=nwin))
    return pd.DataFrame(rows)


def make_panel_d(ax_vaso, ax_ultra, band_df):
    """Paired dot plots per mouse (mouse-level mean across cycles), band power
    on log10 y, Awake vs NREM, one axis per band."""
    results = {}
    for ax, band_name, greyed in [(ax_vaso, "vasomotion", False), (ax_ultra, "ultra_slow", True)]:
        sub = band_df[band_df.band == band_name]
        mouse_state = sub.groupby(["mouse", "state"], as_index=False)["power"].mean()
        piv = mouse_state.pivot(index="mouse", columns="state", values="power")
        piv = piv.dropna(subset=["Awake", "NREM"]) if {"Awake", "NREM"}.issubset(piv.columns) else piv.iloc[0:0]
        color = "0.65" if greyed else C.GROUP_COL["WT"]
        for mouse, row in piv.iterrows():
            ax.plot([0, 1], [row["Awake"], row["NREM"]], "-o", color=color, alpha=0.8, ms=5, lw=1.2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Awake", "NREM"])
        ax.set_yscale("log")
        ax.set_ylabel(f"{band_name} band power\n(ΔD/D₀²)")
        title = f"4D {band_name}"
        if greyed:
            title += " — not resolvable in 60-s windows"
            ax.set_title(title, fontsize=8, color="0.4")
        else:
            ax.set_title(title, fontsize=9)

        pt = C.paired_test(piv["Awake"], piv["NREM"]) if len(piv) else C.paired_test([], [])
        lmm_df = sub.dropna(subset=["power"]).copy()
        lmm_df["log_power"] = np.log10(np.clip(lmm_df["power"], 1e-12, None))
        lmm_res = C.lmm(lmm_df, "log_power", "state", group="mouse")
        results[band_name] = dict(paired=pt, lmm=lmm_res, n_mice=len(piv))
        ratio = (10 ** (np.log10(np.clip(piv["NREM"], 1e-12, None))
                        - np.log10(np.clip(piv["Awake"], 1e-12, None))).mean()
                 if len(piv) else np.nan)
        ax.text(0.5, 1.12, f"n={len(piv)} days, NREM/Awake ratio={ratio:.2f}\n"
                           f"paired-t p={pt['t_p']:.3g}, Wilcoxon p={pt['wilcoxon_p']:.3g}",
                transform=ax.transAxes, ha="center", fontsize=7)
    return results


# --------------------------------------------------------------------------- #
# Figure E: astrocyte Ca2+ spectra, coherence, cross-correlation
# --------------------------------------------------------------------------- #
def make_panel_e(fig, gs_slice, cycles):
    ax_spec, ax_coh, ax_xcorr = (fig.add_subplot(gs_slice[0]), fig.add_subplot(gs_slice[1]),
                                 fig.add_subplot(gs_slice[2]))

    # whole-record Ca spectra (mouse-averaged), same convention as 4B
    f, rows = whole_spectra(cycles, "ca")
    mice, mouse_mat, mean, sem = mouse_avg_spectra(f, rows)
    for name, (lo, hi) in C.BANDS.items():
        ax_spec.axvspan(lo, hi, color="grey", alpha=0.12, lw=0)
    ax_spec.plot(f, mean, color=C.SIG_COL["ca"], lw=1.8, label=f"whole (n={len(mice)} days)")
    ax_spec.fill_between(f, mean - sem, mean + sem, color=C.SIG_COL["ca"], alpha=0.3, lw=0)

    # state Ca spectra
    for state in ("Awake", "NREM"):
        f_s, by_mouse, _ = state_spectra_all_mice_generic(cycles, "ca", state)
        mat = np.array(list(by_mouse.values())) if by_mouse else np.empty((0, len(C.STATE_F)))
        if mat.shape[0] == 0:
            continue
        m, s = mean_sem(mat, axis=0)
        ax_spec.plot(f_s, m, color=C.STATE_COL[state], lw=1.4, ls="--",
                    label=f"{state} 60s (n={mat.shape[0]} days)")
    ax_spec.set_xscale("log")
    ax_spec.set_yscale("log")
    ax_spec.set_xlim(0.005, C.F_MAX)
    ax_spec.set_xlabel("frequency (Hz)")
    ax_spec.set_ylabel("Ca²⁺ PSD\n(ΔF/F₀² / Hz)")
    ax_spec.set_title("4E Ca²⁺ spectra", fontsize=9)
    ax_spec.legend(fontsize=6, loc="lower left")

    # coherence per mouse (whole record, same HP/detrend as psd_whole), mean
    # coherence per band reported in a table
    coh_rows = []
    xc_rows = []
    per_mouse_coh = {}
    for k, cyc in cycles.items():
        if "art" not in cyc.sig or "ca" not in cyc.sig:
            continue
        t_a, y_a = cyc.sig["art"]
        t_c, y_c = cyc.sig["ca"]
        fs = cyc.fs["art"]  # art and ca are both ~1.955 Hz, same clock-ish grid
        ya = C._fill(y_a)
        yc = C._fill(y_c)
        if ya is None or yc is None:
            continue
        n = min(len(ya), len(yc))
        if n < 256:
            continue
        ya = ya[:n]
        yc = yc[:n]
        from vasomotion_app.preprocess import highpass_filter as hp
        from scipy.signal import detrend as sp_detrend
        ya_f = sp_detrend(hp(ya, fs, C.HP_CUTOFF, C.HP_ORDER), type="linear")
        yc_f = sp_detrend(hp(yc, fs, C.HP_CUTOFF, C.HP_ORDER), type="linear")
        f_coh, Cxy = coherence(ya_f, yc_f, fs=fs, nperseg=256, noverlap=128)
        per_mouse_coh.setdefault(cyc.info.mouse, []).append(Cxy)

        for band_name, band in C.BANDS.items():
            m = freq_mask(f_coh, band)
            coh_rows.append(dict(key=k, mouse=cyc.info.mouse, band=band_name,
                                 mean_coh=float(np.nanmean(Cxy[m])) if m.sum() else np.nan))

        # cross-correlation of band-passed Ca and arteriole (per cycle)
        for band_name, (lo, hi) in C.BANDS.items():
            a_bp = bandpass_filter(ya, fs, lo, hi)
            c_bp = bandpass_filter(yc, fs, lo, hi)
            if not (np.isfinite(a_bp).all() and np.isfinite(c_bp).all()):
                continue
            a_bp = (a_bp - a_bp.mean()) / (a_bp.std() + 1e-12)
            c_bp = (c_bp - c_bp.mean()) / (c_bp.std() + 1e-12)
            xc = correlate(a_bp, c_bp, mode="full") / len(a_bp)
            lags = (np.arange(len(xc)) - (len(a_bp) - 1)) / fs
            keep = np.abs(lags) <= 30
            xc_rows.append(dict(key=k, mouse=cyc.info.mouse, band=band_name,
                                lag=lags[keep], xcorr=xc[keep]))

    coh_df = pd.DataFrame(coh_rows)
    mouse_coh_mat = np.array([np.mean(v, axis=0) for v in per_mouse_coh.values()]) if per_mouse_coh else np.empty((0,))
    if mouse_coh_mat.size:
        mcoh, scoh = mean_sem(mouse_coh_mat, axis=0)
        for name, (lo, hi) in C.BANDS.items():
            ax_coh.axvspan(lo, hi, color="grey", alpha=0.12, lw=0)
        ax_coh.plot(f_coh, mcoh, color="black", lw=1.6)
        ax_coh.fill_between(f_coh, mcoh - scoh, mcoh + scoh, color="black", alpha=0.25, lw=0)
    ax_coh.set_xscale("log")
    ax_coh.set_xlim(0.005, C.F_MAX)
    ax_coh.set_ylim(0, 1)
    ax_coh.set_xlabel("frequency (Hz)")
    ax_coh.set_ylabel("art–Ca² coherence")
    ax_coh.set_title(f"4E coherence (n={len(per_mouse_coh)} days)", fontsize=9)

    # cross-correlation, mouse-averaged per band
    xc_df_rows = []
    for band_name in C.BANDS:
        band_rows = [r for r in xc_rows if r["band"] == band_name]
        by_mouse = {}
        for r in band_rows:
            by_mouse.setdefault(r["mouse"], []).append(r["xcorr"])
        if not by_mouse:
            continue
        lag = band_rows[0]["lag"]
        mat = np.array([np.mean(v, axis=0) for v in by_mouse.values()])
        m, s = mean_sem(mat, axis=0)
        ax_xcorr.plot(lag, m, label=f"{band_name} (n={len(by_mouse)} days)")
        ax_xcorr.fill_between(lag, m - s, m + s, alpha=0.25, lw=0)
        peak_lag = lag[np.nanargmax(np.abs(m))]
        for mouse, mcorr in by_mouse.items():
            xc_df_rows.append(dict(band=band_name, mouse=mouse,
                                   peak_lag_s=lag[np.nanargmax(np.abs(np.mean(mcorr, axis=0)))],
                                   peak_xcorr=np.max(np.abs(np.mean(mcorr, axis=0)))))
        ax_xcorr.axvline(peak_lag, color="grey", lw=0.6, ls=":")
    ax_xcorr.axvline(0, color="k", lw=0.6)
    ax_xcorr.set_xlabel("lag (s), + = Ca leads art")
    ax_xcorr.set_ylabel("cross-corr (band-passed, z)")
    ax_xcorr.set_title("4E Ca–arteriole cross-correlation", fontsize=9)
    ax_xcorr.legend(fontsize=7)

    xc_summary = pd.DataFrame(xc_df_rows)
    return coh_df, xc_summary


def state_spectra_all_mice_generic(cycles, signal, state, min_windows=2):
    per_mouse_cycle = {}
    win_counts = {}
    for k, cyc in cycles.items():
        if signal not in cyc.sig:
            continue
        t, y = cyc.sig[signal]
        fs = cyc.fs[signal]
        f, P, nwin = C.psd_state(t, y, fs, cyc.labels, state)
        win_counts.setdefault(cyc.info.mouse, 0)
        win_counts[cyc.info.mouse] += nwin
        if P is None:
            continue
        per_mouse_cycle.setdefault(cyc.info.mouse, []).append(P)
    out = {}
    for mouse, plist in per_mouse_cycle.items():
        if win_counts.get(mouse, 0) < min_windows:
            continue
        out[mouse] = np.mean(plist, axis=0)
    return C.STATE_F, out, win_counts


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    C.ensure_dirs()
    keys, cycles, mice = load_wt()
    print(f"WT: {len(keys)} cycles, {len(mice)} days")

    # --- Figure A ---
    fig_a, seg_table = make_fig_a(cycles)
    C.save_fig(fig_a, "fig4_A_example")
    C.save_table(seg_table, "fig4_A_segment")
    plt.close(fig_a)

    # --- Figure B-E composite ---
    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.1, 1.0, 0.9, 1.0], hspace=0.55, wspace=0.38)

    ax_b = fig.add_subplot(gs[0, 0])
    b_table = make_panel_b(ax_b, cycles)

    c_slice = [gs[0, 1], gs[0, 2], gs[1, 1:]]  # raw spectrum, excl spectrum, bar strip spans both cols
    # gs[1, 1:] is a 2-col span; split manually below instead for clean strip
    c_slice = [gs[0, 1], gs[0, 2]]
    ax_c_raw = fig.add_subplot(c_slice[0])
    ax_c_excl = fig.add_subplot(c_slice[1])
    ax_strip = fig.add_subplot(gs[1, 1:])

    win_table, freq_table, state_mats = make_panel_c(fig, [c_slice[0], c_slice[1], gs[1, 1:]], cycles)
    # make_panel_c created its own axes via add_subplot on the passed slices;
    # discard the placeholders above to avoid duplicate empty axes
    fig.delaxes(ax_c_raw)
    fig.delaxes(ax_c_excl)
    fig.delaxes(ax_strip)

    ax_d_vaso = fig.add_subplot(gs[1, 0])
    ax_d_ultra = fig.add_subplot(gs[2, 0])
    band_df = cycle_band_power_by_state(cycles)
    d_results = make_panel_d(ax_d_vaso, ax_d_ultra, band_df)

    e_slice = [gs[2, 1], gs[2, 2], gs[3, 0]]
    coh_df, xc_summary = make_panel_e(fig, e_slice, cycles)

    fig.suptitle("Figure 4 (WT) — ultra-slow & vasomotion arteriole oscillations across states\n"
                + C.PRELIM_STAMP, fontsize=10)
    C.save_fig(fig, "fig4_B-E_spectra")
    plt.close(fig)

    # --- tables ---
    C.save_table(b_table, "fig4_B_whole_spectra")
    C.save_table(win_table, "fig4_C_window_counts")
    C.save_table(freq_table, "fig4_C_freq_resolved_test")
    C.save_table(band_df, "fig4_D_band_power_by_cycle")
    C.save_table(coh_df, "fig4_E_coherence")
    C.save_table(xc_summary, "fig4_E_crosscorr_summary")

    lmm_rows = []
    for band_name, res in d_results.items():
        if isinstance(res["lmm"], pd.DataFrame):
            t = res["lmm"].copy()
            t["band"] = band_name
            lmm_rows.append(t)
    lmm_table = pd.concat(lmm_rows, ignore_index=True) if lmm_rows else pd.DataFrame()
    if not lmm_table.empty:
        C.save_table(lmm_table, "fig4_D_lmm")

    paired_rows = []
    for band_name, res in d_results.items():
        p = dict(res["paired"])
        p["band"] = band_name
        paired_rows.append(p)
    paired_table = pd.DataFrame(paired_rows)
    C.save_table(paired_table, "fig4_D_paired_test")

    write_notes(keys, mice, win_table, paired_table, lmm_table, freq_table, coh_df, xc_summary, seg_table)
    print("done.")


def write_notes(keys, mice, win_table, paired_table, lmm_table, freq_table, coh_df, xc_summary, seg_table):
    n_sig_bins = int(((freq_table.q < 0.05) & freq_table.q.notna()).sum())
    n_tested_bins = int(freq_table.p.notna().sum())

    def fmt_row(band):
        r = paired_table[paired_table.band == band]
        if r.empty:
            return "n/a"
        r = r.iloc[0]
        ratio = 10 ** (np.log10(max(r.mean_b, 1e-12)) - np.log10(max(r.mean_a, 1e-12))) if np.isfinite(r.mean_a) and np.isfinite(r.mean_b) else np.nan
        return (f"n={int(r.n)} days, Awake mean={r.mean_a:.3g}, NREM mean={r.mean_b:.3g}, "
               f"NREM/Awake ratio={ratio:.2f}, paired-t p={r.t_p:.3g}, "
               f"Wilcoxon p={r.wilcoxon_p:.3g} (min attainable {r.wilcoxon_min_p:.3g})")

    win_summary = (win_table[~win_table.transitions_excluded]
                  .groupby("state")["n_windows"].agg(["sum", "mean", "min", "max"]))
    excl_summary = (win_table[win_table.transitions_excluded]
                    .groupby("state")["n_windows"].agg(["sum", "mean", "min", "max"]))

    coh_summary = coh_df.groupby("band")["mean_coh"].agg(["mean", "std", "count"]) if not coh_df.empty else None
    xc_summary_g = xc_summary.groupby("band")[["peak_lag_s", "peak_xcorr"]].mean() if not xc_summary.empty else None

    lines = []
    lines.append("# Figure 4 (WT) — ultra-slow & vasomotion arteriole oscillations across states\n")
    lines.append(f"Group WT: {len(keys)} cycles / {len(mice)} recording days. "
                "Bands: ultra_slow 0.005-0.04 Hz, vasomotion 0.04-0.15 Hz (C.BANDS).\n")

    lines.append("## Panels\n")
    seg = seg_table.iloc[0]
    lines.append(f"- **4A** representative segment: longest WT NREM run = {seg.key} "
                "({:.0f}-{:.0f} s, dur {:.0f} s), mouse {}. Shows pupil/whisker/arteriole/Ca and "
                "arteriole band-passed (zero-phase Butterworth) in each band."
                .format(seg.t_start, seg.t_start + seg.dur_s, seg.dur_s, seg.mouse))
    lines.append(f"- **4B** whole-record arteriole PSD (C.psd_whole), mean per cycle -> mean per "
                f"mouse -> mean ± SEM across n={len(mice)} days, log-log, both bands shaded; "
                "faint per-mouse spectra.")
    lines.append("- **4C** state-specific arteriole spectra (Awake vs NREM, C.psd_state, 60-s "
                "windows), raw and transition-excluded (±15 s / 3 bins around any state change or "
                "MA) versions shown side by side. Window counts per mouse in "
                "fig4_C_window_counts.csv; mice with <2 windows in a state are excluded from that "
                "state (see counts below).")
    lines.append(f"- Frequency-resolved Awake-vs-NREM test (per-bin paired Wilcoxon on log power, "
                f"BH-FDR across bins): {n_sig_bins}/{n_tested_bins} frequency bins significant at "
                "q<0.05 (bar strip under 4C).")
    lines.append(f"- **4D vasomotion** (reliable at 60 s): {fmt_row('vasomotion')}")
    lines.append(f"- **4D ultra_slow** (greyed, NOT resolvable in 60-s windows — shown for "
                f"completeness only): {fmt_row('ultra_slow')}")
    if not lmm_table.empty:
        lines.append("- LMM (log10 band power ~ state, random intercept per mouse) fit for both "
                    "bands on cycle-level data; see fig4_D_lmm.csv for coefficients/p-values.")
    else:
        lines.append("- LMM on band power could not be fitted (see common.lmm requirements: "
                    "needs >=3 mice and >=8 observations).")
    if coh_summary is not None:
        lines.append("- **4E** coherence (whole-record, nperseg 256) mean per band across days: " +
                    "; ".join(f"{b}: {row['mean']:.3f} ± {row['std']:.3f} (n={int(row['count'])} obs)"
                              for b, row in coh_summary.iterrows()))
    else:
        lines.append("- **4E** coherence could not be computed (insufficient paired art/Ca data).")
    if xc_summary_g is not None:
        lines.append("- **4E** band-passed Ca–arteriole cross-correlation, mean peak per band: " +
                    "; ".join(f"{b}: lag={row['peak_lag_s']:.2f} s, |xcorr|={row['peak_xcorr']:.3f}"
                              for b, row in xc_summary_g.iterrows()))
    lines.append("")
    lines.append("## Window counts (60-s state windows per mouse)\n")
    lines.append("Raw labels: " + "; ".join(f"{s} total={int(r['sum'])}, mean/mouse={r['mean']:.1f}, "
                                            f"range [{int(r['min'])},{int(r['max'])}]"
                                            for s, r in win_summary.iterrows()))
    lines.append("Transition-excluded (±15 s): " + "; ".join(
        f"{s} total={int(r['sum'])}, mean/mouse={r['mean']:.1f}, range [{int(r['min'])},{int(r['max'])}]"
        for s, r in excl_summary.iterrows()))
    lines.append("")

    lines.append("## Could not make / limitations\n")
    lines.append("- Ultra-slow band (0.005-0.04 Hz) power is NOT resolvable from 60-s state "
                "windows (frequency resolution ~1/60 Hz = 0.017 Hz, too coarse relative to the "
                "band width); 4D shows it greyed out as requested, and no state-specific "
                "ultra-slow claim is made. Only the whole-record spectrum (4B/4E, >=131-s "
                "multitaper segments) resolves it.")
    lines.append("- REM-like state was not analyzed quantitatively (too few/short bouts in most "
                "WT cycles for a 60-s-window spectral estimate); shown nowhere in this figure.")
    lines.append("- Mice with <2 qualifying 60-s windows for a state are excluded from that "
                "state's panel-C/D average (see window-count table); this shrinks n per state "
                "below the full 8 WT days.")
    lines.append("- Coherence and cross-correlation use whole-record Ca and arteriole traces at "
                "their native ~1.955 Hz sampling, high-pass (0.005 Hz, 4th-order Butterworth) and "
                "linearly detrended exactly as in C.psd_whole, per the brief.")
    lines.append("")

    lines.append("## Caveats\n")
    lines.append("- Unit = recording day (mouse), n=8 for WT; cycle- and window-level points are "
                "shown faintly/behind mouse-level summaries throughout.")
    lines.append("- With n=8 days, Wilcoxon's minimum attainable two-sided p is limited (see "
                "fig4_D_paired_test.csv, wilcoxon_min_p column); do not over-interpret p-values "
                "near that floor.")
    lines.append("- No causal claims: band-power differences and coherence/cross-correlation are "
                "descriptive associations, not evidence of a causal arteriole-Ca2+ link.")
    lines.append("- Pupil is shown only as within-cycle z (not comparable in absolute units "
                "across sessions), per common.py's normalisation.")
    lines.append("")

    lines.append("## Reproduce\n")
    lines.append("```\n./venv/bin/python -B -m scripts.figs_prelim.fig4\n```\n")

    with open(C.OUT / "notes" / "fig4.md", "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
