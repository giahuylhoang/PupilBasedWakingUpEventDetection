"""Figure 7 — IP3R2 KD vs scramble: ultra-slow / vasomotion arteriole power.

Run with:  ./venv/bin/python -B scripts/figs_prelim/fig7.py

Everything here uses scripts.figs_prelim.common (C) for data loading,
normalisation, spectra and statistics. See common.py's module docstring and
scripts/figs_prelim/AGENT_BRIEF.md for the shared conventions. This script
owns only the fig7_* outputs and notes/fig7.md.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.signal import coherence as sp_coherence

sys.path.insert(0, ".")
from scripts.figs_prelim import common as C  # noqa: E402

GROUPS = ["SCR", "KD"]
BAND_NAMES = ["ultra_slow", "vasomotion"]


# --------------------------------------------------------------------------- #
# helpers local to this figure (brief: do not edit common.py)
# --------------------------------------------------------------------------- #
def whole_record_df(keys):
    """One row per cycle: whole-record PSD + band powers for art and ca."""
    rows = []
    psd_cache = {}
    for k in keys:
        cyc = C.load_cycle(k)
        if "art" not in cyc.sig:
            continue
        t, y = cyc.sig["art"]
        f, P = C.psd_whole(y, cyc.fs["art"])
        psd_cache[k] = (f, P)
        row = dict(key=k, mouse=cyc.info.mouse, group=cyc.info.group)
        for bname, band in C.BANDS.items():
            row[f"art_{bname}"] = np.log10(C.band_power(f, P, band) + 1e-12) if P is not None else np.nan
        if "ca" in cyc.sig:
            tc, yc = cyc.sig["ca"]
            fc, Pc = C.psd_whole(yc, cyc.fs["ca"])
            for bname, band in C.BANDS.items():
                row[f"ca_{bname}"] = np.log10(C.band_power(fc, Pc, band) + 1e-12) if Pc is not None else np.nan
        else:
            for bname in C.BANDS:
                row[f"ca_{bname}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows), psd_cache


def freq_matrix(keys, psd_cache, signal="art"):
    """Interpolate every cycle's whole-record PSD onto a common log-f grid."""
    f_common = np.geomspace(0.006, 0.49, 120)
    mat, meta = [], []
    for k in keys:
        if signal == "art":
            f, P = psd_cache.get(k, (None, None))
        else:
            cyc = C.load_cycle(k)
            if signal not in cyc.sig:
                continue
            t, y = cyc.sig[signal]
            f, P = C.psd_whole(y, cyc.fs[signal])
        if P is None:
            continue
        logP = np.interp(np.log10(f_common), np.log10(np.clip(f, 1e-6, None)), np.log10(P + 1e-12))
        mat.append(logP)
        info = C.parse_key(k)
        meta.append(dict(key=k, mouse=info.mouse, group=info.group))
    return f_common, np.array(mat), pd.DataFrame(meta)


def state_df(keys, states=("Awake", "NREM")):
    """One row per cycle x state: band power (vasomotion reliable; ultra_slow greyed)."""
    rows = []
    psd_cache = {}
    for k in keys:
        cyc = C.load_cycle(k)
        if "art" not in cyc.sig:
            continue
        t, y = cyc.sig["art"]
        fs = cyc.fs["art"]
        for st in states:
            f, P, nwin = C.psd_state(t, y, fs, cyc.labels, st)
            psd_cache[(k, st)] = (f, P, nwin)
            row = dict(key=k, mouse=cyc.info.mouse, group=cyc.info.group, state=st, n_windows=nwin)
            for bname, band in C.BANDS.items():
                row[f"art_{bname}"] = (
                    np.log10(C.band_power(f, P, band) + 1e-12) if (P is not None and nwin > 0) else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows), psd_cache


def bandpass(y, fs, band):
    from vasomotion_app.preprocess import highpass_filter
    from scipy.signal import butter, filtfilt

    lo, hi = band
    y = np.asarray(y, float)
    fin = np.isfinite(y)
    if fin.sum() < 20:
        return np.full_like(y, np.nan)
    yy = y.copy()
    idx = np.arange(len(yy))
    yy[~fin] = np.interp(idx[~fin], idx[fin], yy[fin])
    nyq = fs / 2.0
    hi_eff = min(hi, 0.98 * nyq)
    b, a = butter(4, [lo / nyq, hi_eff / nyq], btype="band")
    out = filtfilt(b, a, yy)
    out[~fin] = np.nan
    return out


def sd_by_state_concat(t, y, fs, labels, band, state=None):
    """SD of band-passed signal; whole-record or concatenated state-pure 60-s windows."""
    yb = bandpass(y, fs, band)
    if state is None:
        v = yb[np.isfinite(yb)]
        return float(np.std(v)) if v.size > 10 else np.nan
    starts = C.state_windows(labels, state)
    segs = []
    for s in starts:
        m = (t >= s) & (t < s + C.STATE_WIN_S)
        seg = yb[m]
        if np.isfinite(seg).mean() > 0.9:
            segs.append(seg[np.isfinite(seg)])
    if not segs:
        return np.nan
    allv = np.concatenate(segs)
    return float(np.std(allv)) if allv.size > 10 else np.nan


def event_rate_table(keys):
    ev = C.event_table(keys)
    rows = []
    for k in keys:
        cyc = C.load_cycle(k)
        dur_h = cyc.duration / 3600.0
        sub = ev[ev["key"] == k] if not ev.empty else ev
        n2w = (sub["type"] == "N2W").sum() if not sub.empty else 0
        ma = (sub["type"] == "MA").sum() if not sub.empty else 0
        rows.append(dict(key=k, mouse=cyc.info.mouse, group=cyc.info.group,
                         dur_h=dur_h, n2w=n2w, ma=ma, n2w_ma=n2w + ma))
    df = pd.DataFrame(rows)
    agg = df.groupby(["group", "mouse"], as_index=False).agg(
        dur_h=("dur_h", "sum"), n2w=("n2w", "sum"), ma=("ma", "sum"), n2w_ma=("n2w_ma", "sum"))
    agg["n2w_ma_per_h"] = agg["n2w_ma"] / agg["dur_h"]
    return agg


def mouse_level(df, cols, by=("group", "mouse")):
    return df.groupby(list(by), as_index=False)[cols].mean()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    C.ensure_dirs()
    notes = []

    # ----------------------------------------------------------------- #
    # Data pools
    # ----------------------------------------------------------------- #
    whole_keys = C.keys_for(GROUPS, labeled_only=False)
    whole_keys = [k for k in whole_keys if "art" in C.load_cycle(k).sig]
    whole_df, psd_cache_whole = whole_record_df(whole_keys)
    n_whole = whole_df.groupby("group")["mouse"].nunique().to_dict()
    n_whole_cyc = whole_df.groupby("group")["key"].nunique().to_dict()
    notes.append(f"Whole-record pool (no labels needed): SCR {n_whole_cyc.get('SCR')} cyc / "
                 f"{n_whole.get('SCR')} days; KD {n_whole_cyc.get('KD')} cyc / {n_whole.get('KD')} days.")

    tables = {}

    # =================================================================== #
    # 7A — representative traces
    # =================================================================== #
    def pct_nrem(cyc):
        lab = cyc.labels
        valid = lab[lab != None]  # noqa: E711
        if len(valid) == 0:
            return np.nan
        return 100.0 * np.mean(valid == "NREM")

    C.set_label_mode("manual+ml")
    cands = {g: [] for g in GROUPS}
    for k in whole_keys:
        cyc = C.load_cycle(k)
        if "art" not in cyc.sig or "pupil" not in cyc.sig or "ca" not in cyc.sig:
            continue
        p = pct_nrem(cyc)
        if np.isfinite(p):
            cands[cyc.info.group].append((k, p))

    best_pair, best_diff = None, np.inf
    for ks, ps in cands.get("SCR", []):
        for kk, pk in cands.get("KD", []):
            d = abs(ps - pk)
            if d < best_diff:
                best_diff, best_pair = d, (ks, kk)
    if best_pair is None:
        notes.append("7A: could not find a SCR/KD pair with pupil+arteriole+Ca all present.")
    else:
        fig, axes = plt.subplots(5, 2, figsize=(13, 11), sharex="col")
        sig_rows = [("pupil", "pupil"), ("whisk", "whisk"), ("art", "art"), ("ca", "ca"), (None, "art_us")]
        for col, k in enumerate(best_pair):
            cyc = C.load_cycle(k)
            grp = cyc.info.group
            pnrem = pct_nrem(cyc)
            for row, (sigkey, kind) in enumerate(sig_rows):
                ax = axes[row, col]
                if kind == "art_us":
                    t, y = cyc.sig["art"]
                    yb = bandpass(y, cyc.fs["art"], C.ULTRA_SLOW)
                    ax.plot(t, yb, color=C.GROUP_COL[grp], lw=0.8)
                    ax.set_ylabel("art US-band\n(ΔD/D₀)", fontsize=8)
                else:
                    t, y = cyc.sig[sigkey]
                    ax.plot(t, y, color=C.SIG_COL[sigkey], lw=0.6)
                    ax.set_ylabel(C.SIG_LABEL[sigkey], fontsize=8)
                C.shade_states(ax, cyc.labels, alpha=0.15)
                ax.tick_params(labelsize=7)
                if row == 0:
                    ax.set_title(f"{C.GROUP_LABEL[grp]}  {k}\n%NREM={pnrem:.0f}", fontsize=9)
                if row == len(sig_rows) - 1:
                    ax.set_xlabel("time (s)", fontsize=8)
        handles = [Line2D([0], [0], color=v, lw=4, alpha=0.3) for v in C.STATE_COL.values()]
        fig.legend(handles, list(C.STATE_COL.keys()), loc="upper center", ncol=4, fontsize=8,
                  bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f"Fig 7A — representative SCR vs KD traces (matched %NREM, Δ={best_diff:.1f} pts)\n{C.PRELIM_STAMP}",
                    fontsize=9, y=1.06)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        C.save_fig(fig, "fig7_A_examples")
        plt.close(fig)
        notes.append(f"7A: {best_pair[0]} (SCR, %NREM={cands and dict(cands['SCR'])[best_pair[0]]:.0f}) vs "
                     f"{best_pair[1]} (KD, %NREM={dict(cands['KD'])[best_pair[1]]:.0f}), matched to Δ{best_diff:.1f} pts.")

    # =================================================================== #
    # 7B/7C figure (whole record)
    # =================================================================== #
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])

    # 7B: mouse-mean +/- SEM PSD, log-log, bands shaded, faint per-mouse
    ax7b = fig.add_subplot(gs[0, 0:2])
    f_common, mat, meta = freq_matrix(whole_keys, psd_cache_whole, "art")
    meta["logP_row"] = range(len(meta))
    for g in GROUPS:
        idx_g = meta.index[meta["group"] == g].tolist()
        if not idx_g:
            continue
        # per-mouse mean first
        mouse_means = []
        for m, sub in meta.loc[idx_g].groupby("mouse"):
            mouse_means.append(mat[sub.index.to_numpy()].mean(axis=0))
            ax7b.plot(f_common, 10 ** mouse_means[-1], color=C.GROUP_COL[g], alpha=0.15, lw=0.8)
        mm = np.array(mouse_means)
        mean_logP = mm.mean(axis=0)
        sem_logP = mm.std(axis=0, ddof=1) / np.sqrt(len(mm)) if len(mm) > 1 else np.zeros_like(mean_logP)
        ax7b.plot(f_common, 10 ** mean_logP, color=C.GROUP_COL[g], lw=2.2,
                 label=f"{C.GROUP_LABEL[g]} (n={len(mm)} days)")
        ax7b.fill_between(f_common, 10 ** (mean_logP - sem_logP), 10 ** (mean_logP + sem_logP),
                          color=C.GROUP_COL[g], alpha=0.25, lw=0)
    for bname, band in C.BANDS.items():
        ax7b.axvspan(band[0], band[1], color="grey", alpha=0.12, lw=0)
        ax7b.text(np.sqrt(band[0] * band[1]), ax7b.get_ylim()[1] if False else 1, bname, fontsize=7,
                  ha="center", va="top", color="dimgrey", transform=ax7b.get_xaxis_transform())
    ax7b.set_xscale("log"); ax7b.set_yscale("log")
    ax7b.set_xlabel("frequency (Hz)"); ax7b.set_ylabel("arteriole PSD (density)")
    ax7b.set_title("7B. Whole-record arteriole PSD, mouse-mean ± SEM\n(faint = per-mouse mean)")
    ax7b.legend(fontsize=8, loc="lower left")

    # 7C: band power per mouse, SCR vs KD, two bands
    tabs_7c = []
    for i, bname in enumerate(BAND_NAMES):
        ax = fig.add_subplot(gs[0, 2] if i == 0 else gs[1, 0])
        mdf = mouse_level(whole_df, [f"art_{bname}"])
        for g in GROUPS:
            sub = mdf[mdf["group"] == g]
            x = 0 if g == "SCR" else 1
            jitter = (np.random.default_rng(0).random(len(sub)) - 0.5) * 0.15
            ax.scatter(x + jitter, sub[f"art_{bname}"], color=C.GROUP_COL[g], s=45, zorder=3,
                      label=f"{C.GROUP_LABEL[g]} (n={len(sub)})")
            # faint cycle-level
            csub = whole_df[whole_df["group"] == g]
            ax.scatter(x + (np.random.default_rng(1).random(len(csub)) - 0.5) * 0.3,
                      csub[f"art_{bname}"], color=C.GROUP_COL[g], s=10, alpha=0.2, zorder=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["SCR", "KD"])
        ax.set_ylabel(f"log10 {bname} power (art)")
        a = mdf.loc[mdf.group == "SCR", f"art_{bname}"]
        b = mdf.loc[mdf.group == "KD", f"art_{bname}"]
        gt = C.group_test(a, b)
        lmm_df = C.lmm(whole_df.assign(group=lambda d: d["group"]), f"art_{bname}", "C(group)")
        ax.set_title(f"7C. {bname} power/mouse\nWelch p={gt['welch_p']:.3f}, MW p={gt['mw_p']:.3f}, g={gt['hedges_g']:.2f}",
                    fontsize=9)
        ax.legend(fontsize=7)
        tables.setdefault("fig7_C_band_power_group_test", []).append(
            dict(band=bname, **gt, ratio_KD_over_SCR_log10=(b.mean() - a.mean()) if (len(a) and len(b)) else np.nan))
        if lmm_df is not None:
            lmm_df.insert(0, "band", bname)
            tables.setdefault("fig7_C_lmm", []).append(lmm_df)

    # frequency-resolved per-bin group test with BH-FDR, strip panel
    ax_strip = fig.add_subplot(gs[1, 1:3])
    pvals = []
    for j in range(mat.shape[1]):
        a = mat[(meta["group"] == "SCR").to_numpy(), j]
        b = mat[(meta["group"] == "KD").to_numpy(), j]
        gt = C.group_test(a, b)
        pvals.append(gt["welch_p"])
    qvals = C.bh(pvals)
    sig = np.array([q < 0.05 if np.isfinite(q) else False for q in qvals])
    ax_strip.plot(f_common, -np.log10(np.clip(pvals, 1e-300, 1)), color="black", lw=1)
    ax_strip.scatter(f_common[sig], -np.log10(np.clip(np.array(pvals)[sig], 1e-300, 1)), color="crimson",
                    s=20, zorder=3, label="BH q<0.05")
    for bname, band in C.BANDS.items():
        ax_strip.axvspan(band[0], band[1], color="grey", alpha=0.12, lw=0)
    ax_strip.axhline(-np.log10(0.05), color="grey", ls="--", lw=0.8)
    ax_strip.set_xscale("log")
    ax_strip.set_xlabel("frequency (Hz)"); ax_strip.set_ylabel("-log10(Welch p), SCR vs KD")
    ax_strip.set_title("7C (strip). Frequency-resolved group test on log power, BH-FDR across bins")
    ax_strip.legend(fontsize=8)

    tab_7c_freq = pd.DataFrame(dict(freq_hz=f_common, welch_p=pvals, bh_q=qvals, bh_sig_q05=sig))
    C.save_table(tab_7c_freq, "fig7_C_freq_resolved_group_test")

    fig.suptitle(f"Fig 7B-C — whole-record arteriole spectra and ultra-slow/vasomotion band power, SCR vs KD\n{C.PRELIM_STAMP}",
                fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    C.save_fig(fig, "fig7_B-C_whole")
    plt.close(fig)

    for name in ("fig7_C_band_power_group_test",):
        C.save_table(pd.DataFrame(tables[name]), name)
    if "fig7_C_lmm" in tables:
        C.save_table(pd.concat(tables["fig7_C_lmm"], ignore_index=True), "fig7_C_lmm")

    # =================================================================== #
    # 7D-7H figure: state-specific spectra, band power x state, time-domain,
    # Ca + coherence, controls. Both label modes.
    # =================================================================== #
    label_modes = ["manual+ml", "manual"]
    state_results = {}
    for mode in label_modes:
        C.set_label_mode(mode)
        keys_labeled = C.keys_for(GROUPS, labeled_only=True)
        sdf, scache = state_df(keys_labeled)
        state_results[mode] = (keys_labeled, sdf, scache)
    C.set_label_mode("manual+ml")  # restore primary for remaining panels

    fig2 = plt.figure(figsize=(18, 16))
    gs2 = fig2.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1])

    # 7D: state-specific spectra (vasomotion-era 60s windows), Awake and NREM, SCR vs KD
    # show primary (manual+ml) with manual overlay as thin dashed
    for col, state in enumerate(["Awake", "NREM"]):
        ax = fig2.add_subplot(gs2[0, col])
        for g in GROUPS:
            for mode, ls, alpha_line in (("manual+ml", "-", 1.0), ("manual", "--", 0.6)):
                keys_labeled, sdf, scache = state_results[mode]
                sub_keys = [k for k in keys_labeled if C.parse_key(k).group == g]
                Ps = []
                for k in sub_keys:
                    f, P, nwin = scache.get((k, state), (C.STATE_F, None, 0))
                    if P is not None and nwin > 0:
                        Ps.append(P)
                if not Ps:
                    continue
                meanP = np.mean(Ps, axis=0)
                ax.plot(C.STATE_F, meanP, color=C.GROUP_COL[g], lw=2 if mode == "manual+ml" else 1,
                       ls=ls, alpha=alpha_line,
                       label=f"{C.GROUP_LABEL[g]} {mode} (n={len(Ps)} cyc)")
        ax.axvspan(*C.ULTRA_SLOW, color="grey", alpha=0.25, lw=0)
        ax.text(np.sqrt(C.ULTRA_SLOW[0] * C.ULTRA_SLOW[1]), 0.95, "ultra-slow\n(NOT resolved\nat 60 s)",
               transform=ax.get_xaxis_transform(), fontsize=7, ha="center", va="top", color="dimgrey")
        ax.axvspan(*C.VASOMOTION, color="gold", alpha=0.15, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.004, 0.5)
        ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("arteriole PSD")
        ax.set_title(f"7D. {state} spectra, SCR vs KD\n(solid=manual+ml primary, dashed=manual sensitivity)",
                    fontsize=8.5)
        ax.legend(fontsize=6, loc="lower left")

    # 7E: vasomotion band power by state x group, 2x2 mouse-level dots + LMM
    ax7e = fig2.add_subplot(gs2[0, 2])
    mode = "manual+ml"
    keys_labeled, sdf, scache = state_results[mode]
    mdf_state = sdf.groupby(["group", "mouse", "state"], as_index=False)["art_vasomotion"].mean()
    pos = {("SCR", "Awake"): 0, ("KD", "Awake"): 1, ("SCR", "NREM"): 2.2, ("KD", "NREM"): 3.2}
    for (g, st), xp in pos.items():
        sub = mdf_state[(mdf_state.group == g) & (mdf_state.state == st)]
        jitter = (np.random.default_rng(2).random(len(sub)) - 0.5) * 0.15
        ax7e.scatter(xp + jitter, sub["art_vasomotion"], color=C.GROUP_COL[g], s=45,
                    marker="o" if st == "Awake" else "^", label=f"{g}-{st} (n={len(sub)})")
    ax7e.set_xticks([0.5, 2.7]); ax7e.set_xticklabels(["Awake", "NREM"])
    ax7e.set_ylabel("log10 vasomotion power (art)")
    ax7e.set_title("7E. Vasomotion power, state × group\n(mouse-level; manual+ml)", fontsize=9)
    ax7e.legend(fontsize=6, ncol=2)
    lmm_e = C.lmm(sdf, "art_vasomotion", "C(group)*C(state)")
    if lmm_e is not None:
        C.save_table(lmm_e, "fig7_E_lmm_vasomotion_state_group")
    C.save_table(mdf_state, "fig7_E_mouse_state_group_vasomotion")

    # ultra_slow greyed note on 7E panel (not resolvable at 60s) -- separate small axis
    ax7e.text(0.02, 0.02, "ultra-slow band power at state level not shown:\nnot resolvable with 60-s windows",
             transform=ax7e.transAxes, fontsize=6.5, color="dimgrey", va="bottom")

    # 7F: time-domain SD of band-passed arteriole, whole record and per state
    ax7f_bands = {}
    for i, bname in enumerate(BAND_NAMES):
        ax = fig2.add_subplot(gs2[1, i])
        rows = []
        C.set_label_mode("manual+ml")
        for k in whole_keys:
            cyc = C.load_cycle(k)
            if "art" not in cyc.sig:
                continue
            t, y = cyc.sig["art"]
            fs = cyc.fs["art"]
            whole_sd = sd_by_state_concat(t, y, fs, cyc.labels, C.BANDS[bname], state=None)
            row = dict(key=k, mouse=cyc.info.mouse, group=cyc.info.group, sd_whole=whole_sd)
            for st in ("Awake", "NREM"):
                row[f"sd_{st}"] = sd_by_state_concat(t, y, fs, cyc.labels, C.BANDS[bname], state=st)
            rows.append(row)
        fdf = pd.DataFrame(rows)
        mfdf = fdf.groupby(["group", "mouse"], as_index=False)[["sd_whole", "sd_Awake", "sd_NREM"]].mean()
        xpos = {"sd_whole": 0, "sd_Awake": 1.3, "sd_NREM": 2.6}
        for col_, xc in xpos.items():
            for g in GROUPS:
                sub = mfdf[mfdf.group == g]
                dx = -0.15 if g == "SCR" else 0.15
                jitter = (np.random.default_rng(3).random(len(sub)) - 0.5) * 0.1
                ax.scatter(xc + dx + jitter, sub[col_], color=C.GROUP_COL[g], s=40,
                          label=f"{g}" if col_ == "sd_whole" else None)
        ax.set_xticks(list(xpos.values())); ax.set_xticklabels(["whole", "Awake", "NREM"])
        ax.set_ylabel(f"SD band-passed art.\n({bname})")
        gt_whole = C.group_test(mfdf.loc[mfdf.group == "SCR", "sd_whole"], mfdf.loc[mfdf.group == "KD", "sd_whole"])
        ax.set_title(f"7F. {bname} SD (whole/state)\nwhole Welch p={gt_whole['welch_p']:.3f}", fontsize=8.5)
        if i == 0:
            ax.legend(fontsize=7)
        C.save_table(fdf, f"fig7_F_{bname}_sd_bystate_bycycle")
        C.save_table(mfdf, f"fig7_F_{bname}_sd_bystate_bymouse")
        ax7f_bands[bname] = mfdf

    # 7G: Ca ultra-slow/vasomotion power SCR vs KD + arteriole-Ca coherence per band per mouse
    ax7g1 = fig2.add_subplot(gs2[1, 2])
    for i, bname in enumerate(BAND_NAMES):
        mdf_ca = mouse_level(whole_df.dropna(subset=[f"ca_{bname}"]), [f"ca_{bname}"])
        for g in GROUPS:
            sub = mdf_ca[mdf_ca.group == g]
            x = i * 2.5 + (0 if g == "SCR" else 1)
            jitter = (np.random.default_rng(4).random(len(sub)) - 0.5) * 0.15
            ax7g1.scatter(x + jitter, sub[f"ca_{bname}"], color=C.GROUP_COL[g], s=40,
                        label=f"{g}" if i == 0 else None)
    ax7g1.set_xticks([0.5, 3.0]); ax7g1.set_xticklabels(BAND_NAMES)
    ax7g1.set_ylabel("log10 Ca power")
    ax7g1.set_title("7G(i). Astro Ca²⁺ band power, SCR vs KD\n(mouse-level)", fontsize=9)
    ax7g1.legend(fontsize=7)

    # coherence
    ax7g2 = fig2.add_subplot(gs2[2, 2])
    coh_rows = []
    for k in whole_keys:
        cyc = C.load_cycle(k)
        if "art" not in cyc.sig or "ca" not in cyc.sig:
            continue
        _, ya = cyc.sig["art"]
        _, yc = cyc.sig["ca"]
        fs = cyc.fs["art"]
        n = min(len(ya), len(yc))
        ya2, yc2 = ya[:n], yc[:n]
        fin = np.isfinite(ya2) & np.isfinite(yc2)
        if fin.sum() < 256:
            continue
        idx = np.arange(n)
        ya3 = ya2.copy(); yc3 = yc2.copy()
        ya3[~fin] = np.interp(idx[~fin], idx[fin], ya2[fin])
        yc3[~fin] = np.interp(idx[~fin], idx[fin], yc2[fin])
        f_coh, Cxy = sp_coherence(ya3, yc3, fs=fs, nperseg=256)
        row = dict(key=k, mouse=cyc.info.mouse, group=cyc.info.group)
        for bname, band in C.BANDS.items():
            m = (f_coh >= band[0]) & (f_coh < band[1])
            row[f"coh_{bname}"] = float(np.mean(Cxy[m])) if m.sum() else np.nan
        coh_rows.append(row)
    coh_df = pd.DataFrame(coh_rows)
    for i, bname in enumerate(BAND_NAMES):
        mdf_coh = mouse_level(coh_df.dropna(subset=[f"coh_{bname}"]), [f"coh_{bname}"])
        for g in GROUPS:
            sub = mdf_coh[mdf_coh.group == g]
            x = i * 2.5 + (0 if g == "SCR" else 1)
            jitter = (np.random.default_rng(5).random(len(sub)) - 0.5) * 0.15
            ax7g2.scatter(x + jitter, sub[f"coh_{bname}"], color=C.GROUP_COL[g], s=40,
                        label=f"{g}" if i == 0 else None)
    ax7g2.set_xticks([0.5, 3.0]); ax7g2.set_xticklabels(BAND_NAMES)
    ax7g2.set_ylabel("mean coherence (art-Ca)")
    ax7g2.set_ylim(0, 1)
    ax7g2.set_title("7G(ii). Arteriole-Ca coherence (nperseg=256)\nper band, mouse-level", fontsize=9)
    ax7g2.legend(fontsize=7)
    C.save_table(coh_df, "fig7_G_coherence_bycycle")

    # 7H: controls per mouse, group comparisons, and LMM with %NREM covariate
    ax7h = fig2.add_subplot(gs2[2, 0:2])
    ax7h.axis("off")
    C.set_label_mode("manual+ml")
    ctrl_rows = []
    for k in whole_keys:
        cyc = C.load_cycle(k)
        lab = cyc.labels
        valid = lab[lab != None]  # noqa: E711
        pct_nrem_v = 100.0 * np.mean(valid == "NREM") if len(valid) else np.nan
        n_nrem_win = len(C.state_windows(lab, "NREM"))
        art_t, art_y = cyc.sig["art"] if "art" in cyc.sig else (None, None)
        med_diam = np.nan
        if art_t is not None:
            raw_t, raw_area = C.read_trace(k, "arteriole-diameter")
            if raw_area is not None:
                med_diam = float(np.nanmedian(np.sqrt(np.clip(raw_area, 0, None))))
        ctrl_rows.append(dict(key=k, mouse=cyc.info.mouse, group=cyc.info.group,
                              duration_s=cyc.duration, pct_nrem=pct_nrem_v,
                              n_nrem_windows=n_nrem_win, median_diam_px=med_diam))
    ctrl_df = pd.DataFrame(ctrl_rows)
    rate_df = event_rate_table(whole_keys)
    ctrl_mouse = ctrl_df.groupby(["group", "mouse"], as_index=False).agg(
        duration_s=("duration_s", "sum"), pct_nrem=("pct_nrem", "mean"),
        n_nrem_windows=("n_nrem_windows", "sum"), median_diam_px=("median_diam_px", "median"))
    ctrl_mouse = ctrl_mouse.merge(rate_df[["group", "mouse", "n2w_ma_per_h"]], on=["group", "mouse"], how="left")
    C.save_table(ctrl_mouse, "fig7_H_controls_bymouse")

    ctrl_tests = []
    for col_ in ["duration_s", "pct_nrem", "n_nrem_windows", "median_diam_px", "n2w_ma_per_h"]:
        a = ctrl_mouse.loc[ctrl_mouse.group == "SCR", col_]
        b = ctrl_mouse.loc[ctrl_mouse.group == "KD", col_]
        gt = C.group_test(a, b)
        ctrl_tests.append(dict(variable=col_, **gt))
    ctrl_tests_df = pd.DataFrame(ctrl_tests)
    C.save_table(ctrl_tests_df, "fig7_H_controls_group_test")

    # covariate LMM: ultra_slow power ~ group + pct_nrem
    whole_df_pct = whole_df.merge(ctrl_df[["key", "pct_nrem"]], on="key", how="left")
    lmm_cov = C.lmm(whole_df_pct, "art_ultra_slow", "C(group) + pct_nrem")
    if lmm_cov is not None:
        C.save_table(lmm_cov, "fig7_H_lmm_ultraslow_group_pctnrem")

    tbl_txt = ctrl_mouse.round(3).to_string(index=False)
    ax7h.text(0, 1, "7H. Per-mouse controls (SCR vs KD)", fontsize=10, weight="bold", va="top",
             transform=ax7h.transAxes)
    ax7h.text(0, 0.9, tbl_txt, fontsize=6.5, family="monospace", va="top", transform=ax7h.transAxes)
    cov_line = "LMM FIT_FAILED" if lmm_cov is None or "FIT_FAILED" in str(lmm_cov.get("term", [""]).iloc[0]) else "see table"
    ax7h.text(0, 0.08,
             f"Group tests (Welch p / MW p): " +
             "; ".join(f"{r.variable}: {r.welch_p:.3f}/{r.mw_p:.3f}" for r in ctrl_tests_df.itertuples()),
             fontsize=6.5, va="top", transform=ax7h.transAxes)
    ax7h.text(0, 0.0, f"ultra_slow ~ group + pct_nrem LMM: {cov_line}", fontsize=6.5, va="top",
             transform=ax7h.transAxes)

    # event-rate confound panel
    ax_conf = fig2.add_subplot(gs2[2, 2:3]) if False else fig2.add_subplot(gs2[3, 0])
    for g in GROUPS:
        sub = rate_df[rate_df.group == g]
        x = 0 if g == "SCR" else 1
        jitter = (np.random.default_rng(6).random(len(sub)) - 0.5) * 0.15
        ax_conf.scatter(x + jitter, sub["n2w_ma_per_h"], color=C.GROUP_COL[g], s=45,
                       label=f"{g} (n={len(sub)} days)")
    ax_conf.set_xticks([0, 1]); ax_conf.set_xticklabels(["SCR", "KD"])
    ax_conf.set_ylabel("(N2W+MA) / h")
    gt_rate = C.group_test(rate_df.loc[rate_df.group == "SCR", "n2w_ma_per_h"],
                           rate_df.loc[rate_df.group == "KD", "n2w_ma_per_h"])
    ax_conf.set_title(f"Event-rate confound: N2W+MA rate\nWelch p={gt_rate['welch_p']:.3f}, MW p={gt_rate['mw_p']:.3f}",
                      fontsize=8.5)
    ax_conf.legend(fontsize=7)
    C.save_table(rate_df, "fig7_eventrate_confound_bymouse")
    C.save_table(pd.DataFrame([dict(**gt_rate)]), "fig7_eventrate_confound_group_test")

    ax_cav = fig2.add_subplot(gs2[3, 1:3])
    ax_cav.axis("off")
    ax_cav.text(0, 1,
               "CAVEAT: arousal recurrence (~0.006-0.008 Hz, i.e. N2W/MA cadence) falls inside the\n"
               "ultra-slow band (0.005-0.04 Hz). Lower ultra-slow power in KD could partly reflect\n"
               "fewer/weaker arousal events rather than a change in intrinsic oscillation amplitude.\n"
               "See N2W+MA rate panel (left) and the pct_nrem-covariate LMM (7H) for partial control.",
               fontsize=8.5, va="top", transform=ax_cav.transAxes, color="firebrick",
               bbox=dict(boxstyle="round", fc="#fff3e0", ec="firebrick", alpha=0.8))

    fig2.suptitle(f"Fig 7D-H — state-specific spectra, band power, time-domain, Ca/coherence, controls\n{C.PRELIM_STAMP}",
                 fontsize=10)
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    C.save_fig(fig2, "fig7_D-H_state_and_controls")
    plt.close(fig2)

    # ----------------------------------------------------------------- #
    # Notes
    # ----------------------------------------------------------------- #
    def fmt_gt(gt, label):
        return (f"{label}: SCR mean={gt['mean_a']:.3f}, KD mean={gt['mean_b']:.3f}, "
               f"n_SCR={gt['n_a']}, n_KD={gt['n_b']}, Hedges g={gt['hedges_g']:.2f}, "
               f"Welch p={gt['welch_p']:.3f}, MW p={gt['mw_p']:.3f}")

    us_gt = next(r for r in tables["fig7_C_band_power_group_test"] if r["band"] == "ultra_slow")
    vm_gt = next(r for r in tables["fig7_C_band_power_group_test"] if r["band"] == "vasomotion")
    ratio_us = 10 ** (us_gt["mean_b"] - us_gt["mean_a"])
    ratio_vm = 10 ** (vm_gt["mean_b"] - vm_gt["mean_a"])

    md = []
    md.append("# Figure 7 — IP3R2 KD vs scramble, ultra-slow / vasomotion arteriole power\n")
    md.append("## Panel summary\n")
    md.append(f"- **7A** representative traces: {best_pair[0] if best_pair else 'N/A'} (SCR) vs "
             f"{best_pair[1] if best_pair else 'N/A'} (KD), matched to |%NREM diff|="
             f"{best_diff:.1f} pts; shows pupil z, whisker z, arteriole dD/D0, Ca dF/F0, and "
             "ultra-slow band-passed arteriole, state-shaded. (`fig7_A_examples`)")
    md.append(f"- **7B** whole-record arteriole PSD, mouse-mean±SEM, log-log, bands shaded, faint "
             f"per-mouse traces underneath. Pool: SCR {n_whole_cyc.get('SCR')} cyc/{n_whole.get('SCR')} days, "
             f"KD {n_whole_cyc.get('KD')} cyc/{n_whole.get('KD')} days — no labels used. (`fig7_B-C_whole`)")
    md.append(f"- **7C** band power per mouse (log10): {fmt_gt(us_gt, 'ultra_slow')}; ratio KD/SCR="
             f"{ratio_us:.2f}x. {fmt_gt(vm_gt, 'vasomotion')}; ratio KD/SCR={ratio_vm:.2f}x. "
             "LMM (cycle-level log10 power ~ C(group), random intercept mouse) in `fig7_C_lmm.csv`. "
             "Frequency-resolved Welch-p per bin with BH-FDR in `fig7_C_freq_resolved_group_test.csv` "
             "and shown as the strip panel under 7B.")
    md.append("- **7D** state-specific (60-s window) arteriole PSD, Awake and NREM, SCR vs KD, run in "
             "both `manual+ml` (solid, primary) and `manual` (dashed, sensitivity) label modes; "
             "ultra-slow is NOT resolvable at 60 s (marked grey/text on panel).")
    md.append("- **7E** vasomotion band power by state×group, mouse-level 2x2 dots (manual+ml); "
             "LMM (log10 power ~ C(group)*C(state), random intercept mouse) in "
             "`fig7_E_lmm_vasomotion_state_group.csv`; mouse-level values in "
             "`fig7_E_mouse_state_group_vasomotion.csv`. Ultra-slow at state level intentionally omitted "
             "(not resolvable at 60 s).")
    md.append("- **7F** SD of band-passed arteriole per mouse (whole record, Awake-concat, NREM-concat), "
             "both bands; per-band tables `fig7_F_<band>_sd_bystate_bycycle/bymouse.csv`.")
    md.append("- **7G** astro Ca2+ ultra-slow/vasomotion band power SCR vs KD (mouse-level, whole record), "
             "and arteriole-Ca coherence (`scipy.signal.coherence`, nperseg=256) averaged per band, per mouse "
             "(`fig7_G_coherence_bycycle.csv`).")
    md.append("- **7H** per-mouse controls: recording duration, %NREM, number of usable NREM 60-s windows, "
             "median raw arteriole diameter (sqrt area, from `C.read_trace`), and N2W+MA rate; group tests in "
             "`fig7_H_controls_group_test.csv`, raw values in `fig7_H_controls_bymouse.csv`. Covariate LMM "
             "(ultra_slow log10 power ~ C(group) + pct_nrem) in `fig7_H_lmm_ultraslow_group_pctnrem.csv` "
             "(or FIT_FAILED noted there if it could not converge).")
    md.append(f"- **Event-rate confound** (explicit caveat): N2W+MA combined rate per mouse "
             f"(SCR mean={gt_rate['mean_a']:.3f}/h, KD mean={gt_rate['mean_b']:.3f}/h, "
             f"Welch p={gt_rate['welch_p']:.3f}, MW p={gt_rate['mw_p']:.3f}) reported alongside ultra-slow "
             "power because arousal recurrence (~0.006-0.008 Hz) falls inside the ultra-slow band "
             "(0.005-0.04 Hz); a lower ultra-slow power in KD could partly reflect fewer/weaker events "
             "rather than purely intrinsic oscillation amplitude. See `fig7_eventrate_confound_*.csv`.")

    md.append("\n## Panels not fully resolvable / caveats\n")
    md.append("- State-specific ultra-slow power is NOT shown as a quantitative panel: 60-s state windows "
             "cannot resolve 0.005-0.04 Hz (fewer than ~2.4 cycles fit in the window); only vasomotion is "
             "quantified at the state level, per the brief.")
    md.append("- n is small (SCR 4 days, KD 6 days) — per `AGENT_BRIEF.md`, a Wilcoxon/MW test at this n "
             "cannot reach very small p; see `wilcoxon_min_p` logic in common.py. All tests reported here "
             "are Welch t / Mann-Whitney (unpaired, `C.group_test`) plus LMM with mouse random intercepts; "
             "none should be read as definitive given n.")
    md.append("- One KD day (`IP3R2_KD_2025.10.25`, cycle10) has no arteriole trace and is excluded from the "
             "whole-record pool (18/19 KD cycles with arteriole used, matching the brief's '18 cycles/6 days').")
    md.append("- `IP3R2_KD_2023.08.26` manual labels are used for the manual-mode sensitivity check; ML "
             "predictions are excluded on that day throughout (per `ML_EXCLUDED_DAYS` in common.py) because "
             "the ML model failed leave-one-day-out validation there.")
    md.append("- Ca channel missing for 1 KD cycle+1 day-cycle combo lacking calcium "
             "(`IP3R2_KD_2023.08.26_cycle4`), reducing n for 7G band-power/coherence slightly below the "
             "whole-record arteriole n; see n reported directly on 7G panels/tables.")

    md.append("\n## Reproduce\n")
    md.append("```\n./venv/bin/python -B scripts/figs_prelim/fig7.py\n```")

    (C.OUT / "notes" / "fig7.md").write_text("\n".join(md) + "\n")

    print("\n".join(notes))
    print("Wrote fig7_A_examples, fig7_B-C_whole, fig7_D-H_state_and_controls + tables + notes/fig7.md")
    print(f"ultra_slow: ratio KD/SCR = {ratio_us:.2f}x, Welch p={us_gt['welch_p']:.3f}, MW p={us_gt['mw_p']:.3f}, "
         f"n_SCR={us_gt['n_a']} n_KD={us_gt['n_b']}")
    print(f"vasomotion: ratio KD/SCR = {ratio_vm:.2f}x, Welch p={vm_gt['welch_p']:.3f}, MW p={vm_gt['mw_p']:.3f}, "
         f"n_SCR={vm_gt['n_a']} n_KD={vm_gt['n_b']}")
    print(f"N2W+MA rate: SCR={gt_rate['mean_a']:.3f}/h KD={gt_rate['mean_b']:.3f}/h p={gt_rate['welch_p']:.3f}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()
