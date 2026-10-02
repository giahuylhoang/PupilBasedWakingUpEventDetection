"""Reserve figures: rare REM-like events, GRAB-NE descriptive dataset, and a
one-page coverage summary.

Outputs (data/RESULTS_figures_prelim_2026-10-01/):
  figures/figR_REM.png / .pdf          - per-event small multiples, REM onset
  tables/figR_REM_events.csv           - descriptive per-event table
  figures/figR_NE.png / .pdf           - GRAB-NE descriptive panels
  tables/figR_NE_state.csv             - NE by state, paired per mouse
  tables/figR_NE_xcorr.csv             - NE-arteriole cross-correlation per mouse
  tables/figR_NE_band.csv              - band power per mouse (whole record)
  figures/figR_coverage.png / .pdf     - dataset summary panel
  tables/figR_coverage.csv             - dataset summary table
  notes/fig_reserve.md

Labels: C.set_label_mode('manual') throughout, except the coverage panel's
label-source column which briefly switches to 'manual+ml' to show what the
main figures use, then switches back.

Run: ./venv/bin/python -B -m scripts.figs_prelim.fig_reserve   (from repo root)
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import coherence as sp_coherence
from scipy.signal import correlate, correlation_lags

sys.path.insert(0, ".")
from scripts.figs_prelim import common as C

C.set_label_mode("manual")

REM_GROUPS = ("WT", "SCR", "NE")
NE_GROUP = "NE"


# --------------------------------------------------------------------------- #
# helpers (local to this script only)
# --------------------------------------------------------------------------- #
def _seg_mean(t, y, t0, t1):
    m = (t >= t0) & (t < t1)
    v = y[m]
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else np.nan


def _rem_bout_duration(cyc, t0):
    """Duration (s) of the REM bout that starts at t0, from the label bouts."""
    b = C.bouts(cyc.labels)
    hit = b[(b["state"] == "REM") & (np.isclose(b["t_start"], t0))]
    if hit.empty:
        return np.nan
    return float(hit["dur_s"].iloc[0])


def _preceding_nrem_window(t0, dur_s=30.0):
    return (max(0.0, t0 - dur_s), t0)


def _cycle_minutes_by_state(cyc):
    out = {}
    for s in ("Awake", "NREM", "REM", "Unclear"):
        n_bins = int(np.sum(np.asarray(cyc.labels, dtype=object) == s))
        out[s] = n_bins * C.BIN / 60.0
    return out


# --------------------------------------------------------------------------- #
# Figure 1: figR_REM
# --------------------------------------------------------------------------- #
def make_fig_rem():
    keys = C.keys_for(REM_GROUPS)
    ev = C.event_table(keys)
    rem = ev[ev["type"] == "REM"].reset_index(drop=True)

    # total labelled time and total REM time across these groups
    total_bins = 0
    rem_bins = 0
    for k in keys:
        cyc = C.load_cycle(k)
        lab = np.asarray(cyc.labels, dtype=object)
        total_bins += int(np.sum(np.isin(lab, list(C.STATES) + ["Unclear"])))
        rem_bins += int(np.sum(lab == "REM"))
    total_rem_s = rem_bins * C.BIN
    total_labelled_s = total_bins * C.BIN
    pct_rem = 100.0 * rem_bins / total_bins if total_bins else np.nan

    signals = ["pupil", "eye", "whisk", "art", "ca", "ne"]
    rows = []
    n_ev = len(rem)
    ncols = 4
    nrows = int(np.ceil(n_ev / ncols)) if n_ev else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.0 * nrows),
                             squeeze=False)

    for i, r in enumerate(rem.itertuples()):
        ax = axes[i // ncols][i % ncols]
        cyc = C.load_cycle(r.key)
        dur = _rem_bout_duration(cyc, r.t0)
        pre0, pre1 = _preceding_nrem_window(r.t0, 30.0)

        row = dict(group=r.group, mouse=r.mouse, key=r.key, t0=r.t0,
                  rem_dur_s=dur)
        plotted_any = False
        clipped_sigs = []
        for sig in signals:
            if sig not in cyc.sig:
                row[f"{sig}_rem_mean"] = np.nan
                row[f"{sig}_prenrem_mean"] = np.nan
                continue
            t, y = cyc.sig[sig]
            fs = C.EPOCH_FS[sig]
            rg = C.rel_grid(fs)
            v = C.epoch(t, y, r.t0, fs, baseline=False)
            if np.isfinite(v).any():
                # whisker spikes can be >10x pupil/arteriole scale; clip the
                # *display* only (never the stored mean) so slower signals
                # stay visible. Clip at the 99th percentile of |v| for pupil
                # and whisker, which are the usual outliers here.
                v_disp = v.copy()
                if sig in ("whisk", "pupil", "eye"):
                    thresh = np.nanpercentile(np.abs(v_disp), 99)
                    if np.isfinite(thresh) and thresh > 0:
                        n_clip = int(np.sum(np.abs(v_disp) > thresh))
                        if n_clip > 0:
                            clipped_sigs.append(f"{sig}({n_clip})")
                        v_disp = np.clip(v_disp, -thresh, thresh)
                ax.plot(rg, v_disp, color=C.SIG_COL[sig], lw=1.1,
                       label=C.SIG_LABEL[sig])
                plotted_any = True
            row[f"{sig}_rem_mean"] = _seg_mean(t, y, r.t0, r.t0 + max(dur, 5.0))
            row[f"{sig}_prenrem_mean"] = _seg_mean(t, y, pre0, pre1)
        rows.append(row)

        ax.axvline(0, color="k", lw=0.8, ls="--")
        if np.isfinite(dur):
            ax.axvspan(0, dur, color=C.STATE_COL["REM"], alpha=0.15, lw=0)
        title = (f"{C.GROUP_LABEL.get(r.group, r.group)} · {r.mouse.split('_')[-1]}\n"
                f"t0={r.t0:.0f}s dur={dur:.0f}s")
        ax.set_title(title, fontsize=8)
        ax.set_xlim(-30, 60)
        ax.tick_params(labelsize=7)
        if clipped_sigs:
            ax.text(0.98, 0.02, "clipped@p99: " + ",".join(clipped_sigs),
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=5.5, color="gray")
        if i % ncols == 0:
            ax.set_ylabel("z / ΔF/F₀ / ΔD/D₀", fontsize=7)
        if i // ncols == nrows - 1:
            ax.set_xlabel("time from REM onset (s)", fontsize=7)
        if not plotted_any:
            ax.text(0.5, 0.5, "no signal", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")

    # turn off unused axes, add one shared legend
    for j in range(n_ev, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles, labels = [], []
    for sig in signals:
        handles.append(plt.Line2D([0], [0], color=C.SIG_COL[sig], lw=1.5))
        labels.append(C.SIG_LABEL[sig])
    fig.legend(handles, labels, loc="lower center", ncol=len(signals),
              fontsize=8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Reserve: all REM-like events, {', '.join(REM_GROUPS)} (n={n_ev} events)\n"
        f"Total REM time {total_rem_s:.0f} s = {pct_rem:.2f}% of {total_labelled_s/60:.0f} min labelled time. "
        f"Descriptive only, no inferential statistics.\n{C.PRELIM_STAMP}",
        fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    C.save_fig(fig, "figR_REM")
    plt.close(fig)

    tab = pd.DataFrame(rows)
    C.save_table(tab, "figR_REM_events")
    return dict(n_events=n_ev, total_rem_s=total_rem_s, pct_rem=pct_rem,
               total_labelled_min=total_labelled_s / 60.0, table=tab)


# --------------------------------------------------------------------------- #
# Figure 2: figR_NE
# --------------------------------------------------------------------------- #
def make_fig_ne():
    keys = C.keys_for(NE_GROUP)
    days = sorted(set(C.load_cycle(k).info.mouse for k in keys))
    ev = C.response_events(keys)   # NE ETAs: pupil-anchored, non-overlapping (v2)

    fig = plt.figure(figsize=(16, 15))   # taller event-average rows
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.5, 1.0], hspace=0.5, wspace=0.42)

    # Panel A: example trace with state shading (pupil, whisk, art, ne)
    axA = fig.add_subplot(gs[0, :])
    # pick the cycle with the most REM/MA/N2W variety and all channels
    best_key, best_score = None, -1
    for k in keys:
        cyc = C.load_cycle(k)
        if not {"pupil", "whisk", "art", "ne"}.issubset(cyc.sig):
            continue
        score = int(np.sum(np.asarray(cyc.labels, dtype=object) != None))
        if score > best_score:
            best_score, best_key = score, k
    cyc = C.load_cycle(best_key)
    shade_sigs = ["pupil", "whisk", "art", "ne"]
    offsets = {}
    off = 0.0
    for sig in shade_sigs:
        t, y = cyc.sig[sig]
        yn = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)
        axA.plot(t, yn + off, color=C.SIG_COL[sig], lw=0.8, label=C.SIG_LABEL[sig])
        offsets[sig] = off
        off -= 6.0
    C.shade_states(axA, cyc.labels, alpha=0.15)
    axA.set_yticks(list(offsets.values()))
    axA.set_yticklabels([C.SIG_LABEL[s] for s in shade_sigs], fontsize=8)
    axA.set_xlabel("time (s)")
    axA.set_title(f"A. Example cycle {best_key} — pupil/whisker/arteriole/NE with manual state shading",
                 fontsize=9)
    axA.legend(loc="upper right", fontsize=6, ncol=4, framealpha=0.5)

    # Panel B: NE and arteriole event-triggered averages for N2W and MA (mouse-level)
    axB1 = fig.add_subplot(gs[1, 0])
    axB2 = fig.add_subplot(gs[1, 1])
    etrig_rows = []
    for ax, etype in zip((axB1, axB2), ("N2W", "MA")):
        sub = ev[ev["type"] == etype]
        n_ev_e = len(sub)
        n_mice_e = sub["mouse"].nunique() if n_ev_e else 0
        for sig, ls in (("ne", "-"), ("art", "--")):
            rg, M = C.epochs_for(sub, sig)
            if M.size == 0 or not np.isfinite(M).any():
                continue
            # mouse-level mean first
            mouse_means = []
            for m in sub["mouse"].unique():
                idx = np.flatnonzero((sub["mouse"] == m).to_numpy())
                row_m = np.nanmean(M[idx], axis=0)
                mouse_means.append(row_m)
            mouse_means = np.array(mouse_means)
            grand = np.nanmean(mouse_means, axis=0)
            sem = (np.nanstd(mouse_means, axis=0, ddof=1) / np.sqrt(max(mouse_means.shape[0], 1))
                  if mouse_means.shape[0] > 1 else np.zeros_like(grand))
            ax.plot(rg, grand, color=C.SIG_COL[sig], ls=ls, lw=1.6,
                   label=f"{C.SIG_LABEL[sig]} (n={mouse_means.shape[0]} days)")
            ax.fill_between(rg, grand - sem, grand + sem, color=C.SIG_COL[sig], alpha=0.2, lw=0)
            etrig_rows.append(dict(event=etype, signal=sig, n_events=n_ev_e,
                                   n_mice=n_mice_e))
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.set_title(f"B. {C.EVENT_LABEL[etype]} (n={n_ev_e} ev, {n_mice_e}/{len(days)} days)", fontsize=8)
        ax.set_xlabel("time from onset (s)", fontsize=8)
        ax.set_ylabel("baseline-corrected", fontsize=8)
        ax.legend(fontsize=6.5)
        ax.tick_params(labelsize=7)
        if n_mice_e < len(days):
            ax.text(0.02, 0.02, f"note: {len(days)-n_mice_e} day(s) had no {etype} events",
                    transform=ax.transAxes, fontsize=6, color="gray")

    # Panel C: NE by state (Awake vs NREM), paired per mouse
    axC = fig.add_subplot(gs[1, 2])
    state_rows = []
    for k in keys:
        cyc = C.load_cycle(k)
        if "ne" not in cyc.sig:
            continue
        t, y = cyc.sig["ne"]
        bb = C.bouts(cyc.labels)
        for st in ("Awake", "NREM"):
            vals = []
            for b in bb[bb["state"] == st].itertuples():
                v = _seg_mean(t, y, b.t_start, b.t_start + b.dur_s)
                if np.isfinite(v):
                    vals.append(v)
            if vals:
                state_rows.append(dict(mouse=cyc.info.mouse, key=k, state=st,
                                       mean_ne=np.mean(vals)))
    state_df = pd.DataFrame(state_rows)
    mouse_state = (state_df.groupby(["mouse", "state"], as_index=False)["mean_ne"].mean()
                  if not state_df.empty else pd.DataFrame())
    pivoted = (mouse_state.pivot(index="mouse", columns="state", values="mean_ne")
              if not mouse_state.empty else pd.DataFrame())
    paired_res = dict(n=0)
    if not pivoted.empty and {"Awake", "NREM"}.issubset(pivoted.columns):
        pivoted = pivoted.dropna(subset=["Awake", "NREM"])
        for mouse, r in pivoted.iterrows():
            axC.plot([0, 1], [r["Awake"], r["NREM"]], "-o", color=C.GROUP_COL["NE"],
                     alpha=0.8, ms=5)
        paired_res = C.paired_test(pivoted["Awake"].to_numpy(), pivoted["NREM"].to_numpy())
    axC.set_xticks([0, 1]); axC.set_xticklabels(["Awake", "NREM"])
    axC.set_ylabel("NE ΔF/F₀ (mean per bout, mouse mean)", fontsize=7.5)
    axC.set_title(f"C. NE by state, paired per day (n={paired_res.get('n', 0)} days)", fontsize=8)
    axC.tick_params(labelsize=7)
    nd = paired_res.get("n", 0)
    axC.text(0.5, -0.28,
             f"wilcoxon p={paired_res.get('wilcoxon_p', np.nan):.3f} "
             f"(min attainable p={paired_res.get('wilcoxon_min_p', np.nan):.3f} at n={nd})",
             transform=axC.transAxes, ha="center", fontsize=6.5)

    # Panel D: NE-arteriole cross-correlation per mouse, whole / Awake / NREM
    axD = fig.add_subplot(gs[2, 0])
    xcorr_rows = []
    max_lag_s = 30.0
    for k in keys:
        cyc = C.load_cycle(k)
        if not {"ne", "art"}.issubset(cyc.sig):
            continue
        t_ne, y_ne = cyc.sig["ne"]
        t_art, y_art = cyc.sig["art"]
        fs = 1.0 / np.nanmedian(np.diff(t_ne))
        lab_bins = np.asarray(cyc.labels, dtype=object)
        bin_idx = np.clip((t_ne / C.BIN).astype(int), 0, len(lab_bins) - 1)
        state_mask_of_t = lab_bins[bin_idx]
        for cond, mask in (("whole", np.ones_like(y_ne, dtype=bool)),
                          ("Awake", state_mask_of_t == "Awake"),
                          ("NREM", state_mask_of_t == "NREM")):
            a = y_ne.copy(); b = y_art.copy()
            a[~mask] = np.nan; b[~mask] = np.nan
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 20:
                continue
            af = a[ok] - np.mean(a[ok])
            bf = b[ok] - np.mean(b[ok])
            if af.std() < 1e-9 or bf.std() < 1e-9:
                continue
            c = correlate(af, bf, mode="full") / (np.std(af) * np.std(bf) * len(af))
            lags = correlation_lags(len(af), len(bf)) / fs
            keep = np.abs(lags) <= max_lag_s
            c, lags = c[keep], lags[keep]
            j = np.argmax(np.abs(c))
            xcorr_rows.append(dict(mouse=cyc.info.mouse, key=k, condition=cond,
                                   peak_r=c[j], lag_s=lags[j], n_samples=int(ok.sum())))
    xcorr_df = pd.DataFrame(xcorr_rows)
    if not xcorr_df.empty:
        mouse_xcorr = xcorr_df.groupby(["mouse", "condition"], as_index=False)["peak_r"].mean()
        conds = ["whole", "Awake", "NREM"]
        xpos = {c: i for i, c in enumerate(conds)}
        for mouse, grp in mouse_xcorr.groupby("mouse"):
            xs = [xpos[c] for c in grp["condition"]]
            axD.plot(xs, grp["peak_r"], "-o", alpha=0.7, color=C.GROUP_COL["NE"], ms=5)
        axD.set_xticks(list(xpos.values())); axD.set_xticklabels(list(xpos.keys()))
    axD.axhline(0, color="k", lw=0.6)
    axD.set_ylabel("peak r (±30 s lag window)", fontsize=7.5)
    nd2 = xcorr_df["mouse"].nunique() if not xcorr_df.empty else 0
    axD.set_title(f"D. NE-arteriole peak cross-corr per day (n={nd2} days)", fontsize=8)
    axD.tick_params(labelsize=7)

    # Panel E: whole-record spectra NE & arteriole with ultra-slow band shaded
    axE = fig.add_subplot(gs[2, 1])
    specs_ne, specs_art = [], []
    for k in keys:
        cyc = C.load_cycle(k)
        if not {"ne", "art"}.issubset(cyc.sig):
            continue
        t_ne, y_ne = cyc.sig["ne"]
        fs_ne = 1.0 / np.nanmedian(np.diff(t_ne))
        f1, P1 = C.psd_whole(y_ne, fs_ne)
        t_art, y_art = cyc.sig["art"]
        fs_art = 1.0 / np.nanmedian(np.diff(t_art))
        f2, P2 = C.psd_whole(y_art, fs_art)
        if f1 is not None:
            specs_ne.append((f1, P1))
        if f2 is not None:
            specs_art.append((f2, P2))
    def _avg_spectra(specs):
        if not specs:
            return None, None
        fgrid = specs[0][0]
        acc = np.mean([np.interp(fgrid, f, P) for f, P in specs], axis=0)
        return fgrid, acc
    fN, PN = _avg_spectra(specs_ne)
    fA, PA = _avg_spectra(specs_art)
    if fN is not None:
        axE.semilogy(fN, PN, color=C.SIG_COL["ne"], label=f"NE (n={len(specs_ne)} cyc)")
    if fA is not None:
        axE.semilogy(fA, PA, color=C.SIG_COL["art"], label=f"arteriole (n={len(specs_art)} cyc)")
    axE.axvspan(*C.ULTRA_SLOW, color="gray", alpha=0.2, lw=0, label="ultra-slow band")
    axE.axvspan(*C.VASOMOTION, color="orange", alpha=0.12, lw=0, label="vasomotion band")
    axE.set_xlim(0, 0.3)
    axE.set_xlabel("frequency (Hz)", fontsize=8)
    axE.set_ylabel("PSD", fontsize=8)
    axE.set_title("E. Whole-record NE & arteriole spectra", fontsize=8)
    axE.legend(fontsize=6)
    axE.tick_params(labelsize=7)

    # Panel F: NE-arteriole coherence per band, per day
    axF = fig.add_subplot(gs[2, 2])
    coh_rows = []
    for k in keys:
        cyc = C.load_cycle(k)
        if not {"ne", "art"}.issubset(cyc.sig):
            continue
        t_ne, y_ne = cyc.sig["ne"]
        t_art, y_art = cyc.sig["art"]
        fs = 1.0 / np.nanmedian(np.diff(t_ne))
        a, b = y_ne.copy(), y_art.copy()
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 64:
            continue
        a, b = a[ok], b[ok]
        nperseg = min(256, len(a))
        f, Cxy = sp_coherence(a, b, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        for band_name, band in C.BANDS.items():
            m = (f >= band[0]) & (f < band[1])
            if m.sum() >= 1:
                coh_rows.append(dict(mouse=cyc.info.mouse, key=k, band=band_name,
                                     mean_coh=float(np.mean(Cxy[m]))))
    coh_df = pd.DataFrame(coh_rows)
    if not coh_df.empty:
        mouse_coh = coh_df.groupby(["mouse", "band"], as_index=False)["mean_coh"].mean()
        bands_order = list(C.BANDS.keys())
        xpos = {b: i for i, b in enumerate(bands_order)}
        for mouse, grp in mouse_coh.groupby("mouse"):
            xs = [xpos[b] for b in grp["band"]]
            axF.plot(xs, grp["mean_coh"], "-o", alpha=0.7, color=C.GROUP_COL["NE"], ms=5)
        axF.set_xticks(list(xpos.values())); axF.set_xticklabels(list(xpos.keys()), rotation=15)
    axF.set_ylim(0, 1)
    nd3 = coh_df["mouse"].nunique() if not coh_df.empty else 0
    axF.set_title(f"F. NE-arteriole coherence by band (n={nd3} days)", fontsize=8)
    axF.set_ylabel("mean coherence", fontsize=7.5)
    axF.tick_params(labelsize=7)

    fig.suptitle(
        f"Reserve: GRAB-NE descriptive dataset (group NE, {len(keys)} cycles / {len(days)} days, no calcium)\n"
        f"n = 3 recording days — purely descriptive, no formal inferential claims.\n{C.PRELIM_STAMP}",
        fontsize=9)
    C.save_fig(fig, "figR_NE")
    plt.close(fig)

    C.save_table(pd.DataFrame(etrig_rows), "figR_NE_etrig_n")
    C.save_table(mouse_state if not mouse_state.empty else pd.DataFrame(), "figR_NE_state")
    C.save_table(xcorr_df, "figR_NE_xcorr")
    C.save_table(coh_df, "figR_NE_coherence")

    return dict(n_days=len(days), n_cycles=len(keys), paired=paired_res,
               n_xcorr_days=nd2, n_coh_days=nd3)


# --------------------------------------------------------------------------- #
# Figure 3: figR_coverage
# --------------------------------------------------------------------------- #
def make_fig_coverage():
    groups = ["WT", "SCR", "KD", "NE"]
    rows = []
    for g in groups:
        C.set_label_mode("manual")
        keys_manual = C.keys_for(g)
        days_manual = sorted(set(C.load_cycle(k).info.mouse for k in keys_manual))

        # label source + minutes, using manual+ml (what the main figures use)
        C.set_label_mode("manual+ml")
        keys = C.keys_for(g)
        days = sorted(set(C.load_cycle(k).info.mouse for k in keys))
        src_counts = {}
        minutes_by_state = {"Awake": 0.0, "NREM": 0.0, "REM": 0.0, "Unclear": 0.0}
        mods_present = set()
        for k in keys:
            cyc = C.load_cycle(k)
            src = C.label_source(k)
            src_counts[src] = src_counts.get(src, 0) + 1
            m = _cycle_minutes_by_state(cyc)
            for s in minutes_by_state:
                minutes_by_state[s] += m[s]
            mods_present |= set(cyc.sig.keys())
        ev = C.event_table(keys)
        ev_counts = ev["type"].value_counts().to_dict() if not ev.empty else {}
        C.set_label_mode("manual")  # switch back immediately after this panel's data

        rows.append(dict(
            group=g, label=C.GROUP_LABEL.get(g, g),
            n_days=len(days), n_cycles=len(keys),
            min_awake=minutes_by_state["Awake"], min_nrem=minutes_by_state["NREM"],
            min_rem=minutes_by_state["REM"], min_unclear=minutes_by_state["Unclear"],
            label_sources=", ".join(f"{k}:{v}" for k, v in sorted(src_counts.items())),
            n2w=ev_counts.get("N2W", 0), ma=ev_counts.get("MA", 0),
            w2n=ev_counts.get("W2N", 0), rem=ev_counts.get("REM", 0),
            modalities=", ".join(sorted(mods_present)),
        ))
    cov = pd.DataFrame(rows)
    C.save_table(cov, "figR_coverage")

    # one-page panel: a table rendered as a figure (what a PI reads first)
    fig, ax = plt.subplots(figsize=(20, 1.3 + 0.7 * len(cov)))
    ax.axis("off")
    col_labels = ["Group", "Days", "Cycles", "Awake\n(min)", "NREM\n(min)",
                 "REM\n(min)", "Unclear\n(min)", "Label source\n(cyc count)",
                 "N2W", "MA", "W2N", "REM\n(events)", "Modalities present"]
    col_widths = [0.09, 0.045, 0.05, 0.06, 0.06, 0.06, 0.07, 0.16,
                 0.045, 0.045, 0.045, 0.06, 0.21]
    cell_text = []
    for r in cov.itertuples():
        cell_text.append([
            r.label, str(r.n_days), str(r.n_cycles),
            f"{r.min_awake:.0f}", f"{r.min_nrem:.0f}", f"{r.min_rem:.1f}", f"{r.min_unclear:.0f}",
            r.label_sources, str(r.n2w), str(r.ma), str(r.w2n), str(r.rem),
            r.modalities,
        ])
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center",
                     cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(wrap=True)
        if row == 0:
            cell.set_text_props(weight="bold", wrap=True)
            cell.set_facecolor("#E0E0E0")
        else:
            g = cov.iloc[row - 1]["group"]
            cell.set_facecolor(C.GROUP_COL.get(g, "#FFFFFF"))
            cell.set_alpha(0.12)

    totals_days = cov["n_days"].sum()
    totals_cyc = cov["n_cycles"].sum()
    ax.set_title(
        f"Reserve: dataset coverage summary — {totals_days} recording days, {totals_cyc} cycles total\n"
        f"Label source shown for 'manual+ml' mode (used by the main figures); "
        f"REM-like events are rare throughout (descriptive only).\n{C.PRELIM_STAMP}",
        fontsize=10, pad=18)
    fig.tight_layout()
    C.save_fig(fig, "figR_coverage")
    plt.close(fig)
    return cov


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    C.ensure_dirs()
    rem_summary = make_fig_rem()
    print("REM:", {k: v for k, v in rem_summary.items() if k != "table"})
    ne_summary = make_fig_ne()
    print("NE:", ne_summary)
    cov = make_fig_coverage()
    print("Coverage:\n", cov.to_string(index=False))
