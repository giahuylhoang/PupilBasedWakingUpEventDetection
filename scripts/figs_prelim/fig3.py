"""Figure 3 (preliminary) — event-triggered responses, WT only.

NREM-to-wake (N2W) and micro-arousal (MA) transitions are associated with
coordinated arteriole dilation and astrocyte Ca2+ elevation; wake-to-NREM
(W2N) is a semi-control; REM-like onsets are shown descriptively only.

Run (from repo root):
    ./venv/bin/python -B -m scripts.figs_prelim.fig3
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, ".")
from scripts.figs_prelim import common as C

GROUP = "WT"
rng = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# data assembly
# --------------------------------------------------------------------------- #
def get_events():
    keys = C.keys_for(GROUP)
    ev = C.response_events(keys)   # pupil-anchored, non-overlapping (v2)
    return keys, ev


def epochs_by_type(ev, ev_type, signal):
    sub = ev[ev["type"] == ev_type].reset_index(drop=True)
    rg, M = C.epochs_for(sub, signal)
    return sub, rg, M


def mouse_level_mean_trace(sub, rg, M):
    """Average events within mouse first, return dict mouse -> trace, and
    stacked array (n_mice, n_time) plus list of mouse ids."""
    mice = sub["mouse"].to_numpy()
    uniq = sorted(set(mice))
    out = np.full((len(uniq), M.shape[1]), np.nan)
    for i, m in enumerate(uniq):
        rows = M[mice == m]
        if rows.size:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                out[i] = np.nanmean(rows, axis=0)
    return uniq, out


def mean_sem_across_mice(mouse_mat):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(mouse_mat, axis=0)
        n = np.sum(np.isfinite(mouse_mat), axis=0)
        sd = np.nanstd(mouse_mat, axis=0, ddof=1)
    sem = sd / np.sqrt(np.maximum(n, 1))
    return mean, sem, n


def event_metrics_table(sub, signal):
    """Per-event peak/AUC/time-to-peak/responder for one signal, keyed to sub rows."""
    rg, M = C.epochs_for(sub, signal)
    peak, auc, tpk = C.response_metrics(rg, M)
    resp = C.responder(rg, M)
    out = sub[["type", "t0", "key", "mouse", "group"]].copy()
    out[f"{signal}_peak"] = peak
    out[f"{signal}_auc"] = auc
    out[f"{signal}_tpk"] = tpk
    out[f"{signal}_responder"] = resp
    return out, rg, M


# --------------------------------------------------------------------------- #
# Figure 3A — schematic
# --------------------------------------------------------------------------- #
def panel_3A(ev):
    """Pick one real N2W event with pupil + whisker + state ribbon."""
    n2w = ev[ev["type"] == "N2W"].reset_index(drop=True)
    chosen = None
    for r in n2w.itertuples():
        cyc = C.load_cycle(r.key)
        if "pupil" in cyc.sig and "whisk" in cyc.sig:
            tp, yp = cyc.sig["pupil"]
            tw, yw = cyc.sig["whisk"]
            win = (tp >= r.t0 - C.EV_PRE) & (tp <= r.t0 + C.EV_POST)
            if np.isfinite(yp[win]).mean() > 0.9:
                chosen = (r, cyc)
                break
    fig = plt.figure(figsize=(7.5, 5.0))
    if chosen is None:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No N2W event with complete pupil+whisker found",
                 ha="center", va="center")
        ax.axis("off")
        return fig, None

    r, cyc = chosen
    t0 = r.t0
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 0.6], hspace=0.08)
    ax_p = fig.add_subplot(gs[0])
    ax_w = fig.add_subplot(gs[1], sharex=ax_p)
    ax_s = fig.add_subplot(gs[2], sharex=ax_p)

    tp, yp = cyc.sig["pupil"]
    tw, yw = cyc.sig["whisk"]
    win_lo, win_hi = t0 - C.EV_PRE, t0 + C.EV_POST
    mp = (tp >= win_lo) & (tp <= win_hi)
    mw = (tw >= win_lo) & (tw <= win_hi)

    ax_p.plot(tp[mp] - t0, yp[mp], color=C.SIG_COL["pupil"], lw=1.1)
    ax_p.set_ylabel(C.SIG_LABEL["pupil"])
    ax_p.axvline(0, color="k", lw=1, ls="--")
    ax_p.axvspan(C.BASE_WIN[0], C.BASE_WIN[1], color="grey", alpha=0.3, label="baseline")
    ax_p.axvspan(C.RESP_WIN[0], C.RESP_WIN[1], color="orange", alpha=0.2, label="response")
    t0_label = getattr(r, "t0_label", None)
    if t0_label is not None and abs(t0_label - t0) > 0.05:
        ax_p.axvline(t0_label - t0, color="k", ls=":", lw=1.0,
                     label=f"label boundary ({t0_label - t0:+.1f} s)")
    ax_p.legend(loc="upper right", fontsize=7, frameon=False)
    ax_p.set_title(f"Fig. 3A — example N2W event ({r.key}, t0={t0:.0f}s)", fontsize=10)

    ax_w.plot(tw[mw] - t0, yw[mw], color=C.SIG_COL["whisk"], lw=1.0)
    ax_w.set_ylabel(C.SIG_LABEL["whisk"])
    ax_w.axvline(0, color="k", lw=1, ls="--")
    ax_w.axvspan(C.BASE_WIN[0], C.BASE_WIN[1], color="grey", alpha=0.3)
    ax_w.axvspan(C.RESP_WIN[0], C.RESP_WIN[1], color="orange", alpha=0.2)

    # state ribbon from 5-s bin labels, within window
    bin0 = int(np.floor(win_lo / C.BIN))
    bin1 = int(np.ceil(win_hi / C.BIN))
    for b in range(max(bin0, 0), min(bin1, len(cyc.labels))):
        lab = cyc.labels[b]
        col = C.STATE_COL.get(lab, "white")
        ax_s.axvspan(b * C.BIN - t0, (b + 1) * C.BIN - t0, color=col, lw=0)
    ax_s.axvline(0, color="k", lw=1, ls="--")
    ax_s.set_yticks([])
    ax_s.set_xlabel("time relative to event onset (s)")
    ax_s.set_ylabel("state", fontsize=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=C.STATE_COL[s]) for s in ("NREM", "Awake")]
    ax_s.legend(handles, ["NREM", "Awake"], loc="upper right", fontsize=6,
                frameon=False, ncol=2, bbox_to_anchor=(1.0, 1.6))

    for ax in (ax_p, ax_w):
        ax.tick_params(labelbottom=False)
    ax_p.set_xlim(win_lo - t0, win_hi - t0)
    fig.text(0.01, 0.01, C.PRELIM_STAMP, fontsize=6, color="grey")
    return fig, r


# --------------------------------------------------------------------------- #
# Figure 3B — N2W event-triggered averages
# --------------------------------------------------------------------------- #
SIGNALS_ETA = ["pupil", "whisk", "art", "ca"]


def panel_eta(ev, ev_type, title_prefix, n_target=None):
    sub = ev[ev["type"] == ev_type].reset_index(drop=True)
    n_days = sub["mouse"].nunique()
    n_ev = len(sub)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharex=True)
    for ax, sig in zip(axes, SIGNALS_ETA):
        rg, M = C.epochs_for(sub, sig)
        mice, mouse_mat = mouse_level_mean_trace(sub, rg, M)
        mean, sem, n_mice_t = mean_sem_across_mice(mouse_mat)
        # faint per-mouse means behind
        for row in mouse_mat:
            if np.isfinite(row).any():
                ax.plot(rg, row, color=C.SIG_COL[sig], alpha=0.15, lw=0.8)
        ax.plot(rg, mean, color=C.SIG_COL[sig], lw=2.0)
        ax.fill_between(rg, mean - sem, mean + sem, color=C.SIG_COL[sig], alpha=0.3)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.axvspan(C.BASE_WIN[0], C.BASE_WIN[1], color="grey", alpha=0.15)
        ax.axvspan(C.RESP_WIN[0], C.RESP_WIN[1], color="orange", alpha=0.1)
        ax.set_title(C.SIG_LABEL[sig], fontsize=10)
        ax.set_xlabel("time (s)")
        ax.set_xlim(-30, 60)
    axes[0].set_ylabel("mean ± SEM across mice")
    n_str = f"{n_days} days, {n_ev} events"
    if n_target is not None and n_ev != n_target:
        n_str += f" (expected ~{n_target})"
    fig.suptitle(f"{title_prefix} ({n_str})", fontsize=12)
    fig.text(0.005, 0.01, C.PRELIM_STAMP, fontsize=6, color="grey")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig, sub


# --------------------------------------------------------------------------- #
# Figure 3C/3E — heatmaps + per-mouse quantification dots
# --------------------------------------------------------------------------- #
def panel_heat_and_quant(ev, ev_type, title_prefix):
    sub = ev[ev["type"] == ev_type].reset_index(drop=True)
    n_days = sub["mouse"].nunique()
    n_ev = len(sub)

    # sort rows by peak pupil response (independent of outcome)
    pup_metrics, rg_pup, M_pup = event_metrics_table(sub, "pupil")
    order = np.argsort(-np.nan_to_num(pup_metrics["pupil_peak"].to_numpy(), nan=-np.inf))

    art_metrics, rg_art, M_art = event_metrics_table(sub, "art")
    ca_metrics, rg_ca, M_ca = event_metrics_table(sub, "ca")

    fig = plt.figure(figsize=(15, 7.5))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1.3, 1.3], hspace=0.45, wspace=0.5)

    # heatmaps (art, ca), rows sorted by pupil peak
    for col, (name, rg, M) in enumerate([("art", rg_art, M_art), ("ca", rg_ca, M_ca)]):
        ax = fig.add_subplot(gs[:, col])
        Ms = M[order]
        vmax = np.nanpercentile(np.abs(Ms), 95) if np.isfinite(Ms).any() else 1.0
        vmax = vmax if vmax > 0 else 1.0
        im = ax.imshow(Ms, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        extent=[rg[0], rg[-1], Ms.shape[0], 0])
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlim(-30, 60)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("events (sorted by peak pupil resp.)")
        ax.set_title(f"{C.SIG_LABEL[name]}\n(n={Ms.shape[0]} events, rows: pupil-sorted)",
                     fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)

    # per-mouse quantification dots
    merged = art_metrics.merge(
        ca_metrics[["key", "t0", "mouse", "ca_peak", "ca_auc", "ca_tpk", "ca_responder"]],
        on=["key", "t0", "mouse"], how="outer")
    merged["lat_diff"] = merged["ca_tpk"] - merged["art_tpk"]

    quant_specs = [
        ("art_peak", "peak Δart"),
        ("ca_peak", "peak ΔCa"),
        ("art_auc", "AUC art"),
        ("art_responder", "resp. prob (art)"),
        ("art_tpk", "t-to-peak art (s)"),
        ("ca_tpk", "t-to-peak Ca (s)"),
        ("lat_diff", "Ca − art latency (s)"),
    ]
    mouse_summ = merged.groupby("mouse", as_index=False)[
        ["art_peak", "ca_peak", "art_auc", "ca_auc", "art_responder", "ca_responder",
         "art_tpk", "ca_tpk", "lat_diff"]].mean()

    ax_q = fig.add_subplot(gs[0, 2:])
    ax_q.axis("off")
    labels = [lab for _, lab in quant_specs]
    xpos = np.arange(len(labels))
    for i, (col, lab) in enumerate(quant_specs):
        vals = mouse_summ[col].dropna().to_numpy()
        if vals.size == 0:
            continue
        jitter = (rng.random(vals.size) - 0.5) * 0.15
        ax_q.scatter(np.full(vals.size, i) + jitter, vals, color="#424242", s=22,
                     alpha=0.8, zorder=3)
        ax_q.scatter([i], [np.nanmean(vals)], color="#D32F2F", s=70, marker="D", zorder=4)
    ax_q.set_axis_on()
    ax_q.set_xticks(xpos)
    ax_q.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax_q.set_ylabel("mouse-level mean", fontsize=8)
    ax_q.set_title(f"per-mouse quantification (n={n_days} days, diamond = group mean)",
                   fontsize=9)
    ax_q.axhline(0, color="grey", lw=0.6, ls=":")

    ax_tab = fig.add_subplot(gs[1, 2:])
    ax_tab.axis("off")
    def _fmt(v):
        return "n/a" if not np.isfinite(v) else f"{v:.3g}"
    rows_txt = []
    for col, lab in quant_specs:
        v = mouse_summ[col].to_numpy()
        v = v[np.isfinite(v)]
        if v.size:
            rows_txt.append([lab, f"{v.mean():.3g}", f"{v.std(ddof=1):.3g}" if v.size > 1 else "n/a",
                              str(v.size)])
        else:
            rows_txt.append([lab, "n/a", "n/a", "0"])
    tbl = ax_tab.table(cellText=rows_txt, colLabels=["metric", "mean", "sd", "n mice"],
                        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.3)
    ax_tab.set_title("mouse-level summary", fontsize=9)

    fig.suptitle(f"{title_prefix} (n={n_days} days, {n_ev} events)", fontsize=12)
    fig.text(0.005, 0.01, C.PRELIM_STAMP, fontsize=6, color="grey")
    return fig, merged, mouse_summ


# --------------------------------------------------------------------------- #
# Figure 3F — W2N ETA + cross-type comparison with LMM
# --------------------------------------------------------------------------- #
def panel_3F(ev):
    w2n = ev[ev["type"] == "W2N"].reset_index(drop=True)
    n_days_w2n = w2n["mouse"].nunique()

    fig = plt.figure(figsize=(15, 6.5))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], hspace=0.5, wspace=0.5)

    for i, sig in enumerate(SIGNALS_ETA):
        ax = fig.add_subplot(gs[0, i])
        rg, M = C.epochs_for(w2n, sig)
        mice, mouse_mat = mouse_level_mean_trace(w2n, rg, M)
        mean, sem, _ = mean_sem_across_mice(mouse_mat)
        for row in mouse_mat:
            if np.isfinite(row).any():
                ax.plot(rg, row, color=C.SIG_COL[sig], alpha=0.15, lw=0.8)
        ax.plot(rg, mean, color=C.SIG_COL[sig], lw=2.0)
        ax.fill_between(rg, mean - sem, mean + sem, color=C.SIG_COL[sig], alpha=0.3)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.axvspan(C.BASE_WIN[0], C.BASE_WIN[1], color="grey", alpha=0.15)
        ax.axvspan(C.RESP_WIN[0], C.RESP_WIN[1], color="orange", alpha=0.1)
        ax.set_title(C.SIG_LABEL[sig], fontsize=9)
        ax.set_xlim(-30, 60)
        ax.set_xlabel("time (s)")
    fig.text(0.01, 0.92, f"W2N semi-control ETA (n={n_days_w2n} days, {len(w2n)} events)",
              fontsize=10, weight="bold")

    # cross-type comparison, mouse-level, art and ca peak
    all_peaks = []
    for ev_type in ("N2W", "MA", "W2N"):
        sub = ev[ev["type"] == ev_type].reset_index(drop=True)
        art_m, _, _ = event_metrics_table(sub, "art")
        ca_m, _, _ = event_metrics_table(sub, "ca")
        m = art_m.merge(ca_m[["key", "t0", "mouse", "ca_peak"]], on=["key", "t0", "mouse"])
        m["type"] = ev_type
        all_peaks.append(m[["type", "mouse", "art_peak", "ca_peak"]])
    peaks_df = pd.concat(all_peaks, ignore_index=True)

    ax_art = fig.add_subplot(gs[1, 0:2])
    ax_ca = fig.add_subplot(gs[1, 2:4])
    type_order = ["N2W", "MA", "W2N"]
    for ax, col, lab in [(ax_art, "art_peak", "peak Δart"), (ax_ca, "ca_peak", "peak ΔCa")]:
        mouse_level = peaks_df.groupby(["type", "mouse"], as_index=False)[col].mean()
        for i, t in enumerate(type_order):
            ev_vals = peaks_df.loc[peaks_df["type"] == t, col].dropna().to_numpy()
            jitter = (rng.random(ev_vals.size) - 0.5) * 0.2
            ax.scatter(np.full(ev_vals.size, i) + jitter, ev_vals, color="grey",
                       alpha=0.15, s=10, zorder=2)
            mv = mouse_level.loc[mouse_level["type"] == t, col].dropna().to_numpy()
            jitter_m = (rng.random(mv.size) - 0.5) * 0.1
            ax.scatter(np.full(mv.size, i) + jitter_m, mv, color="#424242", s=45,
                       alpha=0.9, zorder=3, edgecolors="white", linewidths=0.5)
            if mv.size:
                ax.scatter([i], [mv.mean()], color="#D32F2F", s=90, marker="D", zorder=4)
        ax.set_xticks(range(len(type_order)))
        ax.set_xticklabels(type_order)
        ax.set_ylabel(lab)
        ax.axhline(0, color="grey", lw=0.6, ls=":")
        ax.set_title(f"{lab} by event type (grey=event, dark=mouse mean, diamond=group mean)",
                     fontsize=8.5)

    fig.text(0.005, 0.01, C.PRELIM_STAMP, fontsize=6, color="grey")

    # LMM: event-level peak ~ C(type), random intercept mouse
    lmm_tables = {}
    for col in ("art_peak", "ca_peak"):
        d = peaks_df[["type", "mouse", col]].dropna().rename(columns={col: "value"})
        d["type"] = pd.Categorical(d["type"], categories=type_order)
        t = C.lmm(d, "value", "C(type)")
        lmm_tables[col] = t

    return fig, peaks_df, lmm_tables


# --------------------------------------------------------------------------- #
# Figure 3G — arousal strength vs response
# --------------------------------------------------------------------------- #
def panel_3G(ev):
    sub = ev[ev["type"].isin(["N2W", "MA"])].reset_index(drop=True)
    pup_m, _, _ = event_metrics_table(sub, "pupil")
    art_m, _, _ = event_metrics_table(sub, "art")
    ca_m, _, _ = event_metrics_table(sub, "ca")
    merged = pup_m[["key", "t0", "mouse", "type", "pupil_peak"]].merge(
        art_m[["key", "t0", "mouse", "art_peak"]], on=["key", "t0", "mouse"]).merge(
        ca_m[["key", "t0", "mouse", "ca_peak"]], on=["key", "t0", "mouse"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    nbins = 5
    for ax, ycol, ylab, col in [(axes[0], "art_peak", "peak Δart", C.SIG_COL["art"]),
                                 (axes[1], "ca_peak", "peak ΔCa", C.SIG_COL["ca"])]:
        d = merged[["mouse", "pupil_peak", ycol]].dropna()
        ax.scatter(d["pupil_peak"], d[ycol], color=col, alpha=0.12, s=12, zorder=2,
                   label="events")
        # mouse-level binned means on pupil_peak quantiles
        try:
            d = d.assign(bin=pd.qcut(d["pupil_peak"], nbins, duplicates="drop"))
            mouse_bin = d.groupby(["mouse", "bin"], observed=True).agg(
                x=("pupil_peak", "mean"), y=(ycol, "mean")).reset_index()
            bin_summ = mouse_bin.groupby("bin", observed=True).agg(
                x=("x", "mean"), y=("y", "mean"),
                y_sem=("y", lambda s: s.std(ddof=1) / np.sqrt(max(len(s), 1)))
            ).reset_index()
            ax.errorbar(bin_summ["x"], bin_summ["y"], yerr=bin_summ["y_sem"],
                        fmt="o-", color="#B71C1C", lw=1.8, ms=7, zorder=4,
                        label="mouse-level binned mean")
        except (ValueError, IndexError):
            pass
        ax.set_xlabel("event peak pupil (z, response window)")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(f"{ylab} vs arousal strength (N2W+MA events)", fontsize=9)

    # LMM: peak_art ~ pupil_peak, random intercept mouse
    d_art = merged[["mouse", "pupil_peak", "art_peak"]].dropna().rename(
        columns={"art_peak": "value"})
    lmm_art = C.lmm(d_art, "value", "pupil_peak")
    d_ca = merged[["mouse", "pupil_peak", "ca_peak"]].dropna().rename(
        columns={"ca_peak": "value"})
    lmm_ca = C.lmm(d_ca, "value", "pupil_peak")

    fig.text(0.005, 0.01, C.PRELIM_STAMP, fontsize=6, color="grey")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig, merged, {"art_peak": lmm_art, "ca_peak": lmm_ca}


# --------------------------------------------------------------------------- #
# REM-like descriptive panel
# --------------------------------------------------------------------------- #
def panel_REM(ev):
    rem = ev[ev["type"] == "REM"].reset_index(drop=True)
    if rem.empty:
        return None, rem
    fig, axes = plt.subplots(len(rem), 3, figsize=(10, 2.6 * len(rem)), squeeze=False)
    for i, r in enumerate(rem.itertuples()):
        cyc = C.load_cycle(r.key)
        for j, sig in enumerate(["pupil", "art", "ca"]):
            ax = axes[i][j]
            if sig in cyc.sig:
                t, y = cyc.sig[sig]
                m = (t >= r.t0 - C.EV_PRE) & (t <= r.t0 + C.EV_POST)
                ax.plot(t[m] - r.t0, y[m], color=C.SIG_COL[sig], lw=1.0)
                ax.axvline(0, color="k", lw=0.8, ls="--")
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            if i == 0:
                ax.set_title(C.SIG_LABEL[sig], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{r.key.split('_')[-3]}\nt0={r.t0:.0f}s", fontsize=7)
            if i == len(rem) - 1:
                ax.set_xlabel("time (s)")
    fig.suptitle(f"REM-like onset events, descriptive only (n={len(rem)} events, "
                 f"{rem['mouse'].nunique()} days) — too few for formal stats", fontsize=11)
    fig.text(0.005, 0.01, C.PRELIM_STAMP, fontsize=6, color="grey")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return fig, rem


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    C.ensure_dirs()
    keys, ev = get_events()
    headline = {}

    # --- 3A/3B/3C combined figure: fig3_A-C_N2W ---
    fig_A, ex_event = panel_3A(ev)
    fig_B, n2w_sub = panel_eta(ev, "N2W", "Fig. 3B — NREM-to-wake event-triggered averages",
                                n_target=130)
    fig_C, n2w_merged, n2w_mouse_summ = panel_heat_and_quant(
        ev, "N2W", "Fig. 3C — NREM-to-wake heat maps and quantification")

    C.save_fig(fig_A, "fig3_A_schematic")
    C.save_fig(fig_B, "fig3_B_N2W_eta")
    C.save_fig(fig_C, "fig3_C_N2W_heatmap_quant")
    C.save_table(n2w_merged, "fig3_C_N2W_event_metrics")
    C.save_table(n2w_mouse_summ, "fig3_C_N2W_mouse_summary")
    plt.close(fig_A); plt.close(fig_B); plt.close(fig_C)

    headline["N2W_art_peak_mean"] = n2w_mouse_summ["art_peak"].mean()
    headline["N2W_art_peak_n"] = n2w_mouse_summ["art_peak"].notna().sum()
    headline["N2W_ca_peak_mean"] = n2w_mouse_summ["ca_peak"].mean()
    headline["N2W_ca_peak_n"] = n2w_mouse_summ["ca_peak"].notna().sum()
    headline["N2W_n_events"] = len(ev[ev["type"] == "N2W"])
    headline["N2W_n_days"] = ev.loc[ev["type"] == "N2W", "mouse"].nunique()

    # --- 3D/3E: fig3_D-E_MA ---
    fig_D, ma_sub = panel_eta(ev, "MA", "Fig. 3D — Micro-arousal event-triggered averages",
                               n_target=30)
    fig_E, ma_merged, ma_mouse_summ = panel_heat_and_quant(
        ev, "MA", "Fig. 3E — Micro-arousal heat maps and quantification")
    C.save_fig(fig_D, "fig3_D_MA_eta")
    C.save_fig(fig_E, "fig3_E_MA_heatmap_quant")
    C.save_table(ma_merged, "fig3_E_MA_event_metrics")
    C.save_table(ma_mouse_summ, "fig3_E_MA_mouse_summary")
    plt.close(fig_D); plt.close(fig_E)

    headline["MA_art_peak_mean"] = ma_mouse_summ["art_peak"].mean()
    headline["MA_art_peak_n"] = ma_mouse_summ["art_peak"].notna().sum()
    headline["MA_ca_peak_mean"] = ma_mouse_summ["ca_peak"].mean()
    headline["MA_ca_peak_n"] = ma_mouse_summ["ca_peak"].notna().sum()
    headline["MA_n_events"] = len(ev[ev["type"] == "MA"])
    headline["MA_n_days"] = ev.loc[ev["type"] == "MA", "mouse"].nunique()

    # --- 3F/3G: fig3_F-G_control_and_strength ---
    fig_F, peaks_df, lmm_F = panel_3F(ev)
    fig_G, strength_df, lmm_G = panel_3G(ev)

    fig_combo = plt.figure(figsize=(15, 13))
    fig_combo.suptitle("Fig. 3F-G — W2N semi-control, cross-type comparison, "
                        "and arousal-strength relationship", fontsize=13, y=0.995)
    C.save_fig(fig_F, "fig3_F_W2N_control_and_comparison")
    C.save_fig(fig_G, "fig3_G_arousal_strength")
    plt.close(fig_combo)
    plt.close(fig_F); plt.close(fig_G)

    C.save_table(peaks_df, "fig3_F_type_comparison_peaks")
    C.save_table(strength_df, "fig3_G_arousal_strength_data")

    for name, tbl in lmm_F.items():
        if tbl is not None:
            C.save_table(tbl, f"fig3_F_lmm_{name}_by_type")
    for name, tbl in lmm_G.items():
        if tbl is not None:
            C.save_table(tbl, f"fig3_G_lmm_{name}_vs_pupil")

    headline["W2N_n_events"] = len(ev[ev["type"] == "W2N"])
    headline["W2N_n_days"] = ev.loc[ev["type"] == "W2N", "mouse"].nunique()

    # --- REM descriptive ---
    fig_REM, rem_sub = panel_REM(ev)
    if fig_REM is not None:
        C.save_fig(fig_REM, "fig3_REM_descriptive")
        plt.close(fig_REM)
    headline["REM_n_events"] = len(rem_sub)
    headline["REM_n_days"] = rem_sub["mouse"].nunique() if len(rem_sub) else 0

    write_notes(headline, lmm_F, lmm_G, n2w_mouse_summ, ma_mouse_summ)
    print_summary(headline, lmm_F)


def write_notes(h, lmm_F, lmm_G, n2w_mouse_summ, ma_mouse_summ):
    lines = []
    lines.append("# Figure 3 — event-triggered responses (WT only)\n")
    lines.append(f"\n_{C.PRELIM_STAMP}_\n")
    lines.append("\n## Panels\n")
    lines.append(
        f"- **3A** (`fig3_A_schematic`): one real N2W event — pupil, whisker, state "
        f"ribbon, t0, baseline window {C.BASE_WIN} s and response window {C.RESP_WIN} s "
        f"shaded.\n")
    lines.append(
        f"- **3B** (`fig3_B_N2W_eta`): N2W event-triggered averages (mean ± SEM across "
        f"mice, events averaged within mouse first) for pupil, whisker, arteriole, Ca. "
        f"n = {h['N2W_n_days']} days, {h['N2W_n_events']} events.\n")
    lines.append(
        f"- **3C** (`fig3_C_N2W_heatmap_quant`): arteriole and Ca heat maps for N2W, rows "
        f"sorted by peak pupil response; per-mouse dots for peak art ({h['N2W_art_peak_mean']:.3g} "
        f"mean ΔD/D0, n={h['N2W_art_peak_n']} mice), peak Ca ({h['N2W_ca_peak_mean']:.3g} mean "
        f"ΔF/F0, n={h['N2W_ca_peak_n']} mice), AUC, response probability, time-to-peak, and "
        f"Ca-minus-art latency difference.\n")
    lines.append(
        f"- **3D** (`fig3_D_MA_eta`) / **3E** (`fig3_E_MA_heatmap_quant`): same analysis for "
        f"micro-arousals. n = {h['MA_n_days']} days, {h['MA_n_events']} events (honestly "
        f"small — the brief's expected count of 30 across WT days). Peak art mean "
        f"{h['MA_art_peak_mean']:.3g} (n={h['MA_art_peak_n']} mice), peak Ca mean "
        f"{h['MA_ca_peak_mean']:.3g} (n={h['MA_ca_peak_n']} mice).\n")
    lines.append(
        f"- **3F** (`fig3_F_W2N_control_and_comparison`): W2N semi-control ETA "
        f"(n = {h['W2N_n_days']} days, {h['W2N_n_events']} events) plus mouse-/event-level "
        f"comparison of peak art and peak Ca across N2W vs MA vs W2N, with an event-level "
        f"LMM `peak ~ C(type)` (random intercept = mouse).\n")
    lines.append(
        f"- **3G** (`fig3_G_arousal_strength`): event-level peak pupil (x) vs peak art / "
        f"peak Ca (y), faint events + mouse-level binned means, with LMM "
        f"`peak_art ~ pupil_peak` and `peak_ca ~ pupil_peak` (random intercept = mouse), "
        f"pooling N2W + MA events.\n")
    if h["REM_n_events"] > 0:
        lines.append(
            f"- **REM** (`fig3_REM_descriptive`): all {h['REM_n_events']} REM-like onset "
            f"events shown individually (art, Ca, pupil) from {h['REM_n_days']} day(s). "
            f"Descriptive only — not used for formal comparison.\n")
    else:
        lines.append(
            "- **REM**: no REM-like events were found in the WT N2W/MA/W2N table build; "
            "panel not produced.\n")

    lines.append("\n## LMM results (Fig. 3F, peak ~ C(type), random intercept mouse)\n")
    for name, tbl in lmm_F.items():
        if tbl is None:
            lines.append(f"- `{name}`: could not be fitted (too few mice/observations).\n")
        elif "FIT_FAILED" in str(tbl.get("term", pd.Series()).tolist()):
            lines.append(f"- `{name}`: fit failed — {tbl.iloc[0].get('note', '')}\n")
        else:
            lines.append(f"- `{name}`: fit converged, n_obs={tbl['n_obs'].iloc[0]}, "
                          f"n_mice={tbl['n_mice'].iloc[0]} (full table in "
                          f"`fig3_F_lmm_{name}_by_type.csv`).\n")

    lines.append("\n## LMM results (Fig. 3G, peak ~ pupil_peak, random intercept mouse)\n")
    for name, tbl in lmm_G.items():
        if tbl is None:
            lines.append(f"- `{name}`: could not be fitted (too few mice/observations).\n")
        elif "FIT_FAILED" in str(tbl.get("term", pd.Series()).tolist()):
            lines.append(f"- `{name}`: fit failed — {tbl.iloc[0].get('note', '')}\n")
        else:
            lines.append(f"- `{name}`: fit converged, n_obs={tbl['n_obs'].iloc[0]}, "
                          f"n_mice={tbl['n_mice'].iloc[0]} (full table in "
                          f"`fig3_G_lmm_{name}_vs_pupil.csv`).\n")

    lines.append("\n## Caveats\n")
    lines.append(
        "- Events are pupil-anchored (v2): t0 is pupil dilation onset found within +/-10 s "
        "of the 5-s label boundary (C.response_events), and events with a neighbour within "
        "-15..+20 s are dropped. Events without a clear pupil rise keep the label t0, so "
        "latencies are still only reliable to a few seconds.\n")
    lines.append(
        "- MA n is small (WT: "
        f"{h['MA_n_events']} events / {h['MA_n_days']} days) — statistics reported for "
        "transparency, not as strong evidence; a Wilcoxon at this n cannot reach small p "
        "(see `common.paired_test` / `wilcoxon_min_p`).\n")
    lines.append(
        "- Unit of statistical inference is the recording day (mouse); event- and "
        "cycle-level points are shown faintly for context only, never treated as "
        "independent n.\n")
    lines.append(
        "- Pupil is a within-cycle robust z-score, not comparable in absolute units "
        "across sessions; only within-cycle comparisons (e.g. event vs event within the "
        "same cycle) are meaningful.\n")
    lines.append(
        f"- REM-like events (n={h['REM_n_events']}) are too few for any formal "
        "comparison and are shown descriptively only, per the brief.\n")
    lines.append(
        "- Heat map rows are sorted by peak pupil response in the response window, "
        "independent of the arteriole/Ca outcome, so clustering of color in the "
        "heat maps reflects a real pupil-ordered structure, not post-hoc sorting on the "
        "measured outcome.\n")

    lines.append("\n## Reproduce\n")
    lines.append("```\n./venv/bin/python -B -m scripts.figs_prelim.fig3\n```\n")

    C.ensure_dirs()
    notes_dir = C.OUT / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    (notes_dir / "fig3.md").write_text("".join(lines))


def print_summary(h, lmm_F):
    print("\n=== Figure 3 summary ===")
    print(f"N2W: {h['N2W_n_events']} events / {h['N2W_n_days']} days — "
          f"peak art mean {h['N2W_art_peak_mean']:.3g} (n={h['N2W_art_peak_n']} mice), "
          f"peak Ca mean {h['N2W_ca_peak_mean']:.3g} (n={h['N2W_ca_peak_n']} mice)")
    print(f"MA: {h['MA_n_events']} events / {h['MA_n_days']} days — "
          f"peak art mean {h['MA_art_peak_mean']:.3g} (n={h['MA_art_peak_n']} mice), "
          f"peak Ca mean {h['MA_ca_peak_mean']:.3g} (n={h['MA_ca_peak_n']} mice)")
    print(f"W2N: {h['W2N_n_events']} events / {h['W2N_n_days']} days (semi-control)")
    print(f"REM: {h['REM_n_events']} events / {h['REM_n_days']} days (descriptive only)")


if __name__ == "__main__":
    main()
