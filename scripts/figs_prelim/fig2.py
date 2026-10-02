"""Figure 2 — state-dependent astrocyte Ca2+ and arteriole diameter in RSC (WT only).

Run (from repo root):
    ./venv/bin/python -B scripts/figs_prelim/fig2.py

Outputs -> data/RESULTS_figures_prelim_2026-10-01/figures/fig2_*.png(.pdf)
Tables  -> data/RESULTS_figures_prelim_2026-10-01/tables/fig2_*.csv
Notes   -> data/RESULTS_figures_prelim_2026-10-01/notes/fig2.md

See scripts/figs_prelim/AGENT_BRIEF.md for the rules this script must follow and
scripts/figs_prelim/common.py for the shared definitions (states, events,
normalisation, spectra, stats) used here.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sstats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figs_prelim import common as C  # noqa: E402

GROUP = "WT"
NOTES_PATH = C.OUT / "notes" / "fig2.md"
RNG = np.random.default_rng(2)

# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
KEYS = C.keys_for(GROUP)
CYCLES = {k: C.load_cycle(k) for k in KEYS}
MICE = sorted({c.info.mouse for c in CYCLES.values()})
EVENTS = C.event_table(KEYS)

notes_lines = []


def note(line=""):
    notes_lines.append(line)


# --------------------------------------------------------------------------- #
# panel 2A: representative recording + insets
# --------------------------------------------------------------------------- #
def pick_representative_cycle():
    """Score each cycle by number of clean state transitions, require art+ca."""
    rows = []
    for k, cyc in CYCLES.items():
        if "art" not in cyc.sig or "ca" not in cyc.sig:
            continue
        b = C.bouts(cyc.labels)
        b = b[b["state"].isin(C.STATES)]
        n_trans = max(0, len(b) - 1)
        # fraction of cycle duration covered by finite art & ca samples
        t_art, y_art = cyc.sig["art"]
        t_ca, y_ca = cyc.sig["ca"]
        frac_art = np.isfinite(y_art).mean()
        frac_ca = np.isfinite(y_ca).mean()
        ev_k = EVENTS[EVENTS["key"] == k]
        n_n2w = (ev_k["type"] == "N2W").sum()
        n_ma = (ev_k["type"] == "MA").sum()
        rows.append(dict(key=k, n_trans=n_trans, frac_art=frac_art, frac_ca=frac_ca,
                         n_n2w=n_n2w, n_ma=n_ma))
    df = pd.DataFrame(rows)
    df = df[(df["frac_art"] >= 0.90) & (df["frac_ca"] >= 0.90)]
    df["score"] = df["n_trans"] + df["n_n2w"] + df["n_ma"]
    df = df.sort_values("score", ascending=False)
    return df


def pick_zoom_n2w(cyc, df_cycle_key):
    ev = EVENTS[(EVENTS["key"] == df_cycle_key) & (EVENTS["type"] == "N2W")]
    if ev.empty:
        return None
    # prefer N2W with the longest preceding NREM bout and longest following wake
    ev = ev.assign(score=ev["pre_dur_s"] + ev["post_dur_s"])
    return ev.sort_values("score", ascending=False).iloc[0]


def pick_zoom_nrem(cyc):
    """Longest NREM bout >=60 s in this cycle; falls back to the best bout in
    any art+ca-covered WT cycle if this one has none."""
    b = C.bouts(cyc.labels)
    b = b[(b["state"] == "NREM") & (b["dur_s"] >= 60.0)]
    if not b.empty:
        return cyc, b.sort_values("dur_s", ascending=False).iloc[0]
    best = None
    for k, c2 in CYCLES.items():
        if "art" not in c2.sig or "ca" not in c2.sig:
            continue
        b2 = C.bouts(c2.labels)
        b2 = b2[(b2["state"] == "NREM") & (b2["dur_s"] >= 60.0)]
        if b2.empty:
            continue
        row = b2.sort_values("dur_s", ascending=False).iloc[0]
        if best is None or row["dur_s"] > best[1]["dur_s"]:
            best = (c2, row)
    return best if best is not None else (None, None)


def plot_track(ax, cyc, name, t0, t1, color=None, label=None):
    if name not in cyc.sig:
        ax.set_visible(False)
        return
    t, y = cyc.sig[name]
    m = (t >= t0) & (t <= t1)
    ax.plot(t[m], y[m], lw=0.8, color=color or C.SIG_COL[name])
    ax.set_ylabel(label or C.SIG_LABEL[name], fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xlim(t0, t1)


def mark_events(ax, cyc, t0, t1, t_offset=0.0, label=False):
    ev = EVENTS[EVENTS["key"] == cyc.key]
    ev = ev[(ev["t0"] >= t0) & (ev["t0"] <= t1)]
    for r in ev.itertuples():
        x = r.t0 - t_offset
        if r.type == "N2W":
            ax.axvline(x, color="black", lw=1.0, ls="-", alpha=0.8)
            if label:
                ax.text(x, 1.03, "N2W", transform=ax.get_xaxis_transform(),
                        fontsize=5.5, ha="center", va="bottom", rotation=90)
        elif r.type == "MA":
            ax.axvline(x, color="#6A1B9A", lw=0.9, ls="--", alpha=0.8)
            if label:
                ax.text(x, 1.03, "MA", transform=ax.get_xaxis_transform(),
                        fontsize=5.5, ha="center", va="bottom", rotation=90,
                        color="#6A1B9A")


def make_fig2a():
    cand = pick_representative_cycle()
    if cand.empty:
        note("**2A**: could NOT build a representative-recording panel — no WT "
             "cycle has >=90% finite arteriole AND calcium coverage.")
        return None, None, None
    rep_key = cand.iloc[0]["key"]
    cyc = CYCLES[rep_key]

    sig_order = [s for s in ("pupil", "whisk", "art", "ca") if s in cyc.sig]
    n_tracks = len(sig_order)

    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(n_tracks + 1, 2, height_ratios=[1] * n_tracks + [1.3],
                          width_ratios=[1, 1], hspace=0.6, wspace=0.3,
                          top=0.90, bottom=0.06)

    t_end = cyc.duration
    top_axes = []
    for i, name in enumerate(sig_order):
        ax = fig.add_subplot(gs[i, :])
        plot_track(ax, cyc, name, 0, t_end)
        C.shade_states(ax, cyc.labels, alpha=0.15)
        mark_events(ax, cyc, 0, t_end, label=(i == 0))
        if i < n_tracks - 1:
            ax.set_xticklabels([])
        top_axes.append(ax)
    top_axes[-1].set_xlabel("time (s)", fontsize=8)
    top_axes[0].set_title(f"Fig. 2A — representative WT recording: {rep_key}", fontsize=9, pad=22)
    fig.text(0.5, 0.945, "black = N2W onset  ·  purple dashed = MA onset",
             ha="center", va="bottom", fontsize=7, color="0.3")

    # insets
    n2w_row = pick_zoom_n2w(cyc, rep_key)
    nrem_cyc, nrem_row = pick_zoom_nrem(cyc)

    ax_n2w = fig.add_subplot(gs[n_tracks, 0])
    ax_nrem = fig.add_subplot(gs[n_tracks, 1])

    def draw_inset(ax, src_cyc, t0, t1, title):
        ax.set_title(title, fontsize=8)
        offs = []
        y0 = 1.0
        step = 1.0
        for name in sig_order:
            if name not in src_cyc.sig:
                continue
            t, y = src_cyc.sig[name]
            m = (t >= t0) & (t <= t1)
            yy = y[m]
            if np.isfinite(yy).sum() < 2:
                continue
            yy_n = (yy - np.nanmean(yy))
            scale = np.nanstd(yy_n)
            scale = scale if (np.isfinite(scale) and scale > 1e-9) else 1.0
            yy_n = yy_n / scale
            ax.plot(t[m] - t0, yy_n + y0, lw=0.9, color=C.SIG_COL[name], label=name)
            offs.append((y0, name))
            y0 -= step
        C.shade_states(ax, src_cyc.labels, t_offset=t0, alpha=0.15)
        mark_events(ax, src_cyc, t0, t1, t_offset=t0)
        ax.set_yticks([o for o, _ in offs])
        ax.set_yticklabels([C.SIG_LABEL.get(n, n) for _, n in offs], fontsize=6)
        ax.set_xlabel("time rel. to window start (s)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, t1 - t0)

    if n2w_row is not None:
        t0 = max(0.0, n2w_row.t0 - 30.0)
        t1 = min(t_end, n2w_row.t0 + 60.0)
        draw_inset(ax_n2w, cyc, t0, t1, "Inset i: N2W transition (per-track z-scored)")
        note(f"**2A inset i (N2W)**: window {t0:.0f}-{t1:.0f} s of {rep_key}, "
             f"transition at t0={n2w_row.t0:.0f} s "
             f"(preceding NREM {n2w_row.pre_dur_s:.0f} s, following wake {n2w_row.post_dur_s:.0f} s).")
    else:
        ax_n2w.text(0.5, 0.5, "no qualifying N2W event in this cycle", ha="center",
                   va="center", fontsize=8, transform=ax_n2w.transAxes)
        note("**2A inset i (N2W)**: could NOT find a qualifying N2W transition in "
             "the chosen representative cycle.")

    if nrem_row is not None:
        t0, t1 = nrem_row.t_start, nrem_row.t_start + nrem_row.dur_s
        same = nrem_cyc.key == rep_key
        src_note = "" if same else f" (from {nrem_cyc.key}, no >=60 s NREM bout in the representative cycle itself)"
        draw_inset(ax_nrem, nrem_cyc, t0, t1,
                  f"Inset ii: stable NREM ({t1 - t0:.0f} s, per-track z-scored)"
                  + ("" if same else " *different cycle*"))
        note(f"**2A inset ii (stable NREM)**: window {t0:.0f}-{t1:.0f} s"
             f"{src_note} ({t1 - t0:.0f} s NREM bout).")
    else:
        ax_nrem.text(0.5, 0.5, "no WT cycle has a NREM bout >=60 s\nwith art+ca coverage",
                    ha="center", va="center", fontsize=8, transform=ax_nrem.transAxes)
        note("**2A inset ii (stable NREM)**: could NOT find a NREM bout >=60 s in "
             "ANY WT cycle with art+ca coverage.")

    fig.text(0.5, 1.0, C.PRELIM_STAMP, ha="center", va="top", fontsize=6, color="gray")
    note(f"**2A selection**: picked programmatically by score = n_state_transitions + "
         f"n_N2W + n_MA among cycles with >=90% finite arteriole & calcium coverage; "
         f"winner = {rep_key} (score={cand.iloc[0]['score']:.0f}, "
         f"{cand.iloc[0]['n_trans']:.0f} transitions, {cand.iloc[0]['n_n2w']:.0f} N2W, "
         f"{cand.iloc[0]['n_ma']:.0f} MA).")
    return fig, rep_key, cyc


# --------------------------------------------------------------------------- #
# shared: per-mouse, per-state, per-cycle mean signal tables
# --------------------------------------------------------------------------- #
def cycle_state_means(signal):
    """One row per (key, state) with the mean of `signal` restricted to bins of
    that state, plus mouse/group."""
    rows = []
    for k, cyc in CYCLES.items():
        if signal not in cyc.sig:
            continue
        t, y = cyc.sig[signal]
        n_bins = len(cyc.labels)
        bm = C.bin_series(t, y, n_bins, how="mean")
        lab = np.asarray(cyc.labels, dtype=object)
        for state in C.STATES:
            sel = lab == state
            if sel.sum() == 0:
                continue
            v = bm[sel]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            rows.append(dict(key=k, mouse=cyc.info.mouse, group=cyc.info.group,
                             state=state, value=float(np.mean(v)), n_bins=int(sel.sum())))
    return pd.DataFrame(rows)


def mouse_state_means(cycle_df):
    """Average cycle-level state means to one value per mouse per state."""
    return cycle_df.groupby(["group", "mouse", "state"], as_index=False)["value"].mean()


# --------------------------------------------------------------------------- #
# panel 2B / 2C: paired dot plots by state + stats
# --------------------------------------------------------------------------- #
def paired_dot_plot(ax, mouse_df, label, color):
    piv = mouse_df.pivot(index="mouse", columns="state", values="value")
    for col in C.MAIN_STATES:
        if col not in piv:
            piv[col] = np.nan
    has_rem = "REM" in piv.columns and np.isfinite(piv["REM"]).any()
    for i, mouse in enumerate(piv.index):
        a, b = piv.loc[mouse, "Awake"], piv.loc[mouse, "NREM"]
        if np.isfinite(a) and np.isfinite(b):
            ax.plot([0, 1], [a, b], color="0.6", lw=1.0, alpha=0.8, zorder=1)
    for j, state in enumerate(C.MAIN_STATES):
        vals = piv[state].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        ax.scatter(np.full(vals.shape, j), vals, color=C.STATE_COL[state],
                  s=42, zorder=2, edgecolor="black", linewidth=0.4)
    if has_rem:
        rem = piv["REM"].to_numpy(float)
        rem = rem[np.isfinite(rem)]
        ax.scatter(np.full(rem.shape, 2.0), rem, color=C.STATE_COL["REM"],
                  s=36, marker="D", zorder=2, edgecolor="black", linewidth=0.4,
                  label="REM (descriptive)")
        ax.legend(fontsize=6, loc="upper left")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Awake", "NREM", "REM"], fontsize=8)
        ax.set_xlim(-0.5, 2.5)
    else:
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Awake", "NREM"], fontsize=8)
        ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(label, fontsize=8)
    ax.tick_params(labelsize=7)
    return piv


def run_state_stats(mouse_df, cycle_df, value_label):
    piv = mouse_df.pivot(index="mouse", columns="state", values="value")
    pt = C.paired_test(piv.get("Awake", pd.Series(dtype=float)),
                       piv.get("NREM", pd.Series(dtype=float)))
    cyc_sub = cycle_df[cycle_df["state"].isin(C.MAIN_STATES)].copy()
    cyc_sub["state"] = pd.Categorical(cyc_sub["state"], categories=C.MAIN_STATES)
    lmm_tab = C.lmm(cyc_sub, "value", "C(state)")
    return pt, lmm_tab, piv


def make_fig2bc():
    art_cyc = cycle_state_means("art")
    ca_cyc = cycle_state_means("ca")
    art_mouse = mouse_state_means(art_cyc)
    ca_mouse = mouse_state_means(ca_cyc)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    ax_b, ax_c, ax_ratio = axes

    paired_dot_plot(ax_b, art_mouse, C.SIG_LABEL["art"] + " (state mean)", C.SIG_COL["art"])
    ax_b.set_title("Fig. 2B — arteriole by state", fontsize=9)
    pt_art, lmm_art, piv_art = run_state_stats(art_mouse, art_cyc, "art")

    paired_dot_plot(ax_c, ca_mouse, C.SIG_LABEL["ca"] + " (state mean)", C.SIG_COL["ca"])
    ax_c.set_title("Fig. 2C — astro Ca2+ by state", fontsize=9)
    pt_ca, lmm_ca, piv_ca = run_state_stats(ca_mouse, ca_cyc, "ca")

    # Ca normalised to Awake per mouse (NREM/Awake ratio)
    ratio = (piv_ca["NREM"] / piv_ca["Awake"]).dropna()
    # ratio is only meaningful away from Ca values straddling 0; dF/F0 state means
    # can be negative, so also report the signed difference for context.
    ax_ratio.scatter(np.zeros(len(ratio)), ratio.to_numpy(), color=C.SIG_COL["ca"],
                     s=42, edgecolor="black", linewidth=0.4, zorder=2)
    ax_ratio.axhline(1.0, color="0.5", lw=1.0, ls="--")
    ax_ratio.set_xlim(-0.5, 0.5)
    ax_ratio.set_xticks([0])
    ax_ratio.set_xticklabels(["NREM / Awake"], fontsize=8)
    ax_ratio.set_ylabel("Ca2+ ratio (NREM / Awake)", fontsize=8)
    ax_ratio.set_title("Ca2+ normalised to Awake per mouse", fontsize=9)
    ax_ratio.tick_params(labelsize=7)
    note(f"**2B/C ratio caveat**: per-mouse dF/F0 state means can be near zero or "
         f"negative (median-referenced normalisation), so the NREM/Awake ratio is "
         f"unstable for mice whose Awake mean is close to 0; shown as Grant "
         f"requested but interpret the signed Awake-vs-NREM difference (2C) as primary.")

    for ax in (ax_b, ax_c, ax_ratio):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(C.PRELIM_STAMP, fontsize=6, color="gray", y=1.0)
    fig.text(0.01, 0.01, C.n_note(art_cyc), fontsize=6, color="gray")

    note(f"**2B**: arteriole dD/D0, Awake {pt_art['mean_a']:.4f} vs NREM "
         f"{pt_art['mean_b']:.4f} (n={pt_art['n']} mice), paired t p={pt_art['t_p']:.3g}, "
         f"Wilcoxon p={pt_art['wilcoxon_p']:.3g} (min attainable {pt_art['wilcoxon_min_p']:.3g}).")
    note(f"**2C**: astro Ca2+ dF/F0, Awake {pt_ca['mean_a']:.4f} vs NREM "
         f"{pt_ca['mean_b']:.4f} (n={pt_ca['n']} mice), paired t p={pt_ca['t_p']:.3g}, "
         f"Wilcoxon p={pt_ca['wilcoxon_p']:.3g} (min attainable {pt_ca['wilcoxon_min_p']:.3g}).")
    note(f"**2C ratio**: median NREM/Awake Ca2+ ratio = {np.nanmedian(ratio):.2f} "
         f"(n={ratio.size} mice with finite ratio).")

    return fig, dict(pt_art=pt_art, lmm_art=lmm_art, pt_ca=pt_ca, lmm_ca=lmm_ca,
                     ratio=ratio, art_cyc=art_cyc, ca_cyc=ca_cyc)


# --------------------------------------------------------------------------- #
# panel 2D: variability by state (SD of linearly detrended bout, bouts >=30 s)
# --------------------------------------------------------------------------- #
def bout_detrended_sd(signal, min_dur=30.0):
    rows = []
    for k, cyc in CYCLES.items():
        if signal not in cyc.sig:
            continue
        t, y = cyc.sig[signal]
        b = C.bouts(cyc.labels)
        b = b[(b["state"].isin(C.MAIN_STATES)) & (b["dur_s"] >= min_dur)]
        for r in b.itertuples():
            m = (t >= r.t_start) & (t < r.t_start + r.dur_s)
            seg = y[m]
            fin = np.isfinite(seg)
            if fin.sum() < 10:
                continue
            seg_f = seg.copy()
            idx = np.arange(len(seg_f))
            seg_f[~fin] = np.interp(idx[~fin], idx[fin], seg_f[fin]) if (~fin).any() else seg_f[~fin]
            det = sstats_detrend(seg_f)
            rows.append(dict(key=k, mouse=cyc.info.mouse, group=cyc.info.group,
                             state=r.state, sd=float(np.std(det)), dur_s=r.dur_s))
    return pd.DataFrame(rows)


def sstats_detrend(y):
    from scipy.signal import detrend
    return detrend(y, type="linear")


def make_fig2d():
    art_sd = bout_detrended_sd("art")
    ca_sd = bout_detrended_sd("ca")

    art_mouse_sd = art_sd.groupby(["group", "mouse", "state"], as_index=False)["sd"].mean()
    ca_mouse_sd = ca_sd.groupby(["group", "mouse", "state"], as_index=False)["sd"].mean()
    art_mouse_sd = art_mouse_sd.rename(columns={"sd": "value"})
    ca_mouse_sd = ca_mouse_sd.rename(columns={"sd": "value"})

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    paired_dot_plot(axes[0], art_mouse_sd, "SD of detrended " + C.SIG_LABEL["art"], C.SIG_COL["art"])
    axes[0].set_title("Fig. 2D — arteriole variability by state", fontsize=9)
    axes[0].set_ylim(bottom=0)
    paired_dot_plot(axes[1], ca_mouse_sd, "SD of detrended " + C.SIG_LABEL["ca"], C.SIG_COL["ca"])
    axes[1].set_title("Fig. 2D — Ca2+ variability by state", fontsize=9)
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    art_cyc_sd = art_sd.rename(columns={"sd": "value"})
    ca_cyc_sd = ca_sd.rename(columns={"sd": "value"})
    pt_art_sd, lmm_art_sd, _ = run_state_stats(art_mouse_sd, art_cyc_sd, "art_sd")
    pt_ca_sd, lmm_ca_sd, _ = run_state_stats(ca_mouse_sd, ca_cyc_sd, "ca_sd")

    note(f"**2D arteriole**: SD of detrended dD/D0 within bouts >=30 s, Awake "
         f"{pt_art_sd['mean_a']:.4f} vs NREM {pt_art_sd['mean_b']:.4f} "
         f"(n={pt_art_sd['n']} mice), paired t p={pt_art_sd['t_p']:.3g}, "
         f"Wilcoxon p={pt_art_sd['wilcoxon_p']:.3g}.")
    note(f"**2D Ca2+**: SD of detrended dF/F0 within bouts >=30 s, Awake "
         f"{pt_ca_sd['mean_a']:.4f} vs NREM {pt_ca_sd['mean_b']:.4f} "
         f"(n={pt_ca_sd['n']} mice), paired t p={pt_ca_sd['t_p']:.3g}, "
         f"Wilcoxon p={pt_ca_sd['wilcoxon_p']:.3g}.")
    note(f"**2D bout counts**: arteriole {len(art_sd)} bouts across "
         f"{art_sd['key'].nunique() if len(art_sd) else 0} cycles; Ca2+ "
         f"{len(ca_sd)} bouts across {ca_sd['key'].nunique() if len(ca_sd) else 0} cycles "
         f"(bouts >=30 s, Awake+NREM only).")

    fig.suptitle(C.PRELIM_STAMP, fontsize=6, color="gray", y=1.0)

    return fig, dict(pt_art_sd=pt_art_sd, lmm_art_sd=lmm_art_sd,
                     pt_ca_sd=pt_ca_sd, lmm_ca_sd=lmm_ca_sd)


# --------------------------------------------------------------------------- #
# panel 2E: Ca vs arteriole scatter (5-s bin means) + per-mouse Spearman rho
# --------------------------------------------------------------------------- #
def bin5_table():
    rows = []
    for k, cyc in CYCLES.items():
        if "art" not in cyc.sig or "ca" not in cyc.sig:
            continue
        n_bins = len(cyc.labels)
        t_art, y_art = cyc.sig["art"]
        t_ca, y_ca = cyc.sig["ca"]
        art_b = C.bin_series(t_art, y_art, n_bins)
        ca_b = C.bin_series(t_ca, y_ca, n_bins)
        lab = np.asarray(cyc.labels, dtype=object)
        for i in range(n_bins):
            if np.isfinite(art_b[i]) and np.isfinite(ca_b[i]):
                rows.append(dict(key=k, mouse=cyc.info.mouse, group=cyc.info.group,
                                 bin=i, art=art_b[i], ca=ca_b[i], state=lab[i]))
    return pd.DataFrame(rows)


def per_mouse_spearman(df, state=None):
    rows = []
    sub = df if state is None else df[df["state"] == state]
    for mouse, s in sub.groupby("mouse"):
        s2 = s.dropna(subset=["art", "ca"])
        if len(s2) < 8:
            rows.append(dict(mouse=mouse, rho=np.nan, p=np.nan, n=len(s2)))
            continue
        rho, p = sstats.spearmanr(s2["art"], s2["ca"])
        rows.append(dict(mouse=mouse, rho=rho, p=p, n=len(s2)))
    return pd.DataFrame(rows)


def make_fig2e():
    df = bin5_table()
    fig, (ax_sc, ax_rho) = plt.subplots(1, 2, figsize=(11, 4.6))

    if df.empty:
        ax_sc.text(0.5, 0.5, "no overlapping art+ca bins", ha="center", va="center")
        note("**2E**: could NOT build the scatter — no WT cycle has overlapping "
             "arteriole and calcium coverage.")
        return fig, dict(df=df)

    # downsample for plotting only (keep all data for the stats below)
    plot_df = df
    if len(df) > 4000:
        plot_df = df.sample(4000, random_state=0)
    for state in C.MAIN_STATES + ("REM",):
        sub = plot_df[plot_df["state"] == state]
        if sub.empty:
            continue
        ax_sc.scatter(sub["art"], sub["ca"], s=5, alpha=0.35, color=C.STATE_COL[state],
                     label=f"{state} (n={ (df['state']==state).sum() } bins)", linewidths=0)
    ax_sc.set_xlabel(C.SIG_LABEL["art"], fontsize=8)
    ax_sc.set_ylabel(C.SIG_LABEL["ca"], fontsize=8)
    ax_sc.set_title("Fig. 2E — Ca2+ vs arteriole (5-s bin means)", fontsize=9)
    ax_sc.legend(fontsize=6, markerscale=2, loc="best")
    ax_sc.tick_params(labelsize=7)

    whole = per_mouse_spearman(df).assign(scope="whole")
    awake = per_mouse_spearman(df, "Awake").assign(scope="Awake")
    nrem = per_mouse_spearman(df, "NREM").assign(scope="NREM")
    rho_tab = pd.concat([whole, awake, nrem], ignore_index=True)

    scopes = ["whole", "Awake", "NREM"]
    colors = {"whole": "0.3", "Awake": C.STATE_COL["Awake"], "NREM": C.STATE_COL["NREM"]}
    for j, scope in enumerate(scopes):
        sub = rho_tab[rho_tab["scope"] == scope]
        vals = sub["rho"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        jitter = RNG.normal(0, 0.05, size=vals.shape)
        ax_rho.scatter(np.full(vals.shape, j) + jitter, vals, color=colors[scope],
                      s=36, edgecolor="black", linewidth=0.3, zorder=2)
        if vals.size:
            ax_rho.plot([j - 0.2, j + 0.2], [np.median(vals)] * 2, color="black", lw=1.5, zorder=3)
    ax_rho.axhline(0, color="0.6", lw=0.8, ls="--")
    ax_rho.set_xticks(range(len(scopes)))
    ax_rho.set_xticklabels(scopes, fontsize=8)
    ax_rho.set_ylabel("per-mouse Spearman rho (Ca2+ vs arteriole)", fontsize=8)
    ax_rho.set_title("Per-mouse coupling by scope", fontsize=9)
    ax_rho.tick_params(labelsize=7)

    for ax in (ax_sc, ax_rho):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(C.PRELIM_STAMP, fontsize=6, color="gray", y=1.0)

    for scope in scopes:
        sub = rho_tab[rho_tab["scope"] == scope]
        vals = sub["rho"].dropna().to_numpy(float)
        note(f"**2E rho ({scope})**: median rho={np.median(vals) if vals.size else float('nan'):.2f} "
             f"across n={vals.size} mice.")

    return fig, dict(rho_tab=rho_tab, bin_df=df)


# --------------------------------------------------------------------------- #
# panel 2F: cross-correlation Ca vs arteriole, lags -30..+30 s
# --------------------------------------------------------------------------- #
FS_XC = 1.955
LAG_MAX = 30.0


def xcorr_lags_r(ca, art, fs, lag_max=LAG_MAX):
    """Pearson correlation of ca(t+lag) vs art(t) across lags, NaN-safe via
    pairwise-complete samples at each lag. Returns (lags_s, r)."""
    n = len(ca)
    max_shift = int(round(lag_max * fs))
    lags = np.arange(-max_shift, max_shift + 1)
    r = np.full(lags.shape, np.nan)
    for i, lag in enumerate(lags):
        if lag >= 0:
            a = ca[lag:]
            b = art[: n - lag]
        else:
            a = ca[: n + lag]
            b = art[-lag:]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 20:
            continue
        if np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
            continue
        r[i] = np.corrcoef(a[ok], b[ok])[0, 1]
    return lags / fs, r


def state_pure_stretches(cyc, state, min_dur=60.0):
    b = C.bouts(cyc.labels)
    return b[(b["state"] == state) & (b["dur_s"] >= min_dur)]


def mouse_xcorr(scope):
    """scope in {'whole', 'Awake', 'NREM'}. Returns df: mouse, peak_r, peak_lag."""
    rows = []
    for mouse in MICE:
        mouse_keys = [k for k in KEYS if CYCLES[k].info.mouse == mouse]
        all_r = []
        lags_ref = None
        for k in mouse_keys:
            cyc = CYCLES[k]
            if "art" not in cyc.sig or "ca" not in cyc.sig:
                continue
            t_art, y_art = cyc.sig["art"]
            t_ca, y_ca = cyc.sig["ca"]
            if scope == "whole":
                segments = [(0.0, cyc.duration)]
            else:
                b = state_pure_stretches(cyc, scope, min_dur=60.0)
                segments = [(r.t_start, r.t_start + r.dur_s) for r in b.itertuples()]
            for t0, t1 in segments:
                g = np.arange(t0, t1, 1.0 / FS_XC)
                if len(g) < int(round(60 * FS_XC)):
                    continue
                ca_i = np.interp(g, t_ca, y_ca, left=np.nan, right=np.nan)
                art_i = np.interp(g, t_art, y_art, left=np.nan, right=np.nan)
                lags, r = xcorr_lags_r(ca_i, art_i, FS_XC)
                if np.isfinite(r).sum() == 0:
                    continue
                lags_ref = lags
                all_r.append(r)
        if not all_r:
            rows.append(dict(mouse=mouse, scope=scope, peak_r=np.nan, peak_lag=np.nan, n_segments=0))
            continue
        mean_r = np.nanmean(np.vstack(all_r), axis=0)
        i_peak = np.nanargmax(np.abs(mean_r)) if np.isfinite(mean_r).any() else None
        if i_peak is None:
            rows.append(dict(mouse=mouse, scope=scope, peak_r=np.nan, peak_lag=np.nan,
                             n_segments=len(all_r)))
            continue
        rows.append(dict(mouse=mouse, scope=scope, peak_r=float(mean_r[i_peak]),
                         peak_lag=float(lags_ref[i_peak]), n_segments=len(all_r),
                         mean_r_curve=mean_r, lags=lags_ref))
    return pd.DataFrame(rows)


def make_fig2f():
    scopes = ["whole", "Awake", "NREM"]
    results = {s: mouse_xcorr(s) for s in scopes}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax_curve, ax_dot = axes

    colors = {"whole": "0.3", "Awake": C.STATE_COL["Awake"], "NREM": C.STATE_COL["NREM"]}
    for scope in scopes:
        df = results[scope]
        curves = [row.mean_r_curve for row in df.itertuples() if hasattr(row, "mean_r_curve")
                 and isinstance(row.mean_r_curve, np.ndarray)]
        lags_list = [row.lags for row in df.itertuples() if hasattr(row, "lags")
                    and isinstance(row.lags, np.ndarray)]
        if not curves:
            continue
        lags = lags_list[0]
        M = np.vstack([np.interp(lags, lg, c) for c, lg in zip(curves, lags_list)])
        mean_curve = np.nanmean(M, axis=0)
        sem = np.nanstd(M, axis=0) / max(1, np.sqrt(M.shape[0]))
        ax_curve.plot(lags, mean_curve, color=colors[scope], lw=1.6, label=f"{scope} (n={M.shape[0]} mice)")
        ax_curve.fill_between(lags, mean_curve - sem, mean_curve + sem, color=colors[scope], alpha=0.2)
    ax_curve.axvline(0, color="0.6", lw=0.8, ls="--")
    ax_curve.axhline(0, color="0.6", lw=0.8, ls="--")
    ax_curve.set_xlabel("lag (s), Ca2+ shifted relative to arteriole", fontsize=8)
    ax_curve.set_ylabel("mean cross-correlation r", fontsize=8)
    ax_curve.set_title("Fig. 2F — Ca2+ vs arteriole cross-correlation", fontsize=9)
    ax_curve.legend(fontsize=7)
    ax_curve.tick_params(labelsize=7)

    for j, scope in enumerate(scopes):
        df = results[scope]
        vals = df["peak_r"].dropna().to_numpy(float)
        jitter = RNG.normal(0, 0.05, size=vals.shape)
        ax_dot.scatter(np.full(vals.shape, j) + jitter, vals, color=colors[scope],
                      s=36, edgecolor="black", linewidth=0.3, zorder=2)
        if vals.size:
            ax_dot.plot([j - 0.2, j + 0.2], [np.median(vals)] * 2, color="black", lw=1.5, zorder=3)
    ax_dot.axhline(0, color="0.6", lw=0.8, ls="--")
    ax_dot.set_xticks(range(len(scopes)))
    ax_dot.set_xticklabels(scopes, fontsize=8)
    ax_dot.set_ylabel("peak cross-correlation r (per mouse)", fontsize=8)
    ax_dot.set_title("Peak r by scope", fontsize=9)
    ax_dot.tick_params(labelsize=7)

    for ax in (ax_curve, ax_dot):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(C.PRELIM_STAMP, fontsize=6, color="gray", y=1.0)

    for scope in scopes:
        df = results[scope]
        pr = df["peak_r"].dropna()
        pl = df.loc[df["peak_r"].notna(), "peak_lag"]
        if pr.empty:
            note(f"**2F ({scope})**: could NOT compute — no mouse had a qualifying "
                 f"segment (>=60 s state-pure stretch at ~1.955 Hz with art+ca).")
            continue
        note(f"**2F ({scope})**: median peak r={pr.median():.2f} "
             f"(range {pr.min():.2f} to {pr.max():.2f}), median lag at peak="
             f"{pl.median():.1f} s, n={pr.size} mice.")

    tab = pd.concat([results[s][["mouse", "scope", "peak_r", "peak_lag", "n_segments"]]
                    for s in scopes], ignore_index=True)
    return fig, dict(table=tab)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    warnings.filterwarnings("ignore")
    C.ensure_dirs()
    note("# Figure 2 — state-dependent astrocyte Ca2+ and arteriole diameter (WT, RSC)")
    note("")
    note(f"n WT cycles = {len(KEYS)}; n WT mice (recording days) = {len(MICE)}.")
    note("")

    fig_a, rep_key, rep_cyc = make_fig2a()
    if fig_a is not None:
        C.save_fig(fig_a, "fig2_A_traces")
        plt.close(fig_a)

    fig_bc, res_bc = make_fig2bc()
    fig_d, res_d = make_fig2d()
    fig_e, res_e = make_fig2e()
    fig_f, res_f = make_fig2f()

    # combine B-F into one multi-panel figure file as required by the brief
    # (fig2_B-F_quant); each quant sub-figure is saved under that shared
    # prefix as its own file so every panel stays legible at readable size.
    C.save_fig(fig_bc, "fig2_B-F_quant_BC_state_means")
    C.save_fig(fig_d, "fig2_B-F_quant_D_variability")
    C.save_fig(fig_e, "fig2_B-F_quant_E_coupling")
    C.save_fig(fig_f, "fig2_B-F_quant_F_xcorr")
    plt.close(fig_bc)
    plt.close(fig_d)
    plt.close(fig_e)
    plt.close(fig_f)

    # tables
    C.save_table(res_bc["art_cyc"], "fig2_B_art_state_means_bycycle")
    C.save_table(res_bc["ca_cyc"], "fig2_C_ca_state_means_bycycle")
    if res_bc["lmm_art"] is not None:
        C.save_table(res_bc["lmm_art"], "fig2_B_art_lmm")
    if res_bc["lmm_ca"] is not None:
        C.save_table(res_bc["lmm_ca"], "fig2_C_ca_lmm")
    if res_d["lmm_art_sd"] is not None:
        C.save_table(res_d["lmm_art_sd"], "fig2_D_art_sd_lmm")
    if res_d["lmm_ca_sd"] is not None:
        C.save_table(res_d["lmm_ca_sd"], "fig2_D_ca_sd_lmm")
    C.save_table(res_e["rho_tab"], "fig2_E_spearman_bymouse")
    C.save_table(res_f["table"], "fig2_F_xcorr_bymouse")

    note("")
    note("## Reproduce")
    note("`./venv/bin/python -B scripts/figs_prelim/fig2.py` from the repo root.")
    note("")
    note("## LMM notes")
    for label, tab in (("2B arteriole", res_bc["lmm_art"]), ("2C Ca2+", res_bc["lmm_ca"]),
                       ("2D arteriole SD", res_d["lmm_art_sd"]), ("2D Ca2+ SD", res_d["lmm_ca_sd"])):
        if tab is None:
            note(f"- {label}: LMM could not be fit (too few mice/observations); see paired test only.")
        elif "FIT_FAILED" in tab.get("term", pd.Series(dtype=str)).values:
            note(f"- {label}: LMM fit failed ({tab.iloc[0].get('note', '')}).")
        else:
            note(f"- {label}: LMM fit OK, see fig2_*_lmm.csv.")

    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text("\n".join(notes_lines) + "\n")

    print("Saved figures to", C.FIG_DIR)
    print("Saved tables to", C.TAB_DIR)
    print("Saved notes to", NOTES_PATH)
    print("\n--- headline numbers ---")
    pt_art = res_bc["pt_art"]; pt_ca = res_bc["pt_ca"]
    print(f"Arteriole Awake {pt_art['mean_a']:.4f} vs NREM {pt_art['mean_b']:.4f}, "
          f"n={pt_art['n']} mice, paired t p={pt_art['t_p']:.3g}")
    print(f"Ca2+ Awake {pt_ca['mean_a']:.4f} vs NREM {pt_ca['mean_b']:.4f}, "
          f"n={pt_ca['n']} mice, paired t p={pt_ca['t_p']:.3g}")


if __name__ == "__main__":
    main()
