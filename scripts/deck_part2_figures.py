"""Figures for 'Findings part 2' (2026-10-09): IP3R2 scramble vs KD at NREM -> wake.

Same events and normalisation as part 1 (deck_part1_figures.py):
  events     final labels (PRELIM_LABEL_SET=relabel_2026-10-09), manual only,
             common.response_events type N2W, t0 = pupil onset, every event kept
  arteriole  ΔD/D0 = (D - D0)/D0, D = sqrt(area), D0 = median of -20..0 s
  Ca2+       ΔF/F0 the same way
  pupil      robust z within cycle, minus its -20..0 s median; samples with
             |z| > 6 removed as artifacts (KD 2023.08.26 has spikes to 19 z)
Plot window -30..60 s. Statistics: recording day = unit (Welch t on day means,
Mann-Whitney alongside). Rank correlations are within day, events as units.

Output: data/RESULTS_deck_part2_2026-10-09/
  figures/1_psth.png  2_heatmaps.png  3_summary.png  4_per_day.png
          5_coupling.png  6_arousal.png
  day_values.csv  tests.csv  coupling_arousal.csv  events.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "figs_prelim"))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import deck_part1_figures as P1  # noqa: E402

OUT = C.ROOT / "data" / "RESULTS_deck_part2_2026-10-09"
FIG = OUT / "figures"
RG, PRE, VIEW = P1.RG, P1.PRE, P1.VIEW
RESP = (0.0, 15.0)
GROUPS = ("SCR", "KD")
GLAB = {"SCR": "IP3R2 Scramble", "KD": "IP3R2 KD"}
GCOL = {"SCR": "#2E7D32", "KD": "#C62828"}
GCMAP = {"SCR": plt.get_cmap("Greens"), "KD": plt.get_cmap("Reds")}
SIGS = ("pupil", "ca", "art")
SLAB = {"pupil": "Pupil (z)", "ca": "Ca²⁺ ΔF/F0 (%)", "art": "Arteriole ΔD/D0 (%)"}
STITLE = {"pupil": "Pupil", "ca": "Astrocyte Ca²⁺ (ΔF/F0)", "art": "Arteriole (ΔD/D0)"}
TERT = ("low", "mid", "high")
PUPIL_ART_Z = 6.0      # |robust z| above this = artifact sample, removed
TERT_COL = ("#90CAF9", "#1E88E5", "#0D47A1")

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


# ----------------------------------------------------------------- data
def smooth3(M):
    return pd.DataFrame(M.T).rolling(3, center=True, min_periods=1).mean().to_numpy().T


def load():
    keys = C.keys_for(list(GROUPS))
    ev = C.response_events(keys)
    ev = ev[ev["type"] == "N2W"][["type", "key", "mouse", "t0"]].reset_index(drop=True)
    ev["group"] = np.where(ev["key"].str.contains("_KD_"), "KD", "SCR")
    ev["day"] = ev["mouse"].str.split("_").str[-1]
    for k in ev["key"].unique():                     # pupil artifact mask (KD 2023.08.26: spikes to 19 z)
        sg = P1.signals(k)
        if "pupil" in sg:
            t, y = sg["pupil"]
            y = np.where(np.abs(y) > PUPIL_ART_Z, np.nan, y)
            sg["pupil"] = (t, y)
    E = {s: P1.epoch_matrix(ev, s) for s in SIGS}
    E["art_s"] = smooth3(E["art"])
    m = (RG >= RESP[0]) & (RG <= RESP[1])
    with np.errstate(all="ignore"):
        ev["pupil_rise"] = np.nanmax(E["pupil"][:, m], 1)
        for s, src in (("ca", "ca"), ("art", "art_s")):
            ev[f"{s}_peak"] = np.nanmax(E[src][:, m], 1)
            ev[f"{s}_mean"] = np.nanmean(E[s][:, m], 1)
            Y = np.where(np.isfinite(E[s][:, m]), E[s][:, m], 0.0)
            ok = np.isfinite(E[s][:, m]).mean(1) > 0.8
            ev[f"{s}_auc"] = np.where(ok, np.trapezoid(Y, RG[m], axis=1), np.nan)
    return ev, E


def day_order(ev, g):
    return sorted(ev.loc[ev.group == g, "day"].unique())


def day_means(ev, M, g):
    """{day: mean trace across that day's events}"""
    out = {}
    for d in day_order(ev, g):
        idx = np.flatnonzero(((ev.group == g) & (ev.day == d)).to_numpy())
        X = M[idx]
        X = X[np.isfinite(X).any(1)]
        if len(X):
            with np.errstate(all="ignore"):
                out[d] = np.nanmean(X, 0)
    return out


def scale_bar(ax, dx, dy, ylab):
    """L-shaped bar in an empty strip below / left of the data."""
    y0, y1 = ax.get_ylim()
    ya = y0 - 0.12 * (y1 - y0)
    ax.set_ylim(ya - 0.1 * (y1 - y0), y1)
    xa = VIEW[0] - 6
    ax.set_xlim(VIEW[0] - 14, VIEW[1] + 1)
    ax.plot([xa, xa + dx], [ya, ya], color="k", lw=2.5, clip_on=False)
    ax.plot([xa, xa], [ya, ya + dy], color="k", lw=2.5, clip_on=False)
    ax.text(xa + dx / 2, ya - 0.02 * (y1 - y0), f"{dx:g} s", ha="center", va="top", fontsize=11)
    ax.text(xa - 1, ya + dy / 2, ylab, ha="right", va="center", fontsize=11)


def strip(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])


def pfmt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.2f}" if p >= 0.01 else f"p = {p:.3f}"


# ----------------------------------------------------------------- 1 PSTH
def fig1_psth(ev, E):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    bars = {"pupil": (1.0, "1 z"), "ca": (20.0, "20%"), "art": (2.0, "2%")}
    for ax, s in zip(axs, SIGS):
        M = E["art_s"] if s == "art" else E[s]
        for g in GROUPS:
            D = np.array(list(day_means(ev, M, g).values()))
            mu = np.nanmean(D, 0); se = np.nanstd(D, 0, ddof=1) / np.sqrt(len(D))
            ax.fill_between(RG, mu - se, mu + se, color=GCOL[g], alpha=0.25, lw=0)
            ax.plot(RG, mu, color=GCOL[g], lw=2,
                    label=f"{GLAB[g]} ({len(D)} days, {int((ev.group == g).sum())} events)")
        ax.axvline(0, color="0.5", ls="--", lw=1.2)
        ax.set_xlim(VIEW)
        ax.set_title(STITLE[s], fontsize=15, weight="bold")
        strip(ax)
        dy, lab = bars[s]
        scale_bar(ax, 10.0, dy, lab)
    axs[0].legend(frameon=False, fontsize=10, loc="upper right")
    fig.savefig(FIG / "1_psth.png", dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------- 2 heatmaps
def fig2_heatmaps(ev, E):
    order, bounds = {}, {}
    for g in GROUPS:
        idx = []
        for d in day_order(ev, g):
            sel = np.flatnonzero(((ev.group == g) & (ev.day == d)).to_numpy())
            idx += list(sel[np.argsort(-ev.pupil_rise.to_numpy()[sel])])
        order[g] = np.array(idx)
    n = {g: len(order[g]) for g in GROUPS}
    lims = {}
    for s in SIGS:
        M = E["art_s"] if s == "art" else E[s]
        allv = np.concatenate([M[order[g]].ravel() for g in GROUPS])
        lims[s] = np.nanpercentile(np.abs(allv), 97)
    fig = plt.figure(figsize=(16, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[n["SCR"], n["KD"]], width_ratios=[0.12, 1, 1, 1])
    for r, g in enumerate(GROUPS):
        days = ev.day.to_numpy()[order[g]]
        ud = day_order(ev, g)
        cols = {d: GCMAP[g](0.35 + 0.6 * i / max(1, len(ud) - 1)) for i, d in enumerate(ud)}
        axd = fig.add_subplot(gs[r, 0])
        axd.imshow(np.array([[cols[d]] for d in days]), aspect="auto", interpolation="nearest")
        strip(axd)
        for d in ud:
            ii = np.flatnonzero(days == d)
            axd.text(-0.7, ii.mean(), d[2:], ha="right", va="center", fontsize=9)
        axd.set_ylabel(f"{GLAB[g]}\n{n[g]} events, {len(ud)} days", fontsize=11, weight="bold", color=GCOL[g],
                       labelpad=55)
        axd.yaxis.set_label_position("left"); axd.yaxis.label.set_visible(True)
        for c, s in enumerate(SIGS):
            ax = fig.add_subplot(gs[r, c + 1])
            M = (E["art_s"] if s == "art" else E[s])[order[g]]
            cmap = "PuOr_r" if s == "pupil" else "RdBu_r"
            im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-lims[s], vmax=lims[s], interpolation="nearest",
                           extent=[VIEW[0], VIEW[1], len(M) - 0.5, -0.5])
            ax.axvline(0, color="k", ls="--", lw=1)
            for d in ud[1:]:
                ax.axhline(np.flatnonzero(days == d)[0] - 0.5, color="0.3", lw=0.6)
            ax.set_yticks([])
            if r == 0:
                ax.set_title(STITLE[s], fontsize=14, weight="bold")
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("time from pupil onset (s)")
                fig.colorbar(im, ax=ax, location="bottom", shrink=0.7, label=SLAB[s])
    fig.savefig(FIG / "2_heatmaps.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- 3 summary
METRICS = [("ca_peak", "Ca²⁺ peak", "Peak ΔF/F0 (%)"), ("ca_auc", "Ca²⁺ AUC", "AUC 0–15 s (%·s)"),
           ("art_peak", "Arteriole peak", "Peak ΔD/D0 (%)"), ("art_auc", "Arteriole AUC", "AUC 0–15 s (%·s)")]


def tests(ev):
    D = ev.groupby(["group", "day"]).agg(
        n=("t0", "size"), **{c: (c, "mean") for c in
                            ["pupil_rise", "ca_peak", "ca_auc", "ca_mean", "art_peak", "art_auc", "art_mean"]}
    ).reset_index()
    D.to_csv(OUT / "day_values.csv", index=False)
    rows = []
    for c in ["pupil_rise", "ca_peak", "ca_auc", "ca_mean", "art_peak", "art_auc", "art_mean"]:
        a = D.loc[D.group == "SCR", c].to_numpy(); b = D.loc[D.group == "KD", c].to_numpy()
        r = C.group_test(a, b)
        rows.append(dict(metric=c, scr_days=len(a), kd_days=len(b), scr_mean=a.mean(), kd_mean=b.mean(),
                         hedges_g=r["hedges_g"], welch_p=r["welch_p"], mw_p=r["mw_p"]))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "tests.csv", index=False)
    return D, T


def fig3_summary(ev, D, T):
    rng = np.random.default_rng(1)
    fig, axs = plt.subplots(1, 4, figsize=(16, 5.2), constrained_layout=True)
    for ax, (c, title, ylab) in zip(axs, METRICS):
        top = []
        for x, g in enumerate(GROUPS):
            ud = day_order(ev, g)
            cols = {d: GCMAP[g](0.35 + 0.6 * i / max(1, len(ud) - 1)) for i, d in enumerate(ud)}
            dm = D[D.group == g].set_index("day")[c]
            ax.bar(x, dm.mean(), 0.6, color=GCOL[g], alpha=0.35, edgecolor=GCOL[g])
            ax.errorbar(x, dm.mean(), dm.std(ddof=1) / np.sqrt(len(dm)), color=GCOL[g], capsize=5, lw=1.5)
            e = ev[ev.group == g]
            ax.scatter(x + rng.uniform(-0.22, 0.22, len(e)), e[c], s=22, facecolors="none",
                       edgecolors=[cols[d] for d in e.day], lw=1.1, alpha=0.8, zorder=3)
            ax.scatter(np.full(len(dm), x) + np.linspace(-0.08, 0.08, len(dm)), dm.values, s=120,
                       color=[cols[d] for d in dm.index], edgecolor="white", lw=1, zorder=4)
            top.append(np.nanpercentile(e[c], 98))
        ax.axhline(0, color="k", lw=1)
        p = T.set_index("metric").loc[c, "welch_p"]
        y = max(top) * 1.08
        ax.plot([0, 0, 1, 1], [y * 0.97, y, y, y * 0.97], color="k", lw=1.5)
        ax.text(0.5, y * 1.02, ("ns  " if p >= 0.05 else "*  ") + f"({pfmt(p)})", ha="center", va="bottom", fontsize=11)
        ax.set_ylim(top=y * 1.15)
        ax.set_xticks([0, 1], [GLAB[g] for g in GROUPS], rotation=20)
        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_ylabel(ylab)
    fig.savefig(FIG / "3_summary.png", dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------- 4 per day
def fig4_per_day(ev, E, D):
    cols = [(g, d) for g in GROUPS for d in day_order(ev, g)]
    fig, axs = plt.subplots(2, len(cols), figsize=(2.3 * len(cols), 5.4), sharex=True, sharey="row",
                            constrained_layout=True)
    for j, (g, d) in enumerate(cols):
        idx = np.flatnonzero(((ev.group == g) & (ev.day == d)).to_numpy())
        for i, s in enumerate(("ca", "art")):
            ax = axs[i, j]
            X = (E["art_s"] if s == "art" else E[s])[idx]
            mu, se, nn = P1.msem(X)
            ax.fill_between(RG, mu - se, mu + se, color=GCOL[g], alpha=0.25, lw=0)
            ax.plot(RG, mu, color=GCOL[g], lw=1.5)
            P1.shade(ax)
            if j == 0:
                ax.set_ylabel(SLAB[s])
            if i == 1:
                ax.set_xlabel("time (s)")
                v = D[(D.group == g) & (D.day == d)].art_mean.iloc[0]
                ax.text(0.97, 0.05, f"{v:+.1f}%", transform=ax.transAxes, ha="right", fontsize=9)
            else:
                v = D[(D.group == g) & (D.day == d)].ca_mean.iloc[0]
                ax.set_title(f"{d}\n(n={len(idx)})", fontsize=10, color=GCOL[g], weight="bold")
                ax.text(0.97, 0.9, f"{v:+.0f}%", transform=ax.transAxes, ha="right", fontsize=9)
    fig.suptitle("Each recording day: mean ± SEM across events. Green = scramble, red = KD. "
                 "Corner numbers = mean 0–15 s.", fontsize=10.5, x=0.01, ha="left")
    fig.savefig(FIG / "4_per_day.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- 5 coupling, 6 arousal
def within_day_rho(d, x, y):
    v = d[["day", x, y]].dropna()
    r = v.groupby("day")[[x, y]].rank(pct=True)
    rho, p = stats.spearmanr(r[x], r[y])
    return rho, p, len(v)


def fig5_coupling(ev):
    res = []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, constrained_layout=True)
    for ax, g in zip(axs, GROUPS):
        e = ev[ev.group == g]
        ud = day_order(ev, g)
        cols = {d: GCMAP[g](0.35 + 0.6 * i / max(1, len(ud) - 1)) for i, d in enumerate(ud)}
        ax.scatter(e.ca_mean, e.art_mean, s=34, color=[cols[d] for d in e.day], edgecolor="white", lw=0.5)
        rho, p, n = within_day_rho(e, "ca_mean", "art_mean")
        res.append(dict(group=g, x="ca_mean", y="art_mean", rho=rho, p=p, n_events=n, n_days=len(ud)))
        ax.axhline(0, color="0.7", lw=0.8); ax.axvline(0, color="0.7", lw=0.8)
        ax.set_title(f"{GLAB[g]}  ({n} events, {len(ud)} days)\nwithin-day ρ = {rho:+.2f} ({pfmt(p)})",
                     fontsize=12, weight="bold", color=GCOL[g])
        ax.set_xlabel("Ca²⁺ ΔF/F0, mean 0–15 s (%)")
    axs[0].set_ylabel("Arteriole ΔD/D0, mean 0–15 s (%)")
    fig.savefig(FIG / "5_coupling.png", dpi=170)
    plt.close(fig)
    return res


def fig6_arousal(ev, E):
    res = []
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.8), sharex=True, constrained_layout=True)
    for row, g in enumerate(GROUPS):
        d = ev[(ev.group == g) & ev.pupil_rise.notna()].copy()
        d["tert"] = np.digitize(d.groupby("day").pupil_rise.rank(pct=True), [1 / 3, 2 / 3 + 1e-9])
        for ax, s in zip(axs[row], SIGS):
            M = E["art_s"] if s == "art" else E[s]
            for k in range(3):
                mu, se, nn = P1.msem(M[d.index[d.tert == k].to_numpy()])
                ax.plot(RG, mu, color=TERT_COL[k], lw=1.8, label=f"{TERT[k]} pupil rise (n={nn})")
                ax.fill_between(RG, mu - se, mu + se, color=TERT_COL[k], alpha=0.18, lw=0)
            P1.shade(ax)
            ax.set_ylabel(SLAB[s])
            if row == 1:
                ax.set_xlabel("time from pupil onset (s)")
            if s == "pupil":
                ax.set_title(f"{GLAB[g]}: {d.day.nunique()} days, {len(d)} events (thirds within each day)",
                             fontsize=10, color=GCOL[g], weight="bold")
            else:
                rho, p, n = within_day_rho(d, "pupil_rise", f"{s}_mean")
                res.append(dict(group=g, x="pupil_rise", y=f"{s}_mean", rho=rho, p=p, n_events=n,
                                n_days=d.day.nunique()))
                ax.set_title(f"within-day ρ = {rho:+.2f} ({pfmt(p)})", fontsize=10)
            ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.savefig(FIG / "6_arousal.png", dpi=150)
    plt.close(fig)
    return res


def main():
    C.set_label_mode("manual")
    FIG.mkdir(parents=True, exist_ok=True)
    ev, E = load()
    ev.to_csv(OUT / "events.csv", index=False)
    D, T = tests(ev)
    fig1_psth(ev, E)
    fig2_heatmaps(ev, E)
    fig3_summary(ev, D, T)
    fig4_per_day(ev, E, D)
    R = pd.DataFrame(fig5_coupling(ev) + fig6_arousal(ev, E))
    R.to_csv(OUT / "coupling_arousal.csv", index=False)
    pd.set_option("display.width", 200)
    print(ev.groupby("group").agg(events=("t0", "size"), days=("day", "nunique")).to_string())
    print(D.round(2).to_string(index=False))
    print(T.round(4).to_string(index=False))
    print(R.round(3).to_string(index=False))
    print("->", FIG)


if __name__ == "__main__":
    main()
