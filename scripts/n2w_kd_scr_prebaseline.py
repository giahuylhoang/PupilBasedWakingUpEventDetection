"""KD vs scramble wake-up arteriole response with the textbook denominator.

    ΔD/D0(t) = (D(t) - D_pre) / D_pre,   D_pre = mean D over -5..0 s before onset
    D = sqrt(cross-sectional area)

(The figure set uses (D(t) - D_pre) / D_cycle-median; both are computed here so
the change can be seen.) Events: NREM->wake, pupil-anchored, non-overlapping
(common.response_events), manual labels, label set from PRELIM_LABEL_SET
(run with relabel_2026-10-09 = final labels). Unit for statistics = recording day.

Usage:
    PRELIM_LABEL_SET=relabel_2026-10-09 <python> scripts/n2w_kd_scr_prebaseline.py
Outputs: data/RESULTS_kd_scr_prebaseline_2026-10-09/
    figures/A_per_day.png       mean ± SEM across events, -5..15 s, every SCR / KD day
    figures/B_group.png         group mean ± SEM across days, and KD - SCR with 95% CI
    tables/per_day.csv, group_tests.csv, events.csv
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

sys.path.insert(0, str(Path(__file__).resolve().parent / "figs_prelim"))
import common as C  # noqa: E402

import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pre", type=float, nargs=2, default=[-5.0, 0.0], help="baseline window (s)")
ap.add_argument("--stat", choices=["mean", "median"], default="mean", help="baseline statistic")
ap.add_argument("--view", type=float, nargs=2, default=[-5.0, 15.0], help="plotted window (s)")
ap.add_argument("--signal", choices=["art", "ca"], default="art")
ap.add_argument("--set", choices=["kd_scr", "wt"], default="kd_scr",
                help="kd_scr: scramble vs KD;  wt: WT split into RSC 2023 vs GCaMP8m (2025-26)")
ARGS = ap.parse_args()
PRE_WIN, WIN = tuple(ARGS.pre), tuple(ARGS.view)
_base = C.ROOT / "data" / {"kd_scr": "RESULTS_kd_scr_prebaseline_2026-10-09",
                           "wt": "RESULTS_wt_prebaseline_2026-10-09"}[ARGS.set]
OUT = _base if (PRE_WIN, ARGS.stat, WIN) == ((-5.0, 0.0), "mean", (-5.0, 15.0)) else \
    _base / f"pre{ARGS.stat}_{PRE_WIN[0]:g}_{PRE_WIN[1]:g}__view_{WIN[0]:g}_{WIN[1]:g}"
if ARGS.signal != "art":
    OUT = OUT / ARGS.signal
SIG = ARGS.signal
Q = {"art": "ΔD/D_pre", "ca": "ΔF/F_pre"}[SIG]                 # plotted quantity
SIGNAME = {"art": "arteriole", "ca": "Ca²⁺"}[SIG]
VAR = {"art": "D", "ca": "F"}[SIG]
BASE_LABEL = f"{ARGS.stat} {PRE_WIN[0]:g}…{PRE_WIN[1]:g} s"
if ARGS.set == "kd_scr":
    GROUPS = ("SCR", "KD")
    COL = {g: COL[g] for g in GROUPS}
    LAB = {g: LAB[g] for g in GROUPS}
else:
    GROUPS = ("RSC2023", "GCaMP8m")
    COL = {"RSC2023": "#424242", "GCaMP8m": "#2E7D32"}
    LAB = {"RSC2023": "WT RSC 2023", "GCaMP8m": "WT GCaMP8m 2025-26"}
G0, G1 = GROUPS


def load_events():
    if ARGS.set == "kd_scr":
        ev = C.response_events(C.keys_for(list(GROUPS)))
    else:
        ev = C.response_events(C.keys_for(["WT"]))
        ev["group"] = np.where(ev["key"].str.startswith("Extra_GCaMP8m"), "GCaMP8m", "RSC2023")
    return ev[ev["type"] == "N2W"].reset_index(drop=True)


def epochs(ev):
    """Per event, two versions of the arteriole response on the epoch grid."""
    rg, Y = C.epochs_for(ev, SIG, baseline=False)          # Y = X/X_median - 1 (X = D or F)
    pre = (rg >= PRE_WIN[0]) & (rg < PRE_WIN[1])
    f = np.nanmedian if ARGS.stat == "median" else np.nanmean
    ypre = f(Y[:, pre], axis=1, keepdims=True)
    pre_norm = (Y - ypre) / (1 + ypre)                       # (D - D_pre) / D_pre
    med_norm = Y - ypre                                      # (D - D_pre) / D_median (old)
    return rg, pre_norm, med_norm, (1 + ypre[:, 0])          # last = D_pre / D_median


def metrics(rg, M):
    r = (rg >= C.RESP_WIN[0]) & (rg <= C.RESP_WIN[1])
    seg = M[:, r]
    dt = np.median(np.diff(rg))
    ok = np.isfinite(seg).any(1)
    auc = np.where(ok, np.nansum(seg, 1) * dt, np.nan)
    peak = np.where(ok, np.nanmax(np.where(np.isfinite(seg), seg, -np.inf), 1), np.nan)
    mean = np.where(ok, np.nanmean(seg, 1), np.nan)
    return auc, peak, mean


def day_mean_traces(ev, M, rg, m):
    days = {}
    for day, idx in ev.groupby("mouse").indices.items():
        X = M[idx][:, m]
        X = X[np.isfinite(X).any(1)]
        days[day] = X
    return days


def main():
    C.set_label_mode("manual")
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    ev = load_events()
    rg, P, Mold, ratio = epochs(ev)

    a_new, p_new, mu_new = metrics(rg, P)
    a_old, p_old, mu_old = metrics(rg, Mold)
    E = ev[["group", "mouse", "key", "t0"]].assign(
        auc_pre=a_new, peak_pre=p_new, mean_pre=mu_new,
        auc_median=a_old, peak_median=p_old, mean_median=mu_old, Dpre_over_Dmedian=ratio)
    E["day"] = E["mouse"].str.split("_").str[-1]
    E.to_csv(OUT / "tables" / "events.csv", index=False)

    D = (E.groupby(["group", "day"])
         .agg(n_events=("t0", "size"), auc_pre=("auc_pre", "mean"), peak_pre=("peak_pre", "mean"),
              mean_pre=("mean_pre", "mean"), auc_median=("auc_median", "mean"),
              peak_median=("peak_median", "mean"), Dpre_over_Dmedian=("Dpre_over_Dmedian", "mean"),
              frac_up=("mean_pre", lambda s: (s > 0).mean()))
         .reset_index())
    D.to_csv(OUT / "tables" / "per_day.csv", index=False)

    T = []
    for mtr in ("mean_pre", "auc_pre", "peak_pre", "auc_median", "peak_median"):
        a = D[D.group == G0][mtr].to_numpy()
        b = D[D.group == G1][mtr].to_numpy()
        r = C.group_test(a, b)
        T.append(dict(metric=mtr, **{f"{G0}_days": len(a), f"{G1}_days": len(b), f"{G0}_mean": a.mean(),
                                     f"{G1}_mean": b.mean(), f"{G0}_vs0_p": stats.ttest_1samp(a, 0).pvalue,
                                     f"{G1}_vs0_p": stats.ttest_1samp(b, 0).pvalue},
                      **{k: r[k] for k in ("hedges_g", "welch_p", "mw_p")}))
    T = pd.DataFrame(T)
    T.to_csv(OUT / "tables" / "group_tests.csv", index=False)

    # ------------------------------------------------------------------ figures
    m = (rg >= WIN[0]) & (rg <= WIN[1])
    t = rg[m]
    tr = {g: day_mean_traces(ev[ev.group == g], P[(ev.group == g).to_numpy()], rg, m) for g in GROUPS}
    ylim = []
    for g in GROUPS:
        for X in tr[g].values():
            mu = 100 * np.nanmean(X, 0); se = 100 * np.nanstd(X, 0) / np.sqrt(np.isfinite(X).sum(0))
            ylim += [np.nanmin(mu - se), np.nanmax(mu + se)]
    lo, hi = min(ylim), max(ylim)

    ncol = max(len(tr[G0]), len(tr[G1]))
    fig, axs = plt.subplots(2, ncol, figsize=((2.9 if WIN[1] - WIN[0] <= 30 else 4.2) * ncol, 6.4), sharey=True, sharex=True,
                            constrained_layout=True)
    for row, g in enumerate(GROUPS):
        for j in range(ncol):
            ax = axs[row, j]
            days = sorted(tr[g])
            if j >= len(days):
                ax.axis("off"); continue
            X = tr[g][days[j]]
            mu = 100 * np.nanmean(X, 0); se = 100 * np.nanstd(X, 0) / np.sqrt(np.isfinite(X).sum(0))
            ax.axvspan(*PRE_WIN, color="0.9", zorder=0); ax.axvspan(*C.RESP_WIN, color="#FFF3E0", zorder=0)
            ax.plot(t, mu, color=COL[g], lw=1.8)
            ax.fill_between(t, mu - se, mu + se, color=COL[g], alpha=0.25, lw=0)
            ax.axhline(0, color="0.5", lw=0.6); ax.axvline(0, color="k", lw=0.6)
            row_d = D[(D.group == g) & (D.day == days[j].split("_")[-1])].iloc[0]
            ax.set_title(f"{days[j].split('_')[-1]}  (n={len(X)})\nmean 0–15 s {100 * row_d.mean_pre:+.1f}%",
                         fontsize=9)
            ax.set_ylim(lo, hi)
            if j == 0:
                ax.set_ylabel(f"{LAB[g]}\n{Q} (%)")
            if row == 1:
                ax.set_xlabel("time from pupil onset (s)")
    fig.suptitle(f"Wake-up {SIGNAME} response per recording day, {Q} = ({VAR} − {VAR}_pre)/{VAR}_pre, {VAR}_pre = {BASE_LABEL}; "
                 "mean ± SEM across events (final labels 2026-10-09)", fontsize=10, x=0.01, ha="left")
    fig.savefig(OUT / "figures" / "A_per_day.png", dpi=150)
    plt.close(fig)

    # group mean of day means, and difference
    G = {g: np.array([100 * np.nanmean(X, 0) for X in tr[g].values()]) for g in GROUPS}
    fig, axs = plt.subplots(1, 2, figsize=(10 if WIN[1] - WIN[0] <= 30 else 15, 4.6), constrained_layout=True)
    ax = axs[0]
    for g in GROUPS:
        mu = G[g].mean(0); se = G[g].std(0, ddof=1) / np.sqrt(len(G[g]))
        ax.plot(t, mu, color=COL[g], lw=2, label=f"{LAB[g]} ({len(G[g])} days, "
                f"{sum(len(X) for X in tr[g].values())} events)")
        ax.fill_between(t, mu - se, mu + se, color=COL[g], alpha=0.22, lw=0)
    ax.axvspan(*PRE_WIN, color="0.9", zorder=0); ax.axvspan(*C.RESP_WIN, color="#FFF3E0", zorder=0)
    ax.axhline(0, color="0.5", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("time from pupil onset (s)"); ax.set_ylabel(f"{Q} (%)")
    ax.set_title("group mean ± SEM across days", fontsize=10); ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax = axs[1]
    a, b = G[G0], G[G1]
    diff = b.mean(0) - a.mean(0)
    se = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
    va, vb = a.var(0, ddof=1) / len(a), b.var(0, ddof=1) / len(b)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    ci = stats.t.ppf(0.975, dof) * se
    ax.plot(t, diff, color="k", lw=2)
    ax.fill_between(t, diff - ci, diff + ci, color="0.6", alpha=0.35, lw=0, label="95% CI (Welch, per time point)")
    ax.axvspan(*PRE_WIN, color="0.9", zorder=0); ax.axvspan(*C.RESP_WIN, color="#FFF3E0", zorder=0)
    ax.axhline(0, color="0.5", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    r = T[T.metric == "mean_pre"].iloc[0]
    ax.set_title(f"{LAB[G1]} − {LAB[G0]}   (mean 0–15 s: Welch p = {r.welch_p:.2f}, g = {r.hedges_g:+.2f})", fontsize=10)
    ax.set_xlabel("time from pupil onset (s)"); ax.set_ylabel(f"difference in {Q} (%)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"NREM → wake {SIGNAME}, {LAB[G0]} vs {LAB[G1]} (unit = recording day; {VAR}_pre = {BASE_LABEL})", fontsize=11, x=0.01, ha="left")
    fig.savefig(OUT / "figures" / "B_group.png", dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 220)
    print(D.round(4).to_string(index=False))
    print(T.round(4).to_string(index=False))
    print("D_pre / D_median per event: median %.3f, 5-95%% %.3f-%.3f" %
          tuple(np.nanpercentile(ratio, [50, 5, 95])))
    print("->", OUT)


if __name__ == "__main__":
    main()
