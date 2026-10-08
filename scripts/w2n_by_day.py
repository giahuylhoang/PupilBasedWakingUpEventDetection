"""Does the arteriole mirror itself at falling asleep? Per-day check.

Prediction from the NREM-vs-wake arteriole level on each WT day
(n2w_dip_inspect.py follow-up): days whose vessel is wider in NREM
(GCaMP8m 2025.12.18, 2026.03.03) should DILATE at wake->NREM; the 2023 RSC
days, narrower in NREM, should CONSTRICT. Waking up should show the reverse.

Events: common.response_events (non-overlapping), manual labels.
  W2N = Awake >=30 s then NREM >=30 s, t0 = label boundary (not pupil-anchored:
        falling asleep has no sharp pupil onset).
  N2W = as in the figures, t0 = pupil dilation onset.
Because falling asleep is gradual, the main measure is a LEVEL CHANGE, not
the 0-15 s AUC:  level_change = mean(+10..+40 s) - mean(-30..-10 s),
in % of the cycle median. -10..+10 s is skipped as the transition zone.

Usage: ./venv/bin/python -B scripts/w2n_by_day.py [--group WT]
Outputs: data/RESULTS_wake_arousal_explore_2026-10-02/<group>/w2n_vs_n2w/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent / "figs_prelim"))
import common as C  # noqa: E402

PRE, POST = (-30.0, -10.0), (10.0, 40.0)
VIEW = (-30.0, 60.0)
TYPE_COL = {"W2N": "#6A1B9A", "N2W": "#EF6C00"}
TYPE_NAME = {"W2N": "falling asleep (wake→NREM)", "N2W": "waking up (NREM→wake)"}


def state_level_diff(keys):
    """Per day: mean arteriole in NREM minus Awake, % of cycle median."""
    rows = []
    for k in keys:
        cyc = C.load_cycle(k)
        t, y = cyc.sig["art"]
        b = np.minimum((t // C.BIN).astype(int), len(cyc.labels) - 1)
        st = cyc.labels[b] if len(cyc.labels) else np.array([None] * len(t))
        for s in ("Awake", "NREM"):
            m = (st == s) & np.isfinite(y)
            if m.sum() > 60:
                rows.append(dict(mouse=cyc.info.mouse, state=s, level=100 * y[m].mean(), n=m.sum()))
    d = pd.DataFrame(rows)
    w = d.groupby(["mouse", "state"]).apply(lambda x: np.average(x.level, weights=x.n)).unstack()
    return (w["NREM"] - w["Awake"]).rename("nrem_minus_wake")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="WT")
    a = ap.parse_args()
    C.set_label_mode("manual")
    sub = "w2n_vs_n2w" if C.LABEL_SET == "original" else f"w2n_vs_n2w__{C.LABEL_SET}"
    out = C.ROOT / "data" / "RESULTS_wake_arousal_explore_2026-10-02" / a.group / sub
    out.mkdir(parents=True, exist_ok=True)

    keys = C.keys_for([a.group])
    ev = C.response_events(keys)
    ev = ev[ev["type"].isin(["W2N", "N2W"])].reset_index(drop=True)

    rg, A = C.epochs_for(ev, "art", baseline=False)
    A = 100 * A
    _, P = C.epochs_for(ev, "pupil", baseline=False)
    rgp = C.rel_grid(C.EPOCH_FS["pupil"])
    pre, post = (rg >= PRE[0]) & (rg < PRE[1]), (rg > POST[0]) & (rg <= POST[1])
    ok = (np.isfinite(A[:, pre]).sum(1) > 10) & (np.isfinite(A[:, post]).sum(1) > 10)
    ev["level_change"] = np.where(ok, np.nanmean(A[:, post], 1) - np.nanmean(A[:, pre], 1), np.nan)
    pp, po = (rgp >= PRE[0]) & (rgp < PRE[1]), (rgp > POST[0]) & (rgp <= POST[1])
    ev["pupil_change"] = np.nanmean(P[:, po], 1) - np.nanmean(P[:, pp], 1)
    ev["day"] = ev["mouse"].str.split("_", n=1).str[1]
    ev[["key", "day", "type", "t0", "level_change", "pupil_change"]].to_csv(out / "events.csv", index=False)

    lvl = state_level_diff(keys)
    tab = (ev.groupby(["mouse", "type"])["level_change"]
           .agg(n="count", median="median", frac_up=lambda s: (s > 0).mean()).unstack("type"))
    tab.columns = [f"{t}_{m}" for m, t in tab.columns]
    tab = tab.join(lvl)
    tab.index = tab.index.str.split("_", n=1).str[1]
    tab = tab[["nrem_minus_wake", "W2N_n", "W2N_median", "W2N_frac_up", "N2W_n", "N2W_median", "N2W_frac_up"]]
    tab.to_csv(out / "per_day.csv")

    # ---------------- figure: per-day averages, both transitions ----------------
    days = list(lvl.sort_values(ascending=False).index)
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True, constrained_layout=True)
    m = (rg >= VIEW[0]) & (rg <= VIEW[1])
    for ax, d in zip(axs.flat, days):
        for t in ("W2N", "N2W"):
            sel = ((ev["mouse"] == d) & (ev["type"] == t)).to_numpy()
            X = A[sel][:, m]
            X = X - np.nanmean(A[sel][:, pre], 1, keepdims=True)      # each event relative to its pre-window
            X = X[np.isfinite(X).any(1)]
            if len(X) < 2:
                continue
            mu = np.nanmean(X, 0)
            se = np.nanstd(X, 0) / np.sqrt(np.isfinite(X).sum(0))
            mu_s = pd.Series(mu).rolling(4, center=True, min_periods=1).mean()
            ax.plot(rg[m], mu_s, color=TYPE_COL[t], lw=1.8, label=f"{t} (n={len(X)})")
            ax.fill_between(rg[m], mu_s - se, mu_s + se, color=TYPE_COL[t], alpha=0.18, lw=0)
        ax.axvspan(*PRE, color="0.9", zorder=0)
        ax.axvspan(*POST, color="#FFF3E0", zorder=0)
        ax.axvline(0, color="k", lw=0.6)
        ax.axhline(0, color="0.6", lw=0.6)
        r = tab.loc[d.split("_", 1)[1]]
        ax.set_title(f"{d.split('_', 1)[1]}\nNREM − wake level {r['nrem_minus_wake']:+.1f}%",
                     fontsize=9, color="#C62828" if r["nrem_minus_wake"] > 0 else "#1565C0")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    for ax in axs[-1]:
        ax.set_xlabel("time from transition (s)")
    for ax in axs[:, 0]:
        ax.set_ylabel("arteriole change (% of median)")
    fig.suptitle("Arteriole at falling asleep (purple) vs waking up (orange), per day, ordered by how much "
                 "wider the vessel is in NREM (red title = wider in NREM). Mean ± SEM across events, "
                 "each relative to −30…−10 s; grey/orange = windows for level change",
                 fontsize=10, x=0.01, ha="left")
    fig.savefig(out / "per_day_averages.png", dpi=140)
    plt.close(fig)

    # ---------------- figure: day-level scatter ----------------
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, t in zip(axs, ("W2N", "N2W")):
        x, y = tab["nrem_minus_wake"], tab[f"{t}_median"]
        ax.scatter(x, y, s=50, color=TYPE_COL[t])
        for d in tab.index:
            ax.annotate(d.replace("RestrosplenialCortex_", "RSC ").replace("GCaMP8m_", "8m "),
                        (x[d], y[d]), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.axhline(0, color="0.6", lw=0.6); ax.axvline(0, color="0.6", lw=0.6)
        rho = spearmanr(x, y, nan_policy="omit")[0]
        ax.set_title(f"{TYPE_NAME[t]}: ρ = {rho:+.2f} across {y.notna().sum()} days", fontsize=10)
        ax.set_xlabel("vessel level, NREM − wake (% of median)")
        ax.set_ylabel("median level change at transition (%)")
    fig.savefig(out / "day_scatter.png", dpi=140)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print(tab.round(2).to_string())
    print("\npupil change sanity (median): ", ev.groupby("type")["pupil_change"].median().round(2).to_dict())
    for t in ("W2N", "N2W"):
        rho = spearmanr(tab["nrem_minus_wake"], tab[f"{t}_median"], nan_policy="omit")[0]
        print(f"{t}: day-level rho(NREM-wake level, median change) = {rho:+.2f}")
    print("->", out)


if __name__ == "__main__":
    main()
