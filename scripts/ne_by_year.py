"""GRAB-NE split by recording year (Florence, 2026-10-23): 2025 days (09.09, 09.11)
vs 2026 days (03.10, and 05.08 / 05.10 once they are labelled), like RSC 2023 vs GCaMP8m.

Reuses scripts/ne_labelled_summary.py (final manual labels, same event and
normalisation definitions). Days without labels are skipped automatically, so
re-running after 05.08 / 05.10 are labelled adds them to the 2026 group.

  1_event_curves_by_year   NE / pupil / arteriole around NREM->wake, micro-arousal,
                           REM->wake, wake->NREM; one line per group, mean +- SEM
                           across days (no SEM when a group has 1 day)
  2_state_levels_by_year   NE / pupil / arteriole level in Awake / NREM / REM per day

Usage:
    PRELIM_LABEL_SET=relabel_2026-10-09 venv/bin/python -B scripts/ne_by_year.py
Output: data/RESULTS_ne_by_year_2026-10-23/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "figs_prelim"))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import ne_labelled_summary as N  # noqa: E402

OUT = C.ROOT / "data" / "RESULTS_ne_by_year_2026-10-23"
FIG = OUT / "figures"
YR_COL = {"2025": "#424242", "2026": "#2E7D32"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def labelled_keys():
    out = []
    for k in N.ne_keys():
        L = np.asarray(C.load_cycle(k).labels, dtype=object)
        if len(L) and any(x is not None for x in L):
            out.append(k)
    return out


def main():
    C.set_label_mode("manual")
    FIG.mkdir(parents=True, exist_ok=True)
    keys = labelled_keys()
    ev = N.load_events(keys)
    ev["year"] = ev.day.str[:4]
    mats = {s: N.epochs(ev, s) for s, _ in N.SIGS}
    days = {y: sorted(ev[ev.year == y].day.unique()) for y in YR_COL}

    fig, axs = plt.subplots(3, 4, figsize=(15, 8.5), sharex=True, constrained_layout=True)
    rows = []
    for j, (typ, tl) in enumerate(N.EV_TYPES):
        for i, (s, ylab) in enumerate(N.SIGS):
            ax = axs[i, j]
            for yr, col in YR_COL.items():
                idx = ev.index[(ev.type == typ) & (ev.year == yr)]
                if not len(idx):
                    continue
                mu, se, nd = N.day_mean(ev, mats[s], idx)
                ax.plot(RG := N.RG, mu, color=col, lw=2,
                        label=f"{yr}: {len(idx)} ev, {nd} day{'s' if nd > 1 else ''}")
                ax.fill_between(RG, mu - se, mu + se, color=col, alpha=0.2, lw=0)
                m = (RG >= 0) & (RG <= 15)
                rows.append(dict(type=typ, signal=s, year=yr, events=len(idx), days=nd,
                                 mean_0_15=float(np.nanmean(mu[m]))))
            ax.axvline(0, color="k", lw=0.8); ax.axhline(0, color="0.7", lw=0.8)
            if i == 0:
                ax.set_title(tl, fontsize=11, weight="bold")
                ax.legend(frameon=False, fontsize=8, loc="upper left")
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 2:
                ax.set_xlabel("time from onset (s)")
    for i in range(3):
        lo = min(a.get_ylim()[0] for a in axs[i]); hi = max(a.get_ylim()[1] for a in axs[i])
        for a in axs[i]:
            a.set_ylim(lo, hi)
    fig.suptitle(f"GRAB-NE by year. 2025 = {', '.join(days['2025'])}; 2026 = {', '.join(days['2026'])}. "
                 "Mean ± SEM across days.", fontsize=11, x=0.01, ha="left")
    fig.savefig(FIG / "1_event_curves_by_year.png", dpi=160)
    plt.close(fig)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "event_curves_by_year.csv", index=False)

    B = N.bin_table(keys)
    lev = B[B.state.isin(N.STATES)].groupby(["day", "state"])[["ne", "pupil", "art"]].median().reset_index()
    lev[["ne", "art"]] *= 100
    lev["year"] = lev.day.str[:4]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (s, lab) in zip(axs, (("ne", "NE vs cycle median (%)"), ("pupil", "pupil (robust z)"),
                                  ("art", "arteriole vs cycle median (%)"))):
        for day, g in lev.groupby("day"):
            g = g.set_index("state").reindex(N.STATES)
            col = YR_COL[day[:4]]
            ax.plot(range(3), g[s], "-o", color=col, lw=1.4, ms=6)
            last = 2 if np.isfinite(g[s].iloc[2]) else 1
            ax.text(last + 0.08, g[s].iloc[last], day, fontsize=7.5, va="center", color=col)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_xticks(range(3), N.STATES); ax.set_xlim(-0.3, 2.9); ax.set_ylabel(lab)
    fig.suptitle("Level in each state, per day (grey = 2025, green = 2026)", fontsize=11, x=0.01, ha="left")
    fig.savefig(FIG / "2_state_levels_by_year.png", dpi=160)
    plt.close(fig)
    lev.to_csv(OUT / "state_levels_by_year.csv", index=False)

    pd.set_option("display.width", 200)
    print("labelled cycles:", len(keys), days)
    print(S.pivot_table(index=["type", "signal"], columns="year", values="mean_0_15").round(2))
    print(lev.round(2).to_string(index=False))
    print("->", OUT)


if __name__ == "__main__":
    main()
