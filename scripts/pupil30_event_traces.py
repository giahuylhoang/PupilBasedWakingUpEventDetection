"""Per-event pupil traces for the 30% criterion (2026-10-21, for Florence to compare with
her manually selected SFN 2023 events).

Same events and definitions as scripts/pupil30_summary.py. Writes:
  figures/pupil_traces_by_day.png   one panel per recording day: every event's pupil
                                    % change (dark = passes >= 30%, light = fails) and
                                    the day mean of passing events
  pupil_traces_per_event.pdf        one page per day, one small panel per event, titled
                                    with cycle and onset time in the cycle so events can
                                    be matched to her own list
  event_list.csv                    day, cycle, onset time (s into the cycle), type,
                                    pupil rise %, pass, Ca2+ and arteriole 0-15 s means

Usage:
    PRELIM_LABEL_SET=relabel_2026-10-09 venv/bin/python -B scripts/pupil30_event_traces.py
Output: data/RESULTS_pupil30_summary_2026-10-19/ (new files only)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pupil30_summary as P  # noqa: E402

TYPE_LAB = {"N2W": "NREM → wake", "R2W": "REM → wake"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def main():
    P.FIG.mkdir(parents=True, exist_ok=True)
    ev, mats = P.load()
    pup = mats["pupil"]
    ev = ev.sort_values(["cohort", "day", "type", "cycle", "t0"], ascending=[False, True, True, True, True])
    lst = ev[["cohort", "day", "cycle", "type", "t0", "pupil_rise_pct", "passes", "ca_mean", "art_mean"]].copy()
    lst = lst.rename(columns={"t0": "onset_s_in_cycle", "ca_mean": "ca_mean_0_15_pct", "art_mean": "art_mean_0_15_pct"})
    lst.round(2).to_csv(P.OUT / "event_list.csv", index=False)

    groups = list(ev.groupby(["cohort", "day", "type"], sort=False))
    nc = 4
    nr = int(np.ceil(len(groups) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.1 * nr), sharex=True, constrained_layout=True)
    for ax in axs.flat[len(groups):]:
        ax.axis("off")
    for ax, ((coh, day, typ), g) in zip(axs.flat, groups):
        col = dict(P.GROUPS_COL)[(coh, typ)]
        for i, r in zip(g.index, g.itertuples()):
            ax.plot(P.RG, pup[i], color=col if r.passes else "0.75", lw=0.8, alpha=0.8 if r.passes else 0.9,
                    zorder=2 if r.passes else 1)
        ok = g.index[g.passes]
        if len(ok):
            with np.errstate(all="ignore"):
                ax.plot(P.RG, np.nanmean(pup[ok], 0), color="k", lw=2, zorder=3)
        ax.axvline(0, color="k", lw=0.6); ax.axhline(0, color="0.7", lw=0.6)
        ax.axhline(P.THR, color="#C62828", lw=0.8, ls="--")
        ax.set_title(f"{'RSC 2023' if coh == 'RSC2023' else 'GCaMP8m'} {day}, {TYPE_LAB[typ]}\n"
                     f"{len(ok)} of {len(g)} pass", fontsize=9.5, color=col)
        ax.set_ylabel("pupil (% change)", fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("time from pupil onset (s)")
    fig.suptitle(f"Every wake-up's pupil trace. Coloured = passes ≥ {P.THR:.0f}% (red dashed line), grey = fails, "
                 "black = mean of passing. y-axis differs per panel.", fontsize=11, x=0.01, ha="left")
    fig.savefig(P.FIG / "pupil_traces_by_day.png", dpi=150)
    plt.close(fig)

    with PdfPages(P.OUT / "pupil_traces_per_event.pdf") as pdf:
        for (coh, day, typ), g in groups:
            n = len(g)
            c = 6
            r_ = int(np.ceil(n / c))
            fig, axs = plt.subplots(r_, c, figsize=(16, 2.2 * r_ + 0.8), squeeze=False, sharex=True,
                                    constrained_layout=True)
            for ax in axs.flat[n:]:
                ax.axis("off")
            col = dict(P.GROUPS_COL)[(coh, typ)]
            for ax, (i, r) in zip(axs.flat, zip(g.index, g.itertuples())):
                ax.plot(P.RG, pup[i], color=col if r.passes else "0.6", lw=1.2)
                ax.axvline(0, color="k", lw=0.6); ax.axhline(0, color="0.8", lw=0.6)
                ax.axhline(P.THR, color="#C62828", lw=0.7, ls="--")
                ax.set_title(f"cycle {r.cycle}, {r.t0:.1f} s  |  +{r.pupil_rise_pct:.0f}%"
                             f"{'' if r.passes else '  (fails)'}", fontsize=8.5)
                ax.tick_params(labelsize=7)
            fig.suptitle(f"{'RSC 2023' if coh == 'RSC2023' else 'GCaMP8m'} {day}, {TYPE_LAB[typ]}: pupil % change "
                         f"vs the 20 s before onset. Time in title = pupil onset, seconds into the cycle.",
                         fontsize=11, x=0.01, ha="left")
            pdf.savefig(fig)
            plt.close(fig)
    print(len(ev), "events,", len(groups), "day/type groups ->", P.OUT)


if __name__ == "__main__":
    main()
