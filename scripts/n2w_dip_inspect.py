"""Side-by-side look at NREM->wake arteriole responses on two WT days:
a 2023 RSC day where the arteriole dilates vs GCaMP8m 2026.03.03 where it
mostly goes down. Question: do the 03.03 "dips" look like real constrictions
or like artifacts?

Viewing aid only, no statistics. Same events as n2w_arousal_explore.py
(response_events, N2W, manual labels).

Outputs: data/RESULTS_wake_arousal_explore_2026-10-02/WT/inspect_<A>_vs_<B>/
    events_page_XX.png   each row = one event per day, -60..+60 s, arteriole NOT
                         baseline-subtracted (% of cycle median) so drifts show
    cycles_<day>.png     whole cycles: arteriole + sleep state + event marks
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

sys.path.insert(0, str(Path(__file__).resolve().parent / "figs_prelim"))
import common as C  # noqa: E402

OUT = C.ROOT / "data" / "RESULTS_wake_arousal_explore_2026-10-02" / "WT"
WIN = (-60.0, 60.0)
ROWS = 8


def day_events(day):
    ev = C.response_events([k for k in C.keys_for(["WT"]) if C.parse_key(k).mouse == day])
    return ev[ev["type"] == "N2W"].sort_values(["key", "t0"]).reset_index(drop=True)


def seg(cyc, s, t0, smooth_s=0.0):
    if s not in cyc.sig:
        return None, None
    t, y = cyc.sig[s]
    m = (t >= t0 + WIN[0]) & (t <= t0 + WIN[1])
    yy = y[m].copy()
    if smooth_s:
        n = max(1, int(round(smooth_s * cyc.fs[s])))
        yy = pd.Series(yy).rolling(n, center=True, min_periods=1).mean().to_numpy()
    return t[m] - t0, yy


def art_auc(cyc, t0):
    t, y = cyc.sig["art"]
    b = (t >= t0 - 5) & (t < t0)
    r = (t >= t0) & (t <= t0 + 15)
    return np.trapz(y[r] - np.nanmean(y[b]), t[r]) if b.any() and r.any() else np.nan


def draw_event(axs, r):
    cyc = C.load_cycle(r.key)
    auc = art_auc(cyc, r.t0)
    for ax in axs:
        ax.axvspan(0, 15, color="#FFF3E0", zorder=0)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlim(*WIN)
        ax.tick_params(labelsize=6)
    # state strip on the pupil axis
    lab = cyc.labels
    for b, s in enumerate(lab):
        tb = b * C.BIN - r.t0
        if WIN[0] - 5 <= tb <= WIN[1] and s in C.STATE_COL:
            axs[0].axvspan(tb, tb + C.BIN, ymin=0.92, ymax=1, color=C.STATE_COL[s], lw=0)
    t, y = seg(cyc, "pupil", r.t0)
    axs[0].plot(t, y, color="#5D4037", lw=0.9)
    t, y = seg(cyc, "eye", r.t0)
    if t is not None:
        ax2 = axs[0].twinx()
        ax2.plot(t, y, color="#00897B", lw=0.6, alpha=0.8)
        ax2.set_yticks([])
    t, y = seg(cyc, "ca", r.t0)
    if t is not None:
        axs[1].plot(t, y, color="#2E7D32", lw=0.9)
    t, y = seg(cyc, "art", r.t0)
    axs[2].plot(t, 100 * y, color="#EF9A9A", lw=0.6)
    t, y = seg(cyc, "art", r.t0, smooth_s=2.0)
    axs[2].plot(t, 100 * y, color="#C62828", lw=1.2)
    axs[2].axhline(0, color="0.7", lw=0.5)
    axs[0].set_title(f"{r.key.rsplit('_', 1)[1]}  t={r.t0:.0f}s   art AUC {auc:+.2f}",
                     fontsize=7, loc="left", color="#C62828" if auc < 0 else "#1565C0")
    axs[0].set_ylabel("pupil", fontsize=6)
    axs[1].set_ylabel("Ca²⁺", fontsize=6)
    axs[2].set_ylabel("art %", fontsize=6)


def event_pages(days, evs, out):
    n = max(len(e) for e in evs)
    for p0 in range(0, n, ROWS):
        fig = plt.figure(figsize=(16, 2.1 * ROWS))
        outer = fig.add_gridspec(ROWS, 2, hspace=0.45, wspace=0.12)
        for c, (day, ev) in enumerate(zip(days, evs)):
            for j in range(ROWS):
                i = p0 + j
                if i >= len(ev):
                    continue
                inner = outer[j, c].subgridspec(1, 3, wspace=0.25)
                draw_event([fig.add_subplot(inner[k]) for k in range(3)], ev.iloc[i])
            fig.text(0.13 + 0.42 * c, 0.995, f"{day.split('_', 1)[1]}  ({len(ev)} events)",
                     fontsize=11, weight="bold", va="top")
        fig.text(0.01, 0.002, "pupil (brown) / eye movement (teal) with state strip (blue awake, red NREM); "
                 "Ca²⁺ green; arteriole % of cycle median, raw (pink) and 2-s smoothed (red), "
                 "NOT baseline-subtracted. Orange = 0-15 s window. Title colour: blue AUC>0, red AUC<0.",
                 fontsize=8)
        fig.savefig(out / f"events_page_{p0 // ROWS + 1:02d}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


def cycle_page(day, ev, out):
    keys = sorted({k for k in C.keys_for(["WT"]) if C.parse_key(k).mouse == day},
                  key=lambda k: C.parse_key(k).cycle)
    fig, axs = plt.subplots(len(keys), 1, figsize=(16, 2.2 * len(keys)), squeeze=False,
                            constrained_layout=True)
    for ax, k in zip(axs[:, 0], keys):
        cyc = C.load_cycle(k)
        for b, s in enumerate(cyc.labels):
            if s in C.STATE_COL:
                ax.axvspan(b * C.BIN, (b + 1) * C.BIN, color=C.STATE_COL[s], alpha=0.12, lw=0)
        t, y = cyc.sig["art"]
        ax.plot(t, 100 * y, color="#EF9A9A", lw=0.5)
        ys = pd.Series(y).rolling(4, center=True, min_periods=1).mean().to_numpy()
        ax.plot(t, 100 * ys, color="#C62828", lw=1)
        if "ca" in cyc.sig:
            tc, yc = cyc.sig["ca"]
            a2 = ax.twinx()
            a2.plot(tc, yc, color="#2E7D32", lw=0.6, alpha=0.6)
            a2.set_yticks([])
        for r in ev[ev["key"] == k].itertuples():
            auc = art_auc(cyc, r.t0)
            ax.axvline(r.t0, color="#C62828" if auc < 0 else "#1565C0", lw=1.5, ls="--")
        ax.set_ylabel(f"{k.rsplit('_', 1)[1]}\nart %", fontsize=8)
        ax.set_xlim(0, cyc.duration)
    axs[-1, 0].set_xlabel("time in cycle (s)")
    fig.suptitle(f"{day.split('_', 1)[1]}: whole cycles. Arteriole (red, % of median), Ca²⁺ (green), "
                 "state shading (blue awake, red NREM); dashed = wake-up event "
                 "(blue dilates, red goes down)", fontsize=10, x=0.01, ha="left")
    fig.savefig(out / f"cycles_{day.split('_', 1)[1]}.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="Extra_RestrosplenialCortex_2023.05.07")
    ap.add_argument("--b", default="Extra_GCaMP8m_2026.03.03")
    a = ap.parse_args()
    C.set_label_mode("manual")
    days = [a.a, a.b]
    out = OUT / f"inspect_{a.a.split('_', 1)[1]}_vs_{a.b.split('_', 1)[1]}"
    out.mkdir(parents=True, exist_ok=True)
    evs = [day_events(d) for d in days]
    event_pages(days, evs, out)
    for d, e in zip(days, evs):
        cycle_page(d, e, out)
    print("->", out, [len(e) for e in evs])


if __name__ == "__main__":
    main()
