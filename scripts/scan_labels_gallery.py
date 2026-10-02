#!/usr/bin/env python3
"""Review gallery for the label scan: one small panel per flagged short bout.

Each panel shows pupil z and whisker z for 60 s around the bout, the manual
labels as a ribbon, the flagged bout outlined, and the ML second opinion.
Read-only with respect to labels. Run after scripts/scan_labels.py.

    ./venv/bin/python -B scripts/scan_labels_gallery.py [verdict]   # default: LIKELY MISLABELLED
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figs_prelim import common as C  # noqa: E402

SCAN = ROOT / "data" / "RESULTS_label_scan_2026-10-01"
PER_PAGE = 24
PAD = 30.0


def panel(ax, r):
    cyc = C.load_cycle(r.key)
    lab = C.load_labels(r.key, "manual")
    t0, t1 = r.t_start_s - PAD, r.t_start_s + r.dur_s + PAD
    for b in C.bouts(lab).itertuples():
        if b.state in C.STATE_COL and b.t_start + b.dur_s > t0 and b.t_start < t1:
            ax.axvspan(max(b.t_start, t0), min(b.t_start + b.dur_s, t1), ymin=0, ymax=0.12,
                       color=C.STATE_COL[b.state], lw=0)
    ax.axvspan(r.t_start_s, r.t_start_s + r.dur_s, color="k", alpha=0.08, lw=0)
    ax.axvline(r.t_start_s, color="k", lw=.5); ax.axvline(r.t_start_s + r.dur_s, color="k", lw=.5)
    for s, col in (("pupil", C.SIG_COL["pupil"]), ("whisk", C.SIG_COL["whisk"])):
        if s in cyc.sig:
            t, y = cyc.sig[s]
            m = (t >= t0) & (t <= t1)
            yy = np.clip(y[m], -4, 6) if s == "whisk" else y[m]
            ax.plot(t[m], yy, color=col, lw=.8 if s == "pupil" else .5, alpha=1 if s == "pupil" else .7)
    for lvl, st in ((r.ref_awake_pupil, "Awake"), (r.ref_nrem_pupil, "NREM")):
        if np.isfinite(lvl):
            ax.axhline(lvl, color=C.STATE_COL[st], ls=":", lw=.8)
    ax.set_xlim(t0, t1)
    ax.set_title(f"{r.key.replace('IP3R2_', '').replace('Extra_', '')}  t={r.t_start_s:.0f}s\n"
                 f"labelled {r.label} {r.dur_s:.0f}s | P(NREM)ml={r.ml_p_nrem:.2f}", fontsize=6.5)
    ax.tick_params(labelsize=5)


def main():
    verdict = sys.argv[1] if len(sys.argv) > 1 else "LIKELY MISLABELLED"
    R = pd.read_csv(SCAN / "2_short_bouts.csv")
    R = R[R.verdict == verdict].sort_values(["group", "key", "t_start_s"]).reset_index(drop=True)
    tag = verdict.lower().replace(" ", "_")
    out = SCAN / f"gallery_{tag}"
    out.mkdir(exist_ok=True)
    pages = int(np.ceil(len(R) / PER_PAGE))
    for p in range(pages):
        sub = R.iloc[p * PER_PAGE:(p + 1) * PER_PAGE]
        fig, axes = plt.subplots(4, 6, figsize=(18, 10.5))
        for ax, r in zip(axes.flat, sub.itertuples()):
            panel(ax, r)
        for ax in axes.flat[len(sub):]:
            ax.axis("off")
        fig.suptitle(f"Label scan — {verdict}: page {p + 1}/{pages} ({len(R)} bouts).  Brown = pupil z, "
                     f"green = whisker z (clipped), ribbon = manual label, grey box = flagged bout, "
                     f"dotted = this cycle's typical Awake (blue) / NREM (red) pupil.  Nothing has been relabelled.",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(out / f"page_{p + 1:02d}.png", dpi=130)
        plt.close(fig)
    print(f"{len(R)} bouts -> {pages} pages in {out}")


if __name__ == "__main__":
    main()
