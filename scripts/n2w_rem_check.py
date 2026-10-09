"""Does REM explain the wake-up 'constrictions'? Old vs new labels, WT.

Florence's explanation (2026-10-07): REM dilates the vessel, so waking up
from (or shortly after) REM looks like a constriction. Her 2026-10-07/08
re-label added REM on the GCaMP8m days. This compares, per WT day:
  * wake-up (N2W) events under the 2026-10-01 labels vs the 2026-10-08 labels:
    which events survive, which drop out, and their arteriole level change
  * under the new labels: events with REM in the 180 s before onset vs without

Level change as in w2n_by_day.py: mean(+10..+40 s) - mean(-30..-10 s), % of
cycle median. Events: pupil-anchored, non-overlapping (same rules as figures).

Usage: ./venv/bin/python -B scripts/n2w_rem_check.py
Output: data/RESULTS_wake_arousal_explore_2026-10-02/WT/rem_check_2026-10-08/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "figs_prelim"))
import common as C  # noqa: E402

OLD = C.ROOT / "data" / "labels_snapshot_2026-10-01"
import os  # noqa: E402
NEW = C.ROOT / "data" / os.environ.get("REM_CHECK_NEW", "labels_bucket_2026-10-08_1456")
_TAG = {"labels_bucket_2026-10-08_1456": "2026-10-08"}.get(NEW.name, NEW.name.replace("labels_bucket_", ""))
OUT = C.ROOT / "data" / "RESULTS_wake_arousal_explore_2026-10-02" / "WT" / f"rem_check_{_TAG}"
PRE, POST = (-30.0, -10.0), (10.0, 40.0)
REM_LOOKBACK = 180.0


def n2w_events(label_dir):
    frames = []
    for k in C.keys_for(["WT"], labeled_only=False):
        lab = C._read_label_file(label_dir / f"{k}_labels.csv")
        if not lab.size:
            continue
        e = C.events(lab)
        if e.empty:
            continue
        info = C.parse_key(k)
        frames.append(e.assign(key=k, mouse=info.mouse, group=info.group))
    ev = pd.concat(frames, ignore_index=True)
    ev = C.drop_overlapping(C.anchor_to_pupil(ev))
    return ev[ev["type"] == "N2W"].reset_index(drop=True)


def level_change(ev):
    rg, A = C.epochs_for(ev, "art", baseline=False)
    A = 100 * A
    pre, post = (rg >= PRE[0]) & (rg < PRE[1]), (rg > POST[0]) & (rg <= POST[1])
    ok = (np.isfinite(A[:, pre]).sum(1) > 10) & (np.isfinite(A[:, post]).sum(1) > 10)
    return np.where(ok, np.nanmean(A[:, post], 1) - np.nanmean(A[:, pre], 1), np.nan)


def rem_before(ev, label_dir):
    """Seconds from the end of the last REM bin to the label onset (inf if none)."""
    out = []
    for r in ev.itertuples():
        lab = C._read_label_file(label_dir / f"{r.key}_labels.csv")
        idx = [i for i, v in enumerate(lab) if v == "REM" and (i + 1) * C.BIN <= r.t0_label]
        out.append(r.t0_label - (idx[-1] + 1) * C.BIN if idx else np.inf)
    return np.array(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    o, n = n2w_events(OLD), n2w_events(NEW)
    o["level_change"], n["level_change"] = level_change(o), level_change(n)
    n["s_since_REM"] = rem_before(n, NEW)
    o["s_since_REM_newlabels"] = rem_before(o.assign(t0_label=o["t0_label"]), NEW)
    # match events across label sets by cycle + label onset within 5 s
    o["in_new"] = [((n["key"] == r.key) & (abs(n["t0_label"] - r.t0_label) <= 5)).any() for r in o.itertuples()]
    for d in (o, n):
        d["day"] = d["mouse"].str.split("_", n=1).str[1]
    o.to_csv(OUT / "events_old_labels.csv", index=False)
    n.to_csv(OUT / "events_new_labels.csv", index=False)

    rows = []
    for day in sorted(o["day"].unique()):
        a, b = o[o.day == day], n[n.day == day]
        kept, gone = a[a.in_new], a[~a.in_new]
        rr, nr = b[b.s_since_REM <= REM_LOOKBACK], b[b.s_since_REM > REM_LOOKBACK]
        rows.append(dict(
            day=day, old_n=len(a), old_median=a.level_change.median(), old_frac_down=(a.level_change < 0).mean(),
            new_n=len(b), new_median=b.level_change.median(), new_frac_down=(b.level_change < 0).mean(),
            dropped_n=len(gone), dropped_median=gone.level_change.median(),
            dropped_with_REM_near=int((gone.s_since_REM_newlabels <= REM_LOOKBACK).sum()),
            new_REMnear_n=len(rr), new_REMnear_median=rr.level_change.median(),
            new_noREM_n=len(nr), new_noREM_median=nr.level_change.median(),
            new_noREM_frac_down=(nr.level_change < 0).mean() if len(nr) else np.nan))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "per_day.csv", index=False)

    # figure: per-day medians, old vs new vs new-without-REM
    fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    x = np.arange(len(t))
    for j, (col, ncol, lab, c) in enumerate([("old_median", "old_n", "old labels (10-01)", "0.6"),
                                             ("new_median", "new_n", f"new labels ({_TAG})", "#1E88E5"),
                                             ("new_noREM_median", "new_noREM_n",
                                              f"new labels, no REM in prior {REM_LOOKBACK:.0f} s", "#0D47A1")]):
        ax.bar(x + (j - 1) * 0.27, t[col], 0.27, color=c, label=lab)
        for xi, v, k in zip(x, t[col], t[ncol]):
            if np.isfinite(v):
                ax.text(xi + (j - 1) * 0.27, v + (0.15 if v >= 0 else -0.15), str(k), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=7)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x, [d.replace("RestrosplenialCortex_", "RSC ").replace("GCaMP8m_", "8m ") for d in t.day],
                  rotation=20)
    ax.set_ylabel("median arteriole level change\nat wake-up (% of median)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Wake-up arteriole change per day: does removing REM-related events remove the constriction? "
                 "(numbers = events)", fontsize=10, loc="left")
    fig.savefig(OUT / "per_day_old_vs_new.png", dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))
    print("->", OUT)


if __name__ == "__main__":
    main()
