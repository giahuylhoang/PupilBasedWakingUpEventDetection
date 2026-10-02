#!/usr/bin/env python3
"""Propose corrections for ONE agreed pattern of labelling error. Originals untouched.

Pattern 2 (agreed with Huy, 2026-10-01): a short NREM bout inside wake
(Awake - NREM <= 25 s - Awake) whose pupil stays in the upper half the whole
time: >= 90% of its pupil samples (10 Hz, 1-s smoothed) lie ABOVE the midpoint
between this cycle's typical Awake and typical NREM pupil. Proposed label: Awake.
(The first draft used the bout median, which let through bouts where the pupil
clearly dipped below the midpoint; that is a real dip, not pattern 2.)

Explicitly NOT changed:
  * Pattern 1 -- short wake inside NREM (micro-arousal candidates), whatever the
    pupil does. Partial dilations are kept as Florence labelled them.
  * Anything else: REM, Unclear, long bouts.
  * Cycles whose pupil cannot be trusted (DEGRADED_PUPIL_DAYS): listed for
    video review instead of corrected.

Typical pupil per cycle = median of 5-s bin means over that cycle's stable bouts
(>= 30 s) of each state; within-cycle robust z, despiked (common.load_cycle).

Outputs (new folders; the bucket and data/labels_snapshot_2026-10-01/ are never written):
  data/labels_corrected_2026-10-01/<key>_labels.csv
      same schema as the manual files, annotator "review", plus
      original_label and corrected (bool). Every file is written, even unchanged
      ones, so the folder is a complete drop-in label set.
  data/RESULTS_label_corrections_2026-10-01/
      proposals.csv            one row per changed bout, with the evidence
      needs_video_review.csv   pattern-2 bouts in degraded-pupil cycles
      gallery/                 one panel per proposed change, for accept/reject

    ./venv/bin/python -B scripts/propose_label_corrections.py
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

SNAP = ROOT / "data" / "labels_snapshot_2026-10-01"
OUT_LAB = ROOT / "data" / "labels_corrected_2026-10-01"
OUT = ROOT / "data" / "RESULTS_label_corrections_2026-10-01"
SHORT_NREM_MAX_BINS = 5
STABLE_MIN_BINS = 6
MIN_FRAC_ABOVE = 0.90
DEGRADED_PUPIL_DAYS = frozenset({"IP3R2_KD_2023.08.25", "IP3R2_KD_2023.08.26"})


def pattern2_bouts(labels):
    """(start_bin, n_bins) of NREM bouts <= SHORT_NREM_MAX_BINS with Awake on both sides."""
    b = C.bouts(labels).to_dict("records")
    out = []
    for i in range(1, len(b) - 1):
        p, c, n = b[i - 1], b[i], b[i + 1]
        if c["state"] == "NREM" and c["n_bins"] <= SHORT_NREM_MAX_BINS \
                and p["state"] == "Awake" and n["state"] == "Awake":
            out.append((c["start_bin"], c["n_bins"]))
    return out


def typical_levels(labels, pupil_bins):
    """Median bin pupil over stable (>= 30 s) Awake and NREM bouts."""
    lev = {}
    for st in ("Awake", "NREM"):
        idx = [i for b in C.bouts(labels).itertuples()
               if b.state == st and b.n_bins >= STABLE_MIN_BINS
               for i in range(b.start_bin, b.start_bin + b.n_bins)]
        v = pupil_bins[idx] if idx else np.array([])
        v = v[np.isfinite(v)]
        lev[st] = float(np.median(v)) if v.size else np.nan
    return lev


def decide(frac_samples_above_mid, awake_level, nrem_level):
    """True -> relabel this short NREM bout as Awake: the pupil stayed above the
    midpoint for >= MIN_FRAC_ABOVE of the bout. Refuses when the cycle has no
    sane separation (typical awake pupil not above typical NREM pupil)."""
    if not all(np.isfinite([frac_samples_above_mid, awake_level, nrem_level])):
        return False
    if awake_level <= nrem_level:
        return False
    return frac_samples_above_mid >= MIN_FRAC_ABOVE


def main():
    for d in (OUT_LAB, OUT):
        if d.exists():
            raise SystemExit(f"{d} exists; not overwriting a previous run")
    OUT_LAB.mkdir(parents=True)
    (OUT / "gallery").mkdir(parents=True)
    C.set_label_mode("manual")

    props, review, skipped = [], [], []
    for f in sorted(SNAP.glob("*_labels.csv")):
        key = f.name[:-11]
        info = C.parse_key(key)
        orig = pd.read_csv(f)
        lab = C.load_labels(key, "manual")
        new = lab.copy()
        cyc = C.load_cycle(key)
        cands = pattern2_bouts(lab)
        if cands:
            if "pupil" not in cyc.sig:
                skipped.append(dict(key=key, reason="no pupil"))
            else:
                t, y = cyc.sig["pupil"]
                pb = C.bin_series(t, y, len(lab))
                lev = typical_levels(lab, pb)
                mid = 0.5 * (lev["Awake"] + lev["NREM"])
                ys = pd.Series(y).rolling(int(round(C.FS_BEH)), center=True,
                                          min_periods=1).mean().to_numpy()
                for a, n in cands:
                    med = float(np.nanmedian(pb[a:a + n]))
                    seg = ys[(t >= a * C.BIN) & (t < (a + n) * C.BIN)]
                    seg = seg[np.isfinite(seg)]
                    frac_above = float(np.mean(seg > mid)) if seg.size else np.nan
                    min_pupil = float(seg.min()) if seg.size else np.nan
                    row = dict(key=key, group=info.group, day=info.mouse, start_bin=a,
                               t_start_s=a * C.BIN, dur_s=n * C.BIN, original="NREM",
                               bout_pupil_median=round(med, 3), typical_awake=round(lev["Awake"], 3),
                               typical_nrem=round(lev["NREM"], 3), midpoint=round(mid, 3),
                               margin_above_mid=round(med - mid, 3),
                               pct_samples_above_mid=round(100 * frac_above, 1),
                               min_pupil_minus_mid=round(min_pupil - mid, 3))
                    if info.mouse in DEGRADED_PUPIL_DAYS:
                        review.append({**row, "would_change": decide(frac_above, lev["Awake"], lev["NREM"]),
                                       "note": "degraded pupil: review against eye video"})
                        continue
                    if decide(frac_above, lev["Awake"], lev["NREM"]):
                        new[a:a + n] = "Awake"
                        props.append({**row, "proposed": "Awake"})

        out = orig.copy()
        lab_by_bin = {int(b): v for b, v in enumerate(new)}
        out["original_label"] = out["label"]
        out["label"] = [lab_by_bin.get(int(b), l) for b, l in zip(out["bin"], out["label"])]
        out["corrected"] = out["label"] != out["original_label"]
        out["annotator"] = "review"
        out["ts"] = "2026-10-01"
        out.to_csv(OUT_LAB / f.name, index=False)

    P, V = pd.DataFrame(props), pd.DataFrame(review)
    P.to_csv(OUT / "proposals.csv", index=False)
    V.to_csv(OUT / "needs_video_review.csv", index=False)

    # integrity: corrected files differ from originals ONLY in pattern-2 bins
    changed_bins = 0
    for f in sorted(OUT_LAB.glob("*_labels.csv")):
        a = pd.read_csv(SNAP / f.name)
        b = pd.read_csv(f)
        assert len(a) == len(b) and (a["bin"].values == b["bin"].values).all()
        diff = a["label"].values != b["label"].values
        assert (a.loc[diff, "label"] == "NREM").all() and (b.loc[diff, "label"] == "Awake").all()
        changed_bins += int(diff.sum())
    assert changed_bins == int(P["dur_s"].sum() / C.BIN) if len(P) else changed_bins == 0

    # gallery
    per = 24
    for pg in range(int(np.ceil(len(P) / per))):
        sub = P.iloc[pg * per:(pg + 1) * per]
        fig, axes = plt.subplots(4, 6, figsize=(18, 10.5))
        for ax, r in zip(axes.flat, sub.itertuples()):
            cyc = C.load_cycle(r.key)
            lab = C.load_labels(r.key, "manual")
            t0, t1 = r.t_start_s - 30, r.t_start_s + r.dur_s + 30
            for b in C.bouts(lab).itertuples():
                if b.state in C.STATE_COL and b.t_start + b.dur_s > t0 and b.t_start < t1:
                    ax.axvspan(max(b.t_start, t0), min(b.t_start + b.dur_s, t1), ymax=.1,
                               color=C.STATE_COL[b.state], lw=0)
            ax.axvspan(r.t_start_s, r.t_start_s + r.dur_s, color="k", alpha=.08, lw=0)
            t, y = cyc.sig["pupil"]
            m = (t >= t0) & (t <= t1)
            ax.plot(t[m], y[m], color=C.SIG_COL["pupil"], lw=.9)
            if "whisk" in cyc.sig:
                tw, yw = cyc.sig["whisk"]
                mw = (tw >= t0) & (tw <= t1)
                ax.plot(tw[mw], np.clip(yw[mw], -3, 5), color=C.SIG_COL["whisk"], lw=.4, alpha=.6)
            ax.axhline(r.typical_awake, color=C.STATE_COL["Awake"], ls=":", lw=.9)
            ax.axhline(r.typical_nrem, color=C.STATE_COL["NREM"], ls=":", lw=.9)
            ax.axhline(r.midpoint, color="k", ls="--", lw=.9)
            ax.set_xlim(t0, t1)
            ax.set_title(f"{r.key.replace('IP3R2_', '').replace('Extra_', '')}  t={r.t_start_s:.0f}s\n"
                         f"NREM {r.dur_s:.0f}s -> Awake | {r.pct_samples_above_mid:.0f}% above mid",
                         fontsize=6.5)
            ax.tick_params(labelsize=5)
        for ax in axes.flat[len(sub):]:
            ax.axis("off")
        fig.suptitle(f"Proposed corrections, pattern 2 (short NREM inside wake, pupil above midpoint) — "
                     f"page {pg + 1}. Dashed = midpoint, dotted = typical Awake (blue) / NREM (red) pupil, "
                     f"grey = bout to relabel. Originals unchanged.", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(OUT / "gallery" / f"page_{pg + 1:02d}.png", dpi=130)
        plt.close(fig)

    pd.set_option("display.width", 200)
    total_p2 = len(P) + len(V) + sum(1 for _ in [])
    print(f"proposed changes: {len(P)} bouts, {changed_bins} bins ({changed_bins * C.BIN / 60:.1f} min) "
          f"in {P.key.nunique() if len(P) else 0} files")
    if len(P):
        print("\nby group:")
        print(P.groupby("group").agg(bouts=("key", "size"), files=("key", "nunique"),
                                     minutes=("dur_s", lambda s: s.sum() / 60)).round(1).to_string())
        print("\nduration (s):", P.dur_s.value_counts().sort_index().to_dict())
        print("pupil samples above midpoint: median %.0f%%, min %.0f%%"
              % (P.pct_samples_above_mid.median(), P.pct_samples_above_mid.min()))
        print("lowest pupil in bout relative to midpoint (z): median %+.2f, min %+.2f"
              % (P.min_pupil_minus_mid.median(), P.min_pupil_minus_mid.min()))
    print(f"\nneeds video review (degraded pupil): {len(V)} bouts "
          f"({int(V.would_change.sum()) if len(V) else 0} would have been changed)")
    print(f"skipped (no pupil): {len(skipped)}")
    print(f"integrity check passed: only NREM->Awake in pattern-2 bins; "
          f"{len(list(OUT_LAB.glob('*.csv')))} corrected files written")
    print(f"\nwrote {OUT_LAB}\n      {OUT}")


if __name__ == "__main__":
    main()
