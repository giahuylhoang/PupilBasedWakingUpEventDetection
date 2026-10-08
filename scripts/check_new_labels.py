"""Check a fresh download of manual labels (read-only on both label folders).

    ./venv/bin/python -B scripts/check_new_labels.py NEW_DIR [--old data/labels_snapshot_2026-10-01]

For every KD / scramble cycle on the site:
  coverage        labelled 5-s bins / bins in the cycle; Unclear count
  changed         bins whose label differs from the old snapshot (existing labels edited?)
  state %         Awake / NREM / REM
  pupil check     median pupil z in Awake minus NREM (should be clearly > 0)
  eye check       median eye-speed z in Awake minus NREM, where an eye trace exists
  short bouts     single-bin (5 s) bouts, a sign of noisy labelling
  vs ML (KD)      agreement / Cohen's kappa with the pupil+whisker model predictions
Writes NEW_DIR/../RESULTS_label_check_<name>/{per_cycle.csv, changes.csv}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "figs_prelim"))
import common as C  # noqa: E402

GROUPS = ("KD", "SCR")
PIX_EYE = [C.ROOT / "data" / "pupil_eye_motion_pixel_2026-10-05",
           C.ROOT / "data" / "pupil_eye_motion_pixel_control_2026-10-06"]


def read_labels(p: Path) -> np.ndarray:
    return C._read_label_file(p)


def kappa(a, b):
    m = np.array([x is not None and y is not None and x in C.STATES and y in C.STATES for x, y in zip(a, b)])
    if m.sum() < 20:
        return np.nan, np.nan, 0
    a, b = np.array(a, object)[m], np.array(b, object)[m]
    po = (a == b).mean()
    pe = sum((a == s).mean() * (b == s).mean() for s in C.STATES)
    return po, (po - pe) / (1 - pe) if pe < 1 else np.nan, int(m.sum())


def eye_trace(key):
    for d in PIX_EYE:
        p = d / f"{key}_eyemotion_perframe.csv"
        if p.exists():
            e = pd.read_csv(p, usecols=["time", "eye_speed"])
            return e["time"].to_numpy(float), e["eye_speed"].to_numpy(float)
    cyc = C.load_cycle(key)
    return cyc.sig.get("eye", (None, None))


def state_median(t, y, lab, state):
    if t is None:
        return np.nan
    b = np.minimum((t // C.BIN).astype(int), len(lab) - 1)
    m = (lab[b] == state) & np.isfinite(y)
    return float(np.median(y[m])) if m.sum() > 50 else np.nan


def short_bouts(lab):
    bt = C.bouts(lab)
    return int(((bt["n_bins"] == 1) & bt["state"].isin(C.STATES)).sum()) if len(bt) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("--old", default=str(C.ROOT / "data" / "labels_snapshot_2026-10-01"))
    a = ap.parse_args()
    new, old = Path(a.new), Path(a.old)
    out = new.parent / f"RESULTS_label_check_{new.name}"
    out.mkdir(exist_ok=True)

    rows, changes = [], []
    for key in C.keys_for(GROUPS, labeled_only=False):
        cyc = C.load_cycle(key)
        n_exp = int(np.ceil(max(cyc.sig["pupil"][0][-1] if "pupil" in cyc.sig else 900, 900) / C.BIN))
        L = read_labels(new / f"{key}_labels.csv")
        O = read_labels(old / f"{key}_labels.csv")
        r = dict(group=cyc.info.group, day=cyc.info.mouse.split("_", 2)[-1], cycle=cyc.info.cycle, key=key,
                 n_bins=n_exp, labelled=int(sum(v is not None for v in L)),
                 unclear=int(sum(v == "Unclear" for v in L)), was_labelled=int(sum(v is not None for v in O)))
        r["coverage"] = r["labelled"] / n_exp
        # edits to bins that were labelled before
        n = min(len(L), len(O))
        ch = [(i, O[i], L[i]) for i in range(n) if O[i] is not None and O[i] != L[i]]
        ch += [(i, O[i], None) for i in range(n, len(O)) if O[i] is not None]
        r["changed_vs_old"] = len(ch)
        changes += [dict(key=key, bin=i, old=o, new=nw) for i, o, nw in ch]
        lab = np.array(list(L) + [None] * max(0, n_exp - len(L)), object)
        lv = [v for v in lab if v in C.STATES]
        for s in C.STATES:
            r[f"pct_{s}"] = 100 * lv.count(s) / len(lv) if lv else np.nan
        if "pupil" in cyc.sig:
            t, y = cyc.sig["pupil"]
            r["pupil_awake_minus_nrem"] = state_median(t, y, lab, "Awake") - state_median(t, y, lab, "NREM")
        t, y = eye_trace(key)
        r["eye_awake_minus_nrem"] = state_median(t, y, lab, "Awake") - state_median(t, y, lab, "NREM")
        r["single_bin_bouts"] = short_bouts(lab)
        pp = C.LABEL_PRED_DIR / f"{key}_labels.csv"
        if pp.exists():
            # only bins the model filled (unlabelled in the old snapshot), so the
            # old manual labels copied into the prediction file don't inflate it
            P = read_labels(pp)
            Pm = [P[i] if i < len(P) and (i >= len(O) or O[i] is None) else None for i in range(len(lab))]
            r["ml_agree"], r["ml_kappa"], r["ml_bins"] = kappa(lab, Pm)
        rows.append(r)

    d = pd.DataFrame(rows).sort_values(["group", "day", "cycle"])
    d.to_csv(out / "per_cycle.csv", index=False)
    pd.DataFrame(changes, columns=["key", "bin", "old", "new"]).to_csv(out / "changes.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["group", "day", "cycle", "coverage", "unclear", "was_labelled", "changed_vs_old",
            "pct_Awake", "pct_NREM", "pct_REM", "pupil_awake_minus_nrem", "eye_awake_minus_nrem",
            "single_bin_bouts", "ml_kappa"]
    print(d[cols].round(2).to_string(index=False))
    print("\nlabel values seen:", sorted({v for k in d.key for v in read_labels(new / f'{k}_labels.csv') if v}))
    print("cycles fully labelled:", int((d.coverage >= 0.99).sum()), "/", len(d),
          "| previously labelled bins edited:", len(changes))
    flag = d[(d.coverage < 0.99) | (d.pupil_awake_minus_nrem <= 0.3) | (d.eye_awake_minus_nrem <= 0)]
    print("\nFLAGS (coverage <99%, pupil Awake-NREM <=0.3 z, or eye Awake<=NREM):")
    print(flag[cols].round(2).to_string(index=False) if len(flag) else "none")
    print("->", out)


if __name__ == "__main__":
    main()
