#!/usr/bin/env python3
"""Scan the manual sleep-state labels for likely labelling errors. READ-ONLY.

Never modifies, moves or deletes any label file. Reads the locked snapshot
(data/labels_snapshot_2026-10-01/, byte-identical to the bucket on 2026-10-01) and
writes a report to a new folder.

Checks
  1. Structure: bins missing / duplicated / beyond 180, t_start != 5*bin,
     unknown labels, files edited in several sessions.
  2. Short isolated bouts, both directions:
       NREM inside wake   Awake - NREM(<= 25 s) - Awake
       wake inside NREM   NREM - Awake(<= 15 s) - NREM   (this IS the micro-arousal
                          definition, so these are not errors by default)
     plus short REM, and REM entered directly from Awake (Awake -> REM).
  3. Is the short bout physiologically like the state it was given?
     For each short bout, compare its pupil z and whisker z (within-cycle) with
     the same cycle's long, stable bouts of each state. A bout "looks like" the
     state it is closer to (distance in pupil/whisker, scaled by that cycle's
     spread). Also report an out-of-fold ML second opinion: P(NREM) from the
     pupil+whisker model trained with that recording DAY held out, so the bin's
     own label never trained it.
  4. Per-file "flicker": transitions per hour and short-bout counts, to spot files
     scored with a different style.

Output -> data/RESULTS_label_scan_2026-10-01/
    ./venv/bin/python -B scripts/scan_labels.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figs_prelim import common as C  # noqa: E402

SNAP = ROOT / "data" / "labels_snapshot_2026-10-01"
OUT = ROOT / "data" / "RESULTS_label_scan_2026-10-01"
SHORT_NREM_MAX_BINS = 5      # <= 25 s of NREM inside wake
SHORT_WAKE_MAX_BINS = 3      # <= 15 s of wake inside NREM (= micro-arousal range)
STABLE_MIN_BINS = 6          # reference bouts: >= 30 s
VALID = {"Awake", "NREM", "REM", "Unclear"}


def load_predictor():
    spec = importlib.util.spec_from_file_location("p", ROOT / "scripts" / "predict_states_ml.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    return P


def bouts(lab):
    out, i = [], 0
    while i < len(lab):
        j = i
        while j + 1 < len(lab) and lab[j + 1] == lab[i]:
            j += 1
        out.append((lab[i], i, j - i + 1))
        i = j + 1
    return out


def bin_features(key, n):
    cyc = C.load_cycle(key)
    feats = {}
    for s in ("pupil", "whisk", "eye"):
        if s in cyc.sig:
            t, y = cyc.sig[s]
            feats[s] = C.bin_series(t, y, n)
    return feats


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; not overwriting a previous scan")
    OUT.mkdir(parents=True)
    C.set_label_mode("manual")
    files = sorted(SNAP.glob("*_labels.csv"))

    # ---------------- 1. structure ----------------
    struct = []
    for f in files:
        d = pd.read_csv(f)
        key = f.name[:-11]
        bins = d["bin"].astype(int)
        issues = []
        if bins.duplicated().any():
            issues.append(f"{int(bins.duplicated().sum())} duplicated bins")
        missing = sorted(set(range(int(bins.max()) + 1)) - set(bins))
        if missing:
            issues.append(f"{len(missing)} missing bins (e.g. {missing[:5]})")
        if bins.max() >= 181:
            issues.append(f"bins up to {bins.max()} (>180)")
        bad = set(d["label"].astype(str)) - VALID
        if bad:
            issues.append(f"unknown labels {sorted(bad)}")
        off = (np.abs(d["t_start"] - d["bin"] * C.BIN) > 0.01).sum()
        if off:
            issues.append(f"{off} rows with t_start != 5*bin")
        sessions = d["ts"].astype(str).nunique()
        struct.append(dict(key=key, n_rows=len(d), max_bin=int(bins.max()),
                           annotator=",".join(sorted(d["annotator"].astype(str).unique())),
                           sessions=sessions, last_saved=str(d["ts"].max())[:16],
                           issues="; ".join(issues) or "ok"))
    S = pd.DataFrame(struct)
    S.to_csv(OUT / "1_structure.csv", index=False)

    # ---------------- out-of-fold ML second opinion ----------------
    P = load_predictor()
    feats = []
    for f in files:
        k = f.name[:-11]
        fe = P.cycle_features(k)
        if fe is None:
            continue
        lab = C.load_labels(k, "manual")
        lab = np.concatenate([lab, np.full(max(0, len(fe) - len(lab)), None, dtype=object)])[: len(fe)]
        fe["label"] = lab
        feats.append(fe)
    D = pd.concat(feats, ignore_index=True)
    X = P.feature_columns(D)
    D["p_nrem_oof"] = np.nan
    train = D["label"].isin(["Awake", "NREM"])
    for day in sorted(D["mouse"].unique()):
        te = D["mouse"] == day
        m = P.make_model().fit(D.loc[train & ~te, X], (D.loc[train & ~te, "label"] == "NREM").astype(int))
        D.loc[te, "p_nrem_oof"] = m.predict_proba(D.loc[te, X])[:, 1]
    oof = {(r.key, int(r.bin)): r.p_nrem_oof for r in D[["key", "bin", "p_nrem_oof"]].itertuples()}

    # ---------------- 2+3. short bouts and odd sequences ----------------
    rows, flick = [], []
    for f in files:
        key = f.name[:-11]
        info = C.parse_key(key)
        lab = C.load_labels(key, "manual")
        n = len(lab)
        b = bouts(list(lab))
        fe = bin_features(key, n)

        # per-cycle reference: stable bouts of each state
        ref = {}
        for st in ("Awake", "NREM"):
            idx = [i for (s, a, ln) in b if s == st and ln >= STABLE_MIN_BINS for i in range(a, a + ln)]
            ref[st] = {k: np.nanmedian(v[idx]) if idx else np.nan for k, v in fe.items()}
        spread = {k: np.nanstd(v) or 1.0 for k, v in fe.items()}

        def looks_like(a, ln):
            sl = slice(a, a + ln)
            val = {k: np.nanmean(v[sl]) for k, v in fe.items()}
            dist = {}
            for st in ("Awake", "NREM"):
                ds = [((val[k] - ref[st][k]) / spread[k]) ** 2 for k in ("pupil", "whisk")
                      if k in val and np.isfinite(val[k]) and np.isfinite(ref[st].get(k, np.nan))]
                dist[st] = np.sqrt(np.mean(ds)) if ds else np.nan
            if not all(np.isfinite(list(dist.values()))):
                return val, dist, "unknown"
            return val, dist, min(dist, key=dist.get)

        n_short = 0
        for i, (st, a, ln) in enumerate(b):
            prev = b[i - 1][0] if i > 0 else None
            nxt = b[i + 1][0] if i + 1 < len(b) else None
            kind = None
            if st == "NREM" and prev == "Awake" and nxt == "Awake" and ln <= SHORT_NREM_MAX_BINS:
                kind = "short NREM inside wake"
            elif st == "Awake" and prev == "NREM" and nxt == "NREM" and ln <= SHORT_WAKE_MAX_BINS:
                kind = "short wake inside NREM (micro-arousal)"
            elif st == "REM" and prev == "Awake":
                kind = "REM entered from Awake"
            elif st == "REM" and ln <= 2:
                kind = "short REM (<= 10 s)"
            if kind is None:
                continue
            n_short += 1
            val, dist, like = looks_like(a, ln)
            p = np.nanmean([oof.get((key, j), np.nan) for j in range(a, a + ln)])
            ml_says = "NREM" if p >= 0.5 else "Awake" if np.isfinite(p) else "n/a"
            rows.append(dict(
                key=key, group=info.group, day=info.mouse, kind=kind, label=st,
                start_bin=a, t_start_s=a * C.BIN, dur_s=ln * C.BIN,
                before=prev, after=nxt,
                pupil_z=round(val.get("pupil", np.nan), 2), whisk_z=round(val.get("whisk", np.nan), 2),
                eye_z=round(val.get("eye", np.nan), 2) if "eye" in val else np.nan,
                ref_awake_pupil=round(ref["Awake"].get("pupil", np.nan), 2),
                ref_nrem_pupil=round(ref["NREM"].get("pupil", np.nan), 2),
                signals_look_like=like, ml_p_nrem=round(p, 2) if np.isfinite(p) else np.nan,
                ml_says=ml_says, last_saved=S.set_index("key").loc[key, "last_saved"]))
        trans = sum(1 for i in range(1, len(b)) if b[i][0] != b[i - 1][0])
        flick.append(dict(key=key, group=info.group, day=info.mouse,
                          last_saved=S.set_index("key").loc[key, "last_saved"][:10],
                          transitions_per_h=trans / (n * C.BIN / 3600), short_bouts=n_short,
                          one_bin_bouts=sum(1 for (s, a, ln) in b if ln == 1 and s in ("Awake", "NREM")),
                          pct_unclear=100 * np.mean(lab == "Unclear")))
    R = pd.DataFrame(rows)

    # verdict: agree / suspicious (signals AND ML say the other state) / ambiguous
    def verdict(r):
        if r.kind.startswith("REM"):
            return "check (REM rule)"
        other = "Awake" if r.label == "NREM" else "NREM"
        votes = [r.signals_look_like == other, r.ml_says == other]
        if all(votes):
            return "LIKELY MISLABELLED"
        if any(votes):
            return "ambiguous"
        return "consistent with label"
    if len(R):
        R["verdict"] = R.apply(verdict, axis=1)
    R.to_csv(OUT / "2_short_bouts.csv", index=False)
    F = pd.DataFrame(flick)
    F.to_csv(OUT / "4_flicker_by_file.csv", index=False)

    # ---------------- report ----------------
    pd.set_option("display.width", 220)
    print("=== 1. structure ===")
    print(S.issues.value_counts().to_string())
    print(f"files edited in >1 session: {(S.sessions > 1).sum()} / {len(S)}")
    bad = S[S.issues != "ok"]
    if len(bad):
        print(bad[["key", "n_rows", "max_bin", "issues"]].to_string(index=False))

    print("\n=== 2-3. short bouts: count by kind x verdict ===")
    if len(R):
        print(pd.crosstab(R.kind, R.verdict, margins=True).to_string())
        print("\nby group:")
        print(pd.crosstab([R.group, R.kind], R.verdict).to_string())
        print("\nduration of 'short NREM inside wake' (s):",
              R[R.kind == "short NREM inside wake"].dur_s.value_counts().sort_index().to_dict())
        print("duration of 'short wake inside NREM' (s):",
              R[R.kind.str.startswith("short wake")].dur_s.value_counts().sort_index().to_dict())

    print("\n=== 4. flicker by labelling date ===")
    print(F.groupby("last_saved").agg(files=("key", "size"), trans_per_h=("transitions_per_h", "mean"),
                                      short_bouts=("short_bouts", "mean"),
                                      one_bin=("one_bin_bouts", "mean"),
                                      unclear=("pct_unclear", "mean")).round(1).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
