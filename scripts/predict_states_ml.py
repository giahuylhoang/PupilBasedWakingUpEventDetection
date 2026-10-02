#!/usr/bin/env python3
"""Predict Awake/NREM for the IP3R2 KD cycles Florence has not labelled.

Only 7 of 19 KD cycles carry manual labels (and one of those, 2025.11.01
cycle 9, is labelled for just 29 of 180 bins). This trains a classifier on
every manual label we have and fills in the rest, so Figures 6-8 can use all
19 KD cycles -- clearly marked as predicted.

WHAT THE MODEL MAY SEE
  Pupil and whisker only. Never arteriole, calcium or NE: Figures 7 and 8 test
  whether vascular / Ca2+ signals differ BY STATE between KD and scramble, so
  a state predicted from those signals would make that test circular.
  Eye motion is excluded too -- no IP3R2 cycle has DLC, so a model leaning on
  it could not be applied where it is needed.

  Pupil is not on a common scale across sessions (medians 24 to 11,772, raw in
  2025 vs pre-smoothed in 2023-24), so every pupil feature is computed on the
  within-cycle robust z after a 1-s low-pass. That puts raw and pre-smoothed
  sessions on the same footing; per-bin variance of the raw trace would
  otherwise encode processing generation, not state.

  Features per 5-s bin: pupil level / spread / slope, whisker level / spread /
  high-activity fraction, plus the same summarised over +/-1 and +/-3 bins of
  context and the cycle-wide percentile rank of pupil level. States last tens
  of seconds, so context carries much of the signal.

VALIDATION
  Leave-one-recording-day-out across all labelled days (bins within a day are
  autocorrelated and share an animal, so any random split leaks). Reported
  overall and separately for each held-out KD and scramble day -- the KD-day
  folds are the honest estimate of how well the KD predictions can be trusted.
  Awake vs NREM only: KD has zero REM labels and REM is <2% elsewhere.
  'Unclear' bins are excluded from training and scoring.

  Two decision rules are scored in the same CV and the better one by mean
  fold kappa is used for prediction: raw argmax, and argmax after a 3-bin
  running mean of P(NREM). That choice is made on CV, so its score is mildly
  optimistic; both are reported.

OUTPUTS (new folders; manual labels are never touched, nothing is uploaded)
  data/labels_predicted_2026-10-01/<key>_labels.csv   same schema as manual
        labels + p_nrem, source = predicted
  data/RESULTS_state_ml_2026-10-01/   CV tables, confusion matrices, figure

    ./venv/bin/python scripts/predict_states_ml.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figs_prelim import common as C  # noqa: E402

OUT_LAB = C.LABEL_PRED_DIR       # follows PRELIM_LABEL_SET (v2: original, v3: corrected)
OUT = C.ML_DIR
CLASSES = ("Awake", "NREM")
SEED = 7


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def _smooth(y, fs, win_s):
    k = max(1, int(round(win_s * fs)))
    s = pd.Series(y).rolling(k, center=True, min_periods=1).mean().to_numpy()
    return s


def cycle_features(key: str) -> pd.DataFrame | None:
    cyc = C.load_cycle(key)
    if "pupil" not in cyc.sig or "whisk" not in cyc.sig:
        return None
    n = int(round(900 / C.BIN))
    tp, yp = cyc.sig["pupil"]
    tw, yw = cyc.sig["whisk"]
    yp = _smooth(yp, C.FS_BEH, 1.0)          # equalise raw vs pre-smoothed sessions
    hi = np.nanpercentile(yw, 75)

    rows = []
    for i in range(n):
        a, b = i * C.BIN, (i + 1) * C.BIN
        p = yp[(tp >= a) & (tp < b)]
        w = yw[(tw >= a) & (tw < b)]
        p, w = p[np.isfinite(p)], w[np.isfinite(w)]
        if p.size < 5 or w.size < 5:
            rows.append({})
            continue
        x = np.arange(p.size) / C.FS_BEH
        rows.append(dict(
            pup_mean=p.mean(), pup_sd=p.std(), pup_min=p.min(), pup_max=p.max(),
            pup_slope=np.polyfit(x, p, 1)[0],
            whisk_mean=w.mean(), whisk_sd=w.std(), whisk_p90=np.percentile(w, 90),
            whisk_frac_hi=float(np.mean(w > hi))))
    f = pd.DataFrame(rows).reindex(columns=[
        "pup_mean", "pup_sd", "pup_min", "pup_max", "pup_slope",
        "whisk_mean", "whisk_sd", "whisk_p90", "whisk_frac_hi"])
    base = list(f.columns)
    for w in (1, 3):
        roll = f[base].rolling(2 * w + 1, center=True, min_periods=1).mean()
        f[[f"{c}_ctx{w}" for c in base]] = roll.to_numpy()
    f["pup_rank"] = f["pup_mean"].rank(pct=True)
    f["pup_ctx3_rank"] = f["pup_mean_ctx3"].rank(pct=True)
    f["pup_delta_prev"] = f["pup_mean"].diff()
    f["pup_delta_next"] = -f["pup_mean"].diff(-1)
    f.insert(0, "bin", np.arange(n))
    info = cyc.info
    f.insert(0, "key", key)
    f.insert(1, "mouse", info.mouse)
    f.insert(2, "group", info.group)
    return f


def feature_columns(df):
    return [c for c in df.columns if c not in ("key", "mouse", "group", "bin", "label")]


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def make_model():
    # HistGradientBoosting handles NaN natively (edges of a cycle, gaps).
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, class_weight="balanced", random_state=SEED)


def smooth_proba(df, p, k=3):
    """3-bin running mean of P(NREM) within each cycle."""
    out = np.empty_like(p)
    for key, idx in df.groupby("key").indices.items():
        order = idx[np.argsort(df["bin"].to_numpy()[idx])]
        out[order] = pd.Series(p[order]).rolling(k, center=True, min_periods=1).mean().to_numpy()
    return out


def score(y, yhat):
    return dict(n_bins=len(y), accuracy=float(np.mean(y == yhat)),
                balanced_acc=balanced_accuracy_score(y, yhat),
                kappa=cohen_kappa_score(y, yhat))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    if OUT_LAB.exists() and any(OUT_LAB.iterdir()):
        raise SystemExit(f"{OUT_LAB} already populated; never overwriting a previous run")
    OUT_LAB.mkdir(parents=True, exist_ok=True)

    # every cycle with pupil+whisker: labelled ones train, KD ones get predicted
    C.set_label_mode("manual")          # train on Florence's labels only, never on our own output
    keys = sorted(set(C.labeled_keys("manual")) | set(C.keys_for("KD", labeled_only=False)))
    feats = []
    for k in keys:
        try:
            C.parse_key(k)
        except ValueError:
            continue
        f = cycle_features(k)
        if f is None:
            print(f"  skip {k}: no pupil/whisker")
            continue
        lab = C.load_labels(k, "manual")
        lab = np.concatenate([lab, np.full(max(0, len(f) - len(lab)), None, dtype=object)])[: len(f)]
        f["label"] = lab
        feats.append(f)
    D = pd.concat(feats, ignore_index=True)
    X_cols = feature_columns(D)
    T = D[D["label"].isin(CLASSES)].reset_index(drop=True)
    print(f"cycles={D.key.nunique()}  training bins={len(T)}  days={T.mouse.nunique()}  "
          f"features={len(X_cols)}")
    print(T.groupby("group").agg(days=("mouse", "nunique"), bins=("bin", "size")).to_string())

    # ---- leave-one-day-out CV ----
    y = (T["label"] == "NREM").astype(int).to_numpy()
    p_cv = np.full(len(T), np.nan)
    for day in sorted(T.mouse.unique()):
        te = (T.mouse == day).to_numpy()
        m = make_model().fit(T.loc[~te, X_cols], y[~te])
        p_cv[te] = m.predict_proba(T.loc[te, X_cols])[:, 1]
    p_sm = smooth_proba(T, p_cv)

    rules = {"raw": (p_cv >= 0.5).astype(int), "smoothed_3bin": (p_sm >= 0.5).astype(int)}
    fold_rows, overall_rows = [], []
    for rule, yhat in rules.items():
        overall_rows.append(dict(rule=rule, scope="all held-out days", **score(y, yhat)))
        for g in ("WT", "SCR", "KD", "NE"):
            m = (T.group == g).to_numpy()
            if m.any():
                overall_rows.append(dict(rule=rule, scope=f"{g} days", **score(y[m], yhat[m])))
        for day in sorted(T.mouse.unique()):
            m = (T.mouse == day).to_numpy()
            r = score(y[m], yhat[m])
            r["kappa"] = r["kappa"] if len(np.unique(y[m])) > 1 else np.nan
            fold_rows.append(dict(rule=rule, day=day, group=T.loc[m, "group"].iloc[0], **r))
    majority = int(y.mean() >= 0.5)
    overall_rows.append(dict(rule="majority-class baseline", scope="all held-out days",
                             **score(y, np.full_like(y, majority))))
    O = pd.DataFrame(overall_rows)
    Fd = pd.DataFrame(fold_rows)
    O.to_csv(OUT / "tables" / "cv_overall.csv", index=False)
    Fd.to_csv(OUT / "tables" / "cv_per_day.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n=== leave-one-day-out CV ===")
    print(O.round(3).to_string(index=False))

    mean_k = Fd.groupby("rule").kappa.mean()
    best = mean_k.idxmax()
    print(f"\nmean per-day kappa: {mean_k.round(3).to_dict()}  -> using '{best}'")
    print("\nheld-out KD and scramble days:")
    print(Fd[(Fd.rule == best) & Fd.group.isin(["KD", "SCR"])].round(3).to_string(index=False))

    # ---- confusion + figure ----
    yb = rules[best]
    cm = confusion_matrix(y, yb)
    pd.DataFrame(cm, index=["true Awake", "true NREM"], columns=["pred Awake", "pred NREM"]).to_csv(
        OUT / "tables" / "confusion_all.csv")

    # ---- final model on ALL labelled bins -> predict unlabelled KD bins ----
    final = make_model().fit(T[X_cols], y)
    KD = D[D.group == "KD"].copy()
    KD["p_nrem"] = final.predict_proba(KD[X_cols])[:, 1]
    if best == "smoothed_3bin":
        KD["p_nrem"] = smooth_proba(KD.reset_index(drop=True), KD["p_nrem"].to_numpy())
    KD["pred"] = np.where(KD["p_nrem"] >= 0.5, "NREM", "Awake")

    written = []
    for key, s in KD.groupby("key"):
        s = s.sort_values("bin")
        manual = s["label"].to_numpy(dtype=object)
        has_manual = np.array([m in ("Awake", "NREM", "REM", "Unclear") for m in manual])
        if has_manual.mean() > 0.95:
            continue                              # fully labelled: keep manual only
        out = pd.DataFrame(dict(
            key=key, bin=s["bin"].to_numpy(),
            t_start=s["bin"].to_numpy() * C.BIN, t_end=(s["bin"].to_numpy() + 1) * C.BIN,
            # manual label wins wherever one exists; prediction fills the rest
            label=np.where(has_manual, manual, s["pred"].to_numpy()),
            annotator="ml_v1", ts="2026-10-01",
            source=np.where(has_manual, "manual", "predicted"),
            p_nrem=s["p_nrem"].round(4).to_numpy()))
        out.to_csv(OUT_LAB / f"{key}_labels.csv", index=False)
        written.append(dict(key=key, bins=len(out), manual_bins=int(has_manual.sum()),
                            predicted_bins=int((~has_manual).sum()),
                            pred_nrem_pct=100 * float(np.mean(out.loc[~has_manual, "label"] == "NREM"))
                            if (~has_manual).any() else np.nan,
                            mean_confidence=float(np.mean(np.abs(s["p_nrem"].to_numpy()[~has_manual] - 0.5) * 2))
                            if (~has_manual).any() else np.nan))
    W = pd.DataFrame(written)
    W.to_csv(OUT / "tables" / "predicted_cycles.csv", index=False)
    print(f"\n=== predicted labels written: {len(W)} KD cycles -> {OUT_LAB} ===")
    print(W.round(2).to_string(index=False))

    # sanity: does predicted KD architecture resemble manual KD architecture?
    man = T[T.group == "KD"]
    print(f"\nmanual KD bins NREM%: {100 * np.mean(man.label == 'NREM'):.1f}  "
          f"predicted KD bins NREM%: {100 * np.mean(KD.loc[~KD.label.isin(CLASSES + ('REM', 'Unclear')), 'pred'] == 'NREM'):.1f}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    fb = Fd[Fd.rule == best].sort_values(["group", "day"])
    cols = [C.GROUP_COL.get(g, "k") for g in fb.group]
    ax.bar(range(len(fb)), fb.kappa, color=cols)
    ax.set_xticks(range(len(fb)))
    ax.set_xticklabels([d.replace("Extra_", "").replace("IP3R2_", "") for d in fb.day],
                       rotation=75, fontsize=7)
    ax.axhline(0.6, color="k", ls=":", lw=1)
    ax.set_ylabel("Cohen's κ on held-out day")
    ax.set_title(f"Leave-one-day-out κ ({best})\ncolour = group; dotted = 0.6 (substantial)", fontsize=10)
    for g, c in C.GROUP_COL.items():
        ax.bar([], [], color=c, label=C.GROUP_LABEL[g])
    ax.legend(fontsize=7)

    ax = axes[1]
    cmn = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if cmn[i, j] > .5 else "k")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred Awake", "pred NREM"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true Awake", "true NREM"])
    ov = O[(O.rule == best) & (O.scope == "all held-out days")].iloc[0]
    ax.set_title(f"All held-out bins: acc {ov.accuracy:.2f}, bal {ov.balanced_acc:.2f}, κ {ov.kappa:.2f}",
                 fontsize=10)

    ax = axes[2]
    ex = W.sort_values("predicted_bins", ascending=False).key.iloc[0] if len(W) else None
    if ex is not None:
        s = KD[KD.key == ex].sort_values("bin")
        ax.plot(s.bin * C.BIN, s.p_nrem, color="k", lw=1)
        ax.axhline(0.5, color="0.5", ls="--", lw=.8)
        cyc = C.load_cycle(ex)
        tp, yp = cyc.sig["pupil"]
        ax2 = ax.twinx()
        ax2.plot(tp, yp, color=C.SIG_COL["pupil"], lw=.4, alpha=.6)
        ax2.set_ylabel("pupil z", fontsize=8)
        C.shade_states(ax, np.array(s.pred.tolist(), dtype=object))
        ax.set_ylim(0, 1); ax.set_xlabel("time (s)"); ax.set_ylabel("P(NREM)")
        ax.set_title(f"Example prediction: {ex}", fontsize=9)
    fig.suptitle("ML state model (pupil + whisker only) — trained on manual labels, "
                 "used to fill unlabelled IP3R2 KD cycles.  " + C.PRELIM_STAMP, fontsize=9, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "ml_state_model_cv.png", dpi=160, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
