#!/usr/bin/env python3
"""Do the two event refinements change the results? (sensitivity analysis)

Label-based events have two known weaknesses (see the 2026-10-01 discussion):
  1. t0 is a 5-s label boundary, so the real arousal can sit inside the -5..0 s
     baseline, which shrinks peaks and blurs latency.
  2. Neighbouring events (micro-arousals cluster) can fall inside each other's
     baseline/response windows.

Four event variants, same everything else:
  original      label t0, all events                 (what the figures use)
  anchored      t0 moved to pupil dilation onset     (C.anchor_to_pupil)
  isolated      events with a neighbour within -15..+20 s dropped (C.drop_overlapping)
  both          anchored + isolated

Re-computed under each variant:
  A. WT (Fig 3): N2W and MA response size, response probability, time-to-peak,
     Ca-minus-arteriole latency; mouse-level mean +/- SEM.
  B. KD vs scramble (Fig 8): arteriole and Ca peak / AUC, mixed model
     value ~ group + pupil_peak + (1|day), in both label modes.

Pupil anchoring uses pupil only, never arteriole or Ca, so it cannot select
for the outcome; the same rule is applied identically to KD and scramble.

Outputs -> data/RESULTS_event_refinement_2026-10-01/ (new folder).

    ./venv/bin/python -B scripts/event_refinement_sensitivity.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figs_prelim import common as C  # noqa: E402

OUT = ROOT / "data" / "RESULTS_event_refinement_2026-10-01"
VARIANTS = ("original", "anchored", "isolated", "both")
VAR_COL = {"original": "#424242", "anchored": "#1E88E5", "isolated": "#43A047", "both": "#E53935"}
SIGNALS = ("pupil", "art", "ca")


def variant(ev: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "original":
        e = ev.copy()
    elif name == "anchored":
        e = C.anchor_to_pupil(ev)
    elif name == "isolated":
        e = C.drop_overlapping(ev)
    else:
        e = C.drop_overlapping(C.anchor_to_pupil(ev))
    return e.reset_index(drop=True)


def event_metrics(e: pd.DataFrame) -> pd.DataFrame:
    """Per-event peak / AUC / time-to-peak / responder for pupil, art, ca."""
    out = e[["type", "key", "mouse", "group", "t0"]].copy()
    for s in SIGNALS:
        rg, M = C.epochs_for(e, s)
        pk, auc, tpk = C.response_metrics(rg, M)
        out[f"{s}_peak"], out[f"{s}_auc"], out[f"{s}_tpk"] = pk, auc, tpk
        out[f"{s}_resp"] = C.responder(rg, M)
    out["ca_minus_art_tpk"] = out["ca_tpk"] - out["art_tpk"]
    return out


def mouse_summary(m: pd.DataFrame, cols) -> dict:
    g = m.groupby("mouse")[cols].mean()
    res = {"n_events": len(m), "n_days": m.mouse.nunique()}
    for c in cols:
        v = g[c].dropna()
        res[c] = v.mean()
        res[c + "_sem"] = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    return res


DEGENERATE_RATIO = 0.01   # day variance < 1% of residual variance -> treated as collapsed


def mixed_p(d: pd.DataFrame, value: str):
    """(coef, p, n, day_variance). A day variance that collapses to ~0 means the
    model has silently become event-level OLS -- events treated as independent,
    the May pseudoreplication -- so its p must not be read (flagged downstream)."""
    d = d.dropna(subset=[value, "pupil_peak"])
    if d.mouse.nunique() < 3 or d.scr.nunique() < 2:
        return np.nan, np.nan, len(d), np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = smf.mixedlm(f"{value} ~ scr + pupil_peak", d, groups=d.mouse).fit(reml=True)
            return r.params["scr"], r.pvalues["scr"], len(d), float(r.cov_re.iloc[0, 0]) / float(r.scale)
        except Exception:  # noqa: BLE001
            return np.nan, np.nan, len(d), np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    C.set_label_mode("manual+ml")

    # ---------------- how much do the refinements move / remove events? -------
    ev_all = C.event_table(C.keys_for(["WT", "SCR", "KD"]))
    ev_all = ev_all[ev_all.type.isin(["N2W", "MA"])].reset_index(drop=True)
    anch = C.anchor_to_pupil(ev_all)
    iso = C.drop_overlapping(ev_all)
    rows = []
    for (g, typ), s in anch.groupby(["group", "type"]):
        k = iso[(iso.group == g) & (iso.type == typ)]
        a = s[s.anchored]
        rows.append(dict(group=g, type=typ, n_events=len(s),
                         pct_anchored=100 * s.anchored.mean(),
                         shift_median_s=a.shift_s.median(),
                         shift_iqr_s=f"{a.shift_s.quantile(.25):+.1f} to {a.shift_s.quantile(.75):+.1f}",
                         pct_earlier_than_label=100 * (a.shift_s < 0).mean() if len(a) else np.nan,
                         n_isolated=len(k), pct_kept_isolated=100 * len(k) / len(s)))
    shifts = pd.DataFrame(rows)
    shifts.to_csv(OUT / "event_changes.csv", index=False)
    pd.set_option("display.width", 200)
    print("=== how the refinements change the event sets (manual+ml labels) ===")
    print(shifts.round(1).to_string(index=False))

    # ---------------- A. WT responses (Fig 3) -----------------------------------
    cols = ["art_peak", "ca_peak", "art_auc", "ca_auc", "art_resp", "ca_resp",
            "art_tpk", "ca_tpk", "ca_minus_art_tpk", "pupil_peak"]
    evw = C.event_table(C.keys_for("WT"))
    A, eta = [], {}
    for typ in ("N2W", "MA"):
        base = evw[evw.type == typ].reset_index(drop=True)
        for v in VARIANTS:
            e = variant(base, v)
            m = event_metrics(e)
            A.append(dict(type=typ, variant=v, **mouse_summary(m, cols)))
            for s in SIGNALS:
                rg, M = C.epochs_for(e, s)
                per_mouse = pd.DataFrame(M).groupby(e.mouse.values).mean()
                eta[(typ, v, s)] = (rg, per_mouse.mean().to_numpy(),
                                    per_mouse.std(ddof=1).to_numpy() / np.sqrt(len(per_mouse)))
    A = pd.DataFrame(A)
    A.to_csv(OUT / "A_WT_responses_by_variant.csv", index=False)
    print("\n=== A. WT responses (mouse-level means), Fig 3 ===")
    show = ["type", "variant", "n_events", "n_days", "art_peak", "ca_peak", "art_resp",
            "ca_resp", "art_tpk", "ca_tpk", "ca_minus_art_tpk"]
    print(A[show].round(3).to_string(index=False))

    # ---------------- B. KD vs scramble (Fig 8) ---------------------------------
    B = []
    for mode in ("manual+ml", "manual"):
        C.set_label_mode(mode)
        ev = C.event_table(C.keys_for(["SCR", "KD"]))
        for typ in ("N2W", "MA"):
            base = ev[ev.type == typ].reset_index(drop=True)
            for v in VARIANTS:
                m = event_metrics(variant(base, v))
                m["scr"] = (m.group == "SCR").astype(int)
                for val in ("art_peak", "art_auc", "ca_peak", "ca_auc"):
                    coef, p, n, dvar = mixed_p(m, val)
                    mm = m.groupby(["group", "mouse"])[val].mean().reset_index()
                    gt = C.group_test(mm[mm.group == "SCR"][val], mm[mm.group == "KD"][val])
                    B.append(dict(mode=mode, type=typ, variant=v, measure=val,
                                  n_events=n, n_kd_events=int((m.group == "KD").sum()),
                                  scr_mean=gt["mean_a"], kd_mean=gt["mean_b"],
                                  kd_over_scr=gt["mean_b"] / gt["mean_a"] if gt["mean_a"] else np.nan,
                                  mixed_scr_minus_kd=coef, mixed_p=p, day_var_over_resid=dvar,
                                  degenerate=bool(np.isfinite(dvar) and dvar < DEGENERATE_RATIO),
                                  mouse_welch_p=gt["welch_p"], hedges_g=gt["hedges_g"]))
    C.set_label_mode("manual+ml")
    B = pd.DataFrame(B)
    B.to_csv(OUT / "B_KD_vs_SCR_by_variant.csv", index=False)
    print("\n=== B. KD vs scramble, Fig 8 (mixed model value ~ group + pupil_peak + (1|day)) ===")
    for mode in ("manual+ml", "manual"):
        print(f"\n-- labels: {mode}")
        s = B[B["mode"] == mode]
        piv = s.pivot_table(index=["type", "measure"], columns="variant",
                            values="mixed_p").reindex(columns=VARIANTS)
        rat = s.pivot_table(index=["type", "measure"], columns="variant",
                            values="kd_over_scr").reindex(columns=VARIANTS)
        nev = s.pivot_table(index=["type", "measure"], columns="variant",
                            values="n_events").reindex(columns=VARIANTS)
        deg = s.pivot_table(index=["type", "measure"], columns="variant",
                            values="degenerate", aggfunc="max").reindex(columns=VARIANTS)
        pshow = piv.round(3).astype(str).where(~deg.astype(bool), "DEGEN")
        tab = pd.concat({"mixed p (DEGEN = day variance 0, unreadable)": pshow,
                         "KD/SCR": rat.round(2), "n ev": nev}, axis=1)
        print(tab.to_string())

    # ---------------- figure ----------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.2), sharex=True)
    for r, typ in enumerate(("N2W", "MA")):
        for c, s in enumerate(SIGNALS):
            ax = axes[r, c]
            for v in VARIANTS:
                rg, mu, se = eta[(typ, v, s)]
                n = A[(A.type == typ) & (A.variant == v)].n_events.iloc[0]
                ax.plot(rg, mu, color=VAR_COL[v], lw=1.6, label=f"{v} (n={n})")
                ax.fill_between(rg, mu - se, mu + se, color=VAR_COL[v], alpha=.12, lw=0)
            ax.axvspan(*C.BASE_WIN, color="0.85", lw=0, zorder=0)
            ax.axvspan(*C.RESP_WIN, color="#FFF3E0", lw=0, zorder=0)
            ax.axvline(0, color="k", lw=.7)
            ax.set_xlim(-20, 40)
            ax.set_title(f"WT {C.EVENT_LABEL[typ]} — {C.SIG_LABEL[s]}", fontsize=10)
            if r == 1:
                ax.set_xlabel("time from t0 (s)")
            if c == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Event refinements: label t0 vs pupil-anchored t0 vs non-overlapping events "
                 "(mouse-level mean ± SEM; grey = baseline, orange = response window)\n"
                 + C.PRELIM_STAMP, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "WT_eta_by_variant.png", dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(11, 4.2))
    s = B[(B["mode"] == "manual+ml")]
    labels = [f"{t} {m}" for t in ("N2W", "MA") for m in ("art_peak", "art_auc", "ca_peak", "ca_auc")]
    x = np.arange(len(labels))
    for k, v in enumerate(VARIANTS):
        rows_ = [s[(s.type == lab.split()[0]) & (s.measure == lab.split()[1]) & (s.variant == v)].iloc[0]
                 for lab in labels]
        h = [min(-np.log10(r.mixed_p), 4) for r in rows_]
        bars = ax.bar(x + (k - 1.5) * .2, h, .2, color=VAR_COL[v], label=v)
        for b, r in zip(bars, rows_):
            if r.degenerate:
                b.set_hatch("///"); b.set_facecolor("white"); b.set_edgecolor(VAR_COL[v])
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .05, "fit\ndegenerate",
                        ha="center", fontsize=6, color=VAR_COL[v])
    ax.set_ylim(0, 4.6)
    ax.axhline(-np.log10(.05), color="k", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel("−log10 p  (KD vs scramble, mixed model)")
    ax.set_title("Fig 8 KD vs scramble by event variant (manual+ML labels). Dotted = p 0.05; hatched = day variance\n"
                 "collapsed to 0 (model silently became event-level OLS) -> p unreadable; bars capped at 4",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "KD_vs_SCR_p_by_variant.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
