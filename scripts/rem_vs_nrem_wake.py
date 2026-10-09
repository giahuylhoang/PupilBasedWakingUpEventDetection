"""REM -> wake vs NREM -> wake, arteriole and Ca2+ (Florence's explanation).

Idea tested: REM dilates the vessel, so waking from REM looks like a
constriction, while NREM -> wake dilates.

Events (manual labels, label set from PRELIM_LABEL_SET; run with relabel_2026-10-09):
  N2W  common.response_events, type N2W (NREM >= 30 s, then Awake >= 20 s;
       pupil-anchored; neighbours within -15..+20 s dropped)
  R2W  REM bout >= 20 s, then Awake >= 20 s. t0 = pupil dilation onset
       (common.pupil_onset), else the label boundary. Not overlap-filtered:
       there are too few to lose any; neighbours are reported.
Only days that have R2W events are plotted, so N2W and R2W come from the same
vessels.

Two arteriole views:
  ΔD/D_pre      (D - D_pre)/D_pre, D_pre = median -20..0 s  (relative to what came before)
  D/D_median-1  absolute level vs the cycle median           (where the vessel actually is)

Output: data/RESULTS_rem_vs_nrem_wake_2026-10-09/
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

OUT = C.ROOT / "data" / "RESULTS_rem_vs_nrem_wake_2026-10-09"
PRE = (-20.0, 0.0)
VIEW = (-30.0, 60.0)
COL = {"N2W": "#EF6C00", "R2W": "#6A1B9A"}
NAME = {"N2W": "NREM → wake", "R2W": "REM → wake"}
GROUPS = ("WT", "SCR", "NE")


def r2w_events(keys):
    rows = []
    for k in keys:
        cyc = C.load_cycle(k)
        bt = C.bouts(cyc.labels).to_dict("records")
        for a, b in zip(bt, bt[1:]):
            if a["state"] == "REM" and a["n_bins"] >= 4 and b["state"] == "Awake" and b["n_bins"] >= 4:
                t_lab = b["t_start"]
                on = None
                if "pupil" in cyc.sig:
                    t, y = cyc.sig["pupil"]
                    on = C.pupil_onset(t, y, t_lab)
                rows.append(dict(type="R2W", key=k, mouse=cyc.info.mouse, group=cyc.info.group,
                                 t0=on if on is not None else t_lab, t0_label=t_lab, anchored=on is not None,
                                 pre_dur_s=a["dur_s"], post_dur_s=b["dur_s"]))
    return pd.DataFrame(rows)


def views(ev):
    out = {}
    rg, Y = C.epochs_for(ev, "art", baseline=False)
    pre = (rg >= PRE[0]) & (rg < PRE[1])
    yp = np.nanmedian(Y[:, pre], 1, keepdims=True)
    out["art_rel"] = (rg, 100 * (Y - yp) / (1 + yp))
    out["art_abs"] = (rg, 100 * Y)
    rg, F = C.epochs_for(ev, "ca", baseline=False)
    fp = np.nanmedian(F[:, (rg >= PRE[0]) & (rg < PRE[1])], 1, keepdims=True)
    out["ca_rel"] = (rg, 100 * (F - fp) / (1 + fp))
    rg, P = C.epochs_for(ev, "pupil", baseline=False)
    out["pupil"] = (rg, P)
    return out


ROWS = [("pupil", "pupil (z)"), ("ca_rel", "Ca²⁺ ΔF/F_pre (%)"),
        ("art_rel", "arteriole ΔD/D_pre (%)"), ("art_abs", "arteriole vs cycle median (%)")]


def wmean(rg, M, lo, hi):
    m = (rg >= lo) & (rg <= hi)
    with np.errstate(all="ignore"):
        return np.nanmean(M[:, m], 1)


def main():
    C.set_label_mode("manual")
    OUT.mkdir(parents=True, exist_ok=True)
    keys = C.keys_for(list(GROUPS))
    r2w = r2w_events(keys)
    days = sorted(r2w["mouse"].unique())
    n2w = C.response_events(keys)
    n2w = n2w[(n2w["type"] == "N2W") & n2w["mouse"].isin(days)]
    ev = pd.concat([n2w, r2w], ignore_index=True)[["type", "key", "mouse", "group", "t0", "t0_label",
                                                   "pre_dur_s", "post_dur_s"]].reset_index(drop=True)
    V = views(ev)

    # summary per event
    rg_a, A_rel = V["art_rel"]; _, A_abs = V["art_abs"]; rg_c, F_rel = V["ca_rel"]
    ev["art_rel_0_15"] = wmean(rg_a, A_rel, 0, 15)
    ev["art_abs_pre_-20_0"] = wmean(rg_a, A_abs, -20, 0)
    ev["art_abs_post_10_40"] = wmean(rg_a, A_abs, 10, 40)
    ev["ca_rel_0_15"] = wmean(rg_c, F_rel, 0, 15)
    ev["day"] = ev["mouse"].str.split("_", n=1).str[1]
    ev.to_csv(OUT / "events.csv", index=False)
    S = (ev.groupby(["day", "type"])
         .agg(n=("t0", "size"), art_rel_0_15=("art_rel_0_15", "mean"),
              art_level_before=("art_abs_pre_-20_0", "mean"), art_level_after=("art_abs_post_10_40", "mean"),
              ca_rel_0_15=("ca_rel_0_15", "mean"))
         .reset_index())
    S.to_csv(OUT / "per_day.csv", index=False)

    # figure: rows = measure, cols = day
    fig, axs = plt.subplots(len(ROWS), len(days), figsize=(4.2 * len(days), 2.6 * len(ROWS)),
                            sharex=True, squeeze=False, constrained_layout=True)
    for j, d in enumerate(days):
        for i, (v, lab) in enumerate(ROWS):
            ax = axs[i, j]
            rg, M = V[v]
            m = (rg >= VIEW[0]) & (rg <= VIEW[1])
            for typ in ("N2W", "R2W"):
                sel = ((ev["mouse"] == d) & (ev["type"] == typ)).to_numpy()
                X = M[sel][:, m]
                X = X[np.isfinite(X).any(1)]
                if not len(X):
                    continue
                with np.errstate(all="ignore"):
                    mu = np.nanmean(X, 0); se = np.nanstd(X, 0) / np.sqrt(np.isfinite(X).sum(0))
                ax.plot(rg[m], mu, color=COL[typ], lw=1.6, label=f"{NAME[typ]} (n={len(X)})")
                if len(X) > 1:
                    ax.fill_between(rg[m], mu - se, mu + se, color=COL[typ], alpha=0.2, lw=0)
            if v != "art_abs":
                ax.axvspan(*PRE, color="0.92", zorder=0)
            ax.axvspan(*C.RESP_WIN, color="#FFF3E0", zorder=0)
            ax.axhline(0, color="0.5", lw=0.6); ax.axvline(0, color="k", lw=0.6)
            if j == 0:
                ax.set_ylabel(lab, fontsize=9)
            if i == 0:
                g = ev.loc[ev.mouse == d, "group"].iloc[0]
                ax.set_title(f"{C.GROUP_LABEL[g]}  {d.split('_')[-1]}", fontsize=10)
                ax.legend(fontsize=7, frameon=False, loc="upper left")
            if i == len(ROWS) - 1:
                ax.set_xlabel("time from pupil onset (s)")
    fig.suptitle("REM → wake vs NREM → wake on the days that have both (final labels 2026-10-09). "
                 "Bottom row: arteriole level relative to the cycle median, no baseline subtraction",
                 fontsize=11, x=0.01, ha="left")
    fig.savefig(OUT / "rem_vs_nrem_wake_per_day.png", dpi=140)
    plt.close(fig)

    # pooled WT figure (the two GCaMP8m days), mean of day means
    wt_days = [d for d in days if ev.loc[ev.mouse == d, "group"].iloc[0] == "WT"]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.0), constrained_layout=True)
    for ax, (v, lab) in zip(axs, ROWS):
        rg, M = V[v]
        m = (rg >= VIEW[0]) & (rg <= VIEW[1])
        for typ in ("N2W", "R2W"):
            sel = (ev["type"] == typ) & ev["mouse"].isin(wt_days)
            X = M[sel.to_numpy()][:, m]
            X = X[np.isfinite(X).any(1)]
            with np.errstate(all="ignore"):
                mu = np.nanmean(X, 0); se = np.nanstd(X, 0) / np.sqrt(np.isfinite(X).sum(0))
            ax.plot(rg[m], mu, color=COL[typ], lw=2, label=f"{NAME[typ]} ({len(X)} events)")
            ax.fill_between(rg[m], mu - se, mu + se, color=COL[typ], alpha=0.2, lw=0)
        if v != "art_abs":
            ax.axvspan(*PRE, color="0.92", zorder=0)
        ax.axvspan(*C.RESP_WIN, color="#FFF3E0", zorder=0)
        ax.axhline(0, color="0.5", lw=0.6); ax.axvline(0, color="k", lw=0.6)
        ax.set_title(lab, fontsize=10); ax.set_xlabel("time from pupil onset (s)")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"WT GCaMP8m ({', '.join(d.split('_')[-1] for d in wt_days)}): REM → wake vs NREM → wake, "
                 "mean ± SEM across events (only 2 days have REM → wake)", fontsize=11, x=0.01, ha="left")
    fig.savefig(OUT / "rem_vs_nrem_wake_WT_pooled.png", dpi=140)
    plt.close(fig)

    pd.set_option("display.width", 220)
    print(S.round(2).to_string(index=False))
    print("R2W anchored to pupil:", int(r2w["anchored"].sum()), "/", len(r2w))
    print("->", OUT)


if __name__ == "__main__":
    main()
