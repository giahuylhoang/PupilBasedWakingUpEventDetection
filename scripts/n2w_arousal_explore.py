"""Look at single NREM->wake events: does the arteriole response follow arousal?

Exploration only (no hypothesis tests). Arousal = pupil rise (all events) and
eye movement (only days with DLC eye-motion data). Whisking is deliberately
NOT used (judged unreliable).

Events: common.response_events (pupil-anchored, non-overlapping), type N2W,
locked v2 label set unless PRELIM_LABEL_SET says otherwise.

Usage:
    ./venv/bin/python -B scripts/n2w_arousal_explore.py            # WT
    ./venv/bin/python -B scripts/n2w_arousal_explore.py --group SCR

Pupil-rise tertile cutoffs are fixed from WT (written on the WT run) and reused
for other groups, so "low / mid / high arousal" means the same pupil rise in
every group.

Outputs: data/RESULTS_wake_arousal_explore_2026-10-02/<group>/
    tables/events.csv               one row per event, all per-event measures
    figures/A_heatmap_by_pupil.png  events sorted by pupil rise
    figures/B_heatmap_by_arteriole.png  sorted by arteriole AUC (viewing only)
    figures/C_scatter.png           arousal measures vs arteriole response
    figures/D_tertile_averages.png  low / mid / high pupil-rise averages
    figures/gallery/page_XX.png     every single event, same order as A
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent / "figs_prelim"))
import common as C  # noqa: E402

OUT_ROOT = C.ROOT / "data" / "RESULTS_wake_arousal_explore_2026-10-02"
VIEW = (-20.0, 40.0)            # display window (s)
SIGNALS = ("pupil", "eye", "ca", "art")
SIG_NAME = {"pupil": "pupil (z)", "eye": "eye movement (z)",
            "ca": "Ca²⁺ ΔF/F₀", "art": "arteriole ΔD/D₀"}
CMAP = {"pupil": "PuOr_r", "eye": "PuOr_r", "ca": "RdBu_r", "art": "RdBu_r"}
TERT_NAME = ("low", "mid", "high")
TERT_COL = ("#90CAF9", "#1E88E5", "#0D47A1")


def _win(rg, lo, hi):
    return (rg >= lo) & (rg <= hi)


def _smooth(M, n):
    """Centred moving average along time, display only."""
    if n <= 1:
        return M
    return pd.DataFrame(M.T).rolling(n, center=True, min_periods=1).mean().to_numpy().T


def per_event_table(ev):
    """Per-event measures. Arousal measures never use the arteriole."""
    E = {s: C.epochs_for(ev, s) for s in SIGNALS}
    out = ev[["key", "mouse", "group", "t0", "t0_label", "anchored",
              "pre_dur_s", "post_dur_s"]].copy().reset_index(drop=True)

    rg, P = E["pupil"]
    out["pupil_rise"] = np.nanmax(P[:, _win(rg, 0, 15)], axis=1)
    dP = np.gradient(_smooth(P, 2), axis=1) * C.EPOCH_FS["pupil"]   # z/s, 1-s smoothing
    out["pupil_rise_speed"] = np.nanmax(dP[:, _win(rg, 0, 10)], axis=1)

    rg, Y = E["eye"]
    seg = Y[:, _win(rg, 0, 15)]
    has = np.isfinite(seg).sum(axis=1) >= 10
    out["eye_move"] = np.where(has, np.nanmean(seg, axis=1), np.nan)

    for s in ("ca", "art"):
        rg, M = E[s]
        pk, auc, tpk = C.response_metrics(rg, M)
        out[f"{s}_peak"], out[f"{s}_auc"], out[f"{s}_tpeak"] = pk, auc, tpk
        out[f"{s}_trough"] = np.nanmin(M[:, _win(rg, 0, 15)], axis=1)

    # arteriole before the event, NOT baselined: is the vessel already up/down?
    rg, A = C.epochs_for(ev, "art", baseline=False)
    pre = _win(rg, -20, -5)
    out["art_pre_level"] = np.nanmean(A[:, _win(rg, -5, 0)], axis=1)
    out["art_pre_slope"] = [np.polyfit(rg[pre][np.isfinite(r[pre])], r[pre][np.isfinite(r[pre])], 1)[0]
                            if np.isfinite(r[pre]).sum() > 5 else np.nan for r in A]
    return out, E


def tertile_cutoffs(group, df):
    p = OUT_ROOT / "pupil_rise_cutoffs_WT.json"
    if group == "WT":
        q = df["pupil_rise"].quantile([1 / 3, 2 / 3]).tolist()
        p.write_text(json.dumps({"q33": q[0], "q67": q[1], "n_events": int(df["pupil_rise"].notna().sum())},
                                indent=2))
        return q
    if not p.exists():
        sys.exit(f"Run the WT group first: {p} is needed for fixed cutoffs.")
    d = json.loads(p.read_text())
    return [d["q33"], d["q67"]]


def day_colours(days):
    cmap = plt.get_cmap("tab10")
    return {d: cmap(i % 10) for i, d in enumerate(sorted(days))}


def heatmaps(df, E, order, title, path, dcol):
    fig, axs = plt.subplots(1, 5, figsize=(17, 9), gridspec_kw=dict(width_ratios=[0.25, 1, 1, 1, 1]),
                            constrained_layout=True)
    days = df["mouse"].to_numpy()[order]
    axs[0].imshow(np.array([[dcol[d] for d in days]]).transpose(1, 0, 2), aspect="auto",
                  interpolation="nearest")
    axs[0].set_xticks([]); axs[0].set_ylabel("event (row number = gallery number)")
    axs[0].set_title("day", fontsize=9)
    for ax, s in zip(axs[1:], SIGNALS):
        rg, M = E[s]
        M = _smooth(M, 3 if s in ("ca", "art") else 1)[order]
        m = _win(rg, *VIEW)
        lim = np.nanpercentile(np.abs(M[:, m]), 97) or 1
        im = ax.imshow(M[:, m], aspect="auto", cmap=CMAP[s], vmin=-lim, vmax=lim,
                       extent=[VIEW[0], VIEW[1], len(M) - 0.5, -0.5], interpolation="nearest")
        ax.axvline(0, color="k", lw=0.8)
        ax.axvspan(0, 15, ymin=0.995, ymax=1, color="orange", clip_on=False)
        n_has = int(np.isfinite(M[:, m]).any(axis=1).sum())
        ax.set_title(f"{SIG_NAME[s]}  (n={n_has})", fontsize=10)
        ax.set_xlabel("time from pupil onset (s)")
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, location="bottom", shrink=0.7, pad=0.01)
    fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    handles = [plt.Line2D([], [], color=c, lw=6, label=d.split("_", 1)[1]) for d, c in dcol.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=8, fontsize=8,
               frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def scatter(df, path, dcol, group):
    pairs = [("pupil_rise", "pupil rise, 0–15 s (z)"),
             ("pupil_rise_speed", "pupil rise speed (z/s)"),
             ("eye_move", "eye movement, 0–15 s (z)"),
             ("ca_auc", "Ca²⁺ AUC, 0–15 s"),
             ("art_pre_slope", "arteriole trend −20…−5 s (/s)"),
             ("pre_dur_s", "preceding NREM bout (s)")]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    for ax, (x, lab) in zip(axs.flat, pairs):
        d = df[[x, "art_auc", "mouse"]].dropna()
        ax.scatter(d[x], d["art_auc"], s=18, c=[dcol[m] for m in d["mouse"]], alpha=0.8, lw=0)
        ax.axhline(0, color="0.6", lw=0.7)
        if len(d) >= 5:
            r, _ = spearmanr(d[x], d["art_auc"])
            ax.set_title(f"ρ = {r:+.2f}  (n = {len(d)} events, {d['mouse'].nunique()} days)", fontsize=9)
        ax.set_xlabel(lab); ax.set_ylabel("arteriole AUC, 0–15 s")
    fig.suptitle(f"{C.GROUP_LABEL.get(group, group)} NREM→wake: what goes with the arteriole response?  "
                 "(descriptive; events pooled, colour = day)", fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=150); plt.close(fig)


def tertile_plot(df, E, cuts, path, group):
    t = np.digitize(df["pupil_rise"].to_numpy(), cuts)
    t = np.where(df["pupil_rise"].notna(), t, -1)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.4), constrained_layout=True)
    for ax, s in zip(axs, SIGNALS):
        rg, M = E[s]
        m = _win(rg, *VIEW)
        for k in range(3):
            X = M[t == k][:, m]
            X = X[np.isfinite(X).any(axis=1)]
            if len(X) < 2:
                continue
            mu = np.nanmean(X, 0); se = np.nanstd(X, 0) / np.sqrt(np.isfinite(X).sum(0))
            ax.plot(rg[m], mu, color=TERT_COL[k], lw=1.8, label=f"{TERT_NAME[k]} (n={len(X)})")
            ax.fill_between(rg[m], mu - se, mu + se, color=TERT_COL[k], alpha=0.2, lw=0)
        ax.axvspan(*C.BASE_WIN, color="0.88", zorder=0); ax.axvspan(*C.RESP_WIN, color="#FFF3E0", zorder=0)
        ax.axvline(0, color="k", lw=0.7); ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(SIG_NAME[s]); ax.set_xlabel("time from pupil onset (s)")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{C.GROUP_LABEL.get(group, group)}: events split by pupil rise "
                 f"(cutoffs from WT: {cuts[0]:.2f}, {cuts[1]:.2f} z); mean ± SEM across events",
                 fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=150); plt.close(fig)


def gallery(df, E, order, gdir, dcol, per_page=12):
    gdir.mkdir(parents=True, exist_ok=True)
    for f in gdir.glob("page_*.png"):
        f.unlink()
    for p0 in range(0, len(order), per_page):
        idx = order[p0:p0 + per_page]
        fig = plt.figure(figsize=(18, 11))
        outer = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.25)
        for j, i in enumerate(idx):
            inner = outer[j].subgridspec(3, 1, hspace=0.05)
            r = df.iloc[i]
            for k, (sigs, lab) in enumerate([(("pupil", "eye"), "pupil / eye"), (("ca",), "Ca²⁺"),
                                             (("art",), "arteriole")]):
                ax = fig.add_subplot(inner[k])
                for s in sigs:
                    rg, M = E[s]
                    m = _win(rg, *VIEW)
                    col = {"pupil": "#5D4037", "eye": "#00897B", "ca": "#2E7D32", "art": "#C62828"}[s]
                    ax.plot(rg[m], M[i, m], color=col, lw=1)
                ax.axvspan(0, 15, color="#FFF3E0", zorder=0); ax.axvline(0, color="k", lw=0.6)
                ax.axhline(0, color="0.7", lw=0.5)
                ax.set_xlim(*VIEW); ax.tick_params(labelsize=6)
                ax.set_ylabel(lab, fontsize=7)
                if k < 2:
                    ax.set_xticklabels([])
                if k == 0:
                    ax.set_title(f"#{p0 + j + 1}  {r['mouse'].split('_', 1)[1]} c{r['key'].rsplit('cycle', 1)[1]}"
                                 f"  t={r['t0']:.0f}s\npupil rise {r['pupil_rise']:.2f}  "
                                 f"art AUC {r['art_auc']:+.2f}", fontsize=7, color=dcol[r["mouse"]])
        fig.text(0.01, 0.005, "brown = pupil, teal = eye movement, green = Ca²⁺, red = arteriole; "
                 "baseline −5…0 s subtracted; orange = 0–15 s response window", fontsize=8)
        fig.savefig(gdir / f"page_{p0 // per_page + 1:02d}.png", dpi=110)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="WT", choices=sorted(set(C.COHORT_GROUP.values())))
    a = ap.parse_args()
    C.set_label_mode("manual")          # manual labels only for this exploration
    out = OUT_ROOT / a.group
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    ev = C.response_events(C.keys_for([a.group]))
    ev = ev[ev["type"] == "N2W"].reset_index(drop=True)
    df, E = per_event_table(ev)
    cuts = tertile_cutoffs(a.group, df)
    df["pupil_tertile"] = pd.Series(np.digitize(df["pupil_rise"], cuts)).map(dict(enumerate(TERT_NAME)))
    df.loc[df["pupil_rise"].isna(), "pupil_tertile"] = np.nan

    order = np.argsort(-df["pupil_rise"].fillna(-np.inf).to_numpy(), kind="stable")
    df["gallery_no"] = 0
    df.loc[order, "gallery_no"] = np.arange(1, len(df) + 1)
    df.sort_values("gallery_no").to_csv(out / "tables" / "events.csv", index=False)

    dcol = day_colours(df["mouse"].unique())
    lab = C.GROUP_LABEL.get(a.group, a.group)
    heatmaps(df, E, order, f"{lab} NREM→wake events ({len(df)}), sorted by pupil rise (largest at top)",
             out / "figures" / "A_heatmap_by_pupil.png", dcol)
    o2 = np.argsort(-df["art_auc"].fillna(-np.inf).to_numpy(), kind="stable")
    heatmaps(df, E, o2, f"{lab} NREM→wake events sorted by arteriole AUC (viewing only: sorted by the outcome)",
             out / "figures" / "B_heatmap_by_arteriole.png", dcol)
    scatter(df, out / "figures" / "C_scatter.png", dcol, a.group)
    tertile_plot(df, E, cuts, out / "figures" / "D_tertile_averages.png", a.group)
    gallery(df, E, order, out / "figures" / "gallery", dcol)

    print(f"{a.group}: {len(df)} events, {df['mouse'].nunique()} days, "
          f"eye data {df['eye_move'].notna().sum()}; cutoffs {cuts[0]:.2f}/{cuts[1]:.2f}")
    print(f"arteriole AUC>0: {(df['art_auc'] > 0).sum()}, <0: {(df['art_auc'] < 0).sum()}")
    print(f"Ca AUC>0: {(df['ca_auc'] > 0).sum()}, <0: {(df['ca_auc'] < 0).sum()}")
    print(df.groupby("pupil_tertile")[["pupil_rise", "art_auc", "ca_auc"]].median().round(3))
    print("-> ", out)


if __name__ == "__main__":
    main()
