"""Figures for 'Findings part 1' (2026-10-09): WT wake-up heterogeneity.

Same normalisation everywhere, per event, baseline = median of -20..0 s before
pupil onset:
  arteriole  ΔD/D0 = (D - D0)/D0,  D = sqrt(cross-sectional area)
  Ca2+       ΔF/F0 = (F - F0)/F0
  pupil, eye z-scored within the cycle (robust z), minus the -20..0 s median
No cycle-median normalisation of the response.

Events: final labels (run with PRELIM_LABEL_SET=relabel_2026-10-09), manual only, WT.
  NREM->wake  common.response_events type N2W
  REM->wake   scripts/rem_vs_nrem_wake.r2w_events (REM >= 20 s then Awake >= 20 s)
Eye movement: DLC 2-D eye speed where it exists (GCaMP8m 2026.03.03 / 03.06),
otherwise Florence's pixel-based 1-D speed (2023 RSC days, 2025.12.18); none for
2023.07.12. Both robust-z'd within cycle, so only ranks within a day are compared.

Output: data/RESULTS_deck_part1_2026-10-09/figures/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "figs_prelim"))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from rem_vs_nrem_wake import r2w_events  # noqa: E402

OUT = C.ROOT / "data" / "RESULTS_deck_part1_2026-10-09"
FIG = OUT / "figures"
PRE = (-20.0, 0.0)
VIEW = (-30.0, 60.0)
FS = 2.0                                     # epoch grid for every signal (Hz)
RG = np.arange(VIEW[0], VIEW[1] + 1e-9, 1 / FS)
PIX_EYE = C.ROOT / "data" / "pupil_eye_motion_pixel_control_2026-10-06"
COH = {"RSC2023": "WT RSC 2023", "GCaMP8m": "WT GCaMP8m 2025–26"}
COH_COL = {"RSC2023": "#424242", "GCaMP8m": "#2E7D32"}
TYPE_COL = {"N2W": "#EF6C00", "R2W": "#6A1B9A"}
TYPE_NAME = {"N2W": "NREM → wake", "R2W": "REM → wake"}
SIGS = ("pupil", "eye", "ca", "art")
SIG_LAB = {"pupil": "pupil (z)", "eye": "eye movement (z)", "ca": "Ca²⁺ ΔF/F0 (%)", "art": "arteriole ΔD/D0 (%)"}
TERT = ("low", "mid", "high")
TERT_COL = ("#90CAF9", "#1E88E5", "#0D47A1")


# ----------------------------------------------------------------- signals
def _bin(t, y, fs=FS):
    """Bin-average onto a regular grid (keeps spikes in speed traces)."""
    edges = np.arange(0, t[-1] + 1 / fs, 1 / fs)
    idx = np.digitize(t, edges) - 1
    s = pd.Series(y).groupby(idx).mean()
    grid = edges[:-1] + 0.5 / fs
    out = np.full(len(grid), np.nan)
    k = s.index.to_numpy(); k = k[(k >= 0) & (k < len(grid))]
    out[k] = s.loc[k].to_numpy()
    return grid, out


def eye_signal(key):
    """(t, raw eye speed per frame, source) or None."""
    for cand in (key, key.replace("Extra_", "")):
        p = C.EYE_DIR / f"{cand}_eyemotion_perframe.csv"
        if p.exists():
            d = pd.read_csv(p)
            y = d["eye_speed"].to_numpy(float)
            if "include" in d:
                y[~d["include"].astype(bool).to_numpy()] = np.nan
            return d["time"].to_numpy(float), y, "DLC"
    p = PIX_EYE / f"{key}_eyemotion_perframe.csv"
    if p.exists():
        d = pd.read_csv(p)
        return d["time"].to_numpy(float), d["speed_px_per_frame"].to_numpy(float), "pixel"
    return None


_cache = {}


def signals(key):
    if key in _cache:
        return _cache[key]
    cyc = C.load_cycle(key)
    out = {}
    if "pupil" in cyc.sig:
        out["pupil"] = cyc.sig["pupil"]                         # robust z, 10 Hz
    e = eye_signal(key)
    if e is not None:
        # bin first (0.5 s), then robust z: duplicated video frames make raw
        # frame-to-frame speed mostly 0, which collapses a per-frame MAD
        g, v = _bin(e[0], e[1])
        out["eye"] = (g, C._robust_z(v))
        out["eye_src"] = e[2]
    for s in ("ca", "art"):
        if s in cyc.sig:
            t, y = cyc.sig[s]
            out[s] = (t, 1 + y)                                 # back to X / X_median (ratio cancels)
    _cache[key] = out
    return out


def epoch_matrix(ev, sig):
    M = np.full((len(ev), len(RG)), np.nan)
    pre = (RG >= PRE[0]) & (RG < PRE[1])
    for i, r in enumerate(ev.itertuples()):
        s = signals(r.key)
        if sig not in s:
            continue
        t, y = s[sig]
        fin = np.isfinite(y)
        if fin.sum() < 10:
            continue
        tt = r.t0 + RG
        v = np.interp(tt, t[fin], y[fin], left=np.nan, right=np.nan)
        j = np.clip(np.searchsorted(t[fin], tt), 1, fin.sum() - 1)
        tf = t[fin]
        v[np.minimum(np.abs(tf[j] - tt), np.abs(tf[j - 1] - tt)) > 1.5] = np.nan
        if not np.isfinite(v[pre]).sum():
            continue
        b = np.nanmedian(v[pre])
        M[i] = 100 * (v - b) / b if sig in ("ca", "art") else v - b
    return M


def wmean(M, lo, hi, how="mean"):
    m = (RG >= lo) & (RG <= hi)
    with np.errstate(all="ignore"):
        return (np.nanmax if how == "max" else np.nanmean)(M[:, m], 1)


def msem(X):
    X = X[np.isfinite(X).any(1)]
    with np.errstate(all="ignore"):
        return np.nanmean(X, 0), np.nanstd(X, 0) / np.sqrt(np.isfinite(X).sum(0)), len(X)


def shade(ax, base=True):
    if base:
        ax.axvspan(*PRE, color="0.92", zorder=0)
    ax.axvspan(*C.RESP_WIN, color="#FFF3E0", zorder=0)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.axvline(0, color="k", lw=0.7)


# ----------------------------------------------------------------- events
def load():
    keys = C.keys_for(["WT"])
    n = C.response_events(keys)
    n = n[n["type"] == "N2W"][["type", "key", "mouse", "t0"]]
    r = r2w_events(keys)[["type", "key", "mouse", "t0"]]
    ev = pd.concat([n, r], ignore_index=True)
    ev["cohort"] = np.where(ev["key"].str.startswith("Extra_GCaMP8m"), "GCaMP8m", "RSC2023")
    ev["day"] = ev["mouse"].str.split("_").str[-1]
    E = {s: epoch_matrix(ev, s) for s in SIGS}
    ev["pupil_rise"] = wmean(E["pupil"], 0, 15, "max")
    ev["eye_0_15"] = wmean(E["eye"], 0, 15)
    ev["ca_0_15"] = wmean(E["ca"], 0, 15)
    ev["art_0_15"] = wmean(E["art"], 0, 15)
    ev["eye_src"] = [signals(k).get("eye_src", "") for k in ev["key"]]
    return ev, E


# ----------------------------------------------------------------- figures
def fig1_heatmap(ev, E):
    sel = np.flatnonzero((ev["type"] == "N2W").to_numpy())
    blocks = []
    for coh in ("RSC2023", "GCaMP8m"):
        b = sel[(ev["cohort"].to_numpy()[sel] == coh)]
        blocks.append((coh, b[np.argsort(-ev["pupil_rise"].to_numpy()[b])]))
    order = np.concatenate([b for _, b in blocks])
    n0 = len(blocks[0][1])
    days = sorted(ev.loc[order, "day"].unique())
    dcol = {d: plt.get_cmap("tab10")(i) for i, d in enumerate(days)}
    lims = {"pupil": 2.5, "eye": 4.0, "ca": None, "art": 10.0}
    fig, axs = plt.subplots(1, 5, figsize=(16, 8.6), gridspec_kw=dict(width_ratios=[0.22, 1, 1, 1, 1]),
                            constrained_layout=True)
    axs[0].imshow(np.array([[dcol[d] for d in ev.loc[order, "day"]]]).transpose(1, 0, 2), aspect="auto",
                  interpolation="nearest")
    axs[0].set_xticks([]); axs[0].set_yticks([])
    axs[0].set_title("day", fontsize=9)
    for ax, s in zip(axs[1:], SIGS):
        M = E[s][order]
        if s == "art":
            M = pd.DataFrame(M.T).rolling(3, center=True, min_periods=1).mean().to_numpy().T
        lim = lims[s] or np.nanpercentile(np.abs(M), 96)
        cmap = "PuOr_r" if s in ("pupil", "eye") else "RdBu_r"
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest",
                       extent=[VIEW[0], VIEW[1], len(M) - 0.5, -0.5])
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(n0 - 0.5, color="k", lw=2)
        ax.set_yticks([])
        ax.set_xlabel("time from pupil onset (s)")
        n_has = int(np.isfinite(M).any(1).sum())
        ax.set_title(f"{SIG_LAB[s]}  (n={n_has})", fontsize=10)
        fig.colorbar(im, ax=ax, location="bottom", shrink=0.7, pad=0.01)
    for coh, b in blocks:
        y = n0 / 2 if coh == "RSC2023" else n0 + len(b) / 2
        axs[1].text(VIEW[0] - 3, y, f"{COH[coh]}\n({len(b)} events)", rotation=90, ha="right", va="center",
                    fontsize=10, color=COH_COL[coh], weight="bold")
    h = [plt.Line2D([], [], color=c, lw=6, label=d) for d, c in dcol.items()]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=8, fontsize=8, frameon=False)
    fig.savefig(FIG / "1_heatmap_wt_n2w.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_per_day(ev, E):
    days = (ev[["day", "cohort"]].drop_duplicates()
            .assign(c=lambda d: d.cohort.map({"RSC2023": 0, "GCaMP8m": 1})).sort_values(["c", "day"]))
    fig, axs = plt.subplots(2, len(days), figsize=(2.6 * len(days), 5.6), sharex=True, sharey="row",
                            constrained_layout=True)
    m = np.ones(len(RG), bool)
    for j, (d, coh) in enumerate(zip(days.day, days.cohort)):
        for i, s in enumerate(("ca", "art")):
            ax = axs[i, j]
            for typ in ("N2W", "R2W"):
                sel = ((ev.day == d) & (ev.type == typ)).to_numpy()
                if not sel.any():
                    continue
                X = E[s][sel]
                if s == "art":
                    X = pd.DataFrame(X.T).rolling(3, center=True, min_periods=1).mean().to_numpy().T
                mu, se, n = msem(X)
                if not n:
                    continue
                ax.plot(RG[m], mu, color=TYPE_COL[typ], lw=1.5, label=f"{TYPE_NAME[typ]} ({n})")
                if n > 1:
                    ax.fill_between(RG, mu - se, mu + se, color=TYPE_COL[typ], alpha=0.2, lw=0)
            shade(ax)
            if i == 0:
                ax.set_title(d, fontsize=10, color=COH_COL[coh], weight="bold")
                ax.legend(fontsize=6.5, frameon=False, loc="upper right")
            if j == 0:
                ax.set_ylabel(SIG_LAB[s])
            if i == 1:
                ax.set_xlabel("time (s)")
                ax.set_ylim(-20, 12)
    fig.suptitle("Each WT day: Ca²⁺ and arteriole after waking from NREM (orange) and REM (purple). "
                 "Grey = RSC 2023, green = GCaMP8m 2025–26. Mean ± SEM across events", fontsize=10.5, x=0.01, ha="left")
    fig.savefig(FIG / "2_per_day_n2w_r2w.png", dpi=150)
    plt.close(fig)


def fig3_cohorts(ev):
    E2 = ev[ev.type == "N2W"]
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for ax, s in zip(axs, ("art", "ca")):
        Ms = {}
        for coh in ("RSC2023", "GCaMP8m"):
            dm = []
            for d, g in E2[E2.cohort == coh].groupby("day"):
                X = EV[s][g.index.to_numpy()]
                if s == "art":
                    X = pd.DataFrame(X.T).rolling(3, center=True, min_periods=1).mean().to_numpy().T
                mu, _, n = msem(X)
                if n:
                    dm.append(mu)
            Ms[coh] = np.array(dm)
            mu = Ms[coh].mean(0); se = Ms[coh].std(0, ddof=1) / np.sqrt(len(Ms[coh]))
            ax.plot(RG, mu, color=COH_COL[coh], lw=2.2,
                    label=f"{COH[coh]} ({len(Ms[coh])} days, {int((E2.cohort == coh).sum())} events)")
            ax.fill_between(RG, mu - se, mu + se, color=COH_COL[coh], alpha=0.2, lw=0)
        shade(ax)
        day_val = E2.groupby(["cohort", "day"])[f"{s}_0_15"].mean()
        a, b = day_val["RSC2023"].to_numpy(), day_val["GCaMP8m"].to_numpy()
        p = stats.ttest_ind(a, b, equal_var=False).pvalue
        rows.append(dict(signal=s, rsc_mean=a.mean(), gcamp_mean=b.mean(), rsc_vs0_p=stats.ttest_1samp(a, 0).pvalue,
                         gcamp_vs0_p=stats.ttest_1samp(b, 0).pvalue, welch_p=p))
        ax.set_title(f"{SIG_LAB[s].split(' (')[0]}: mean 0–15 s  RSC {a.mean():+.1f}%  vs  GCaMP8m {b.mean():+.1f}%"
                     f"   (Welch p = {p:.3f}, days)", fontsize=10)
        ax.set_ylabel(SIG_LAB[s]); ax.set_xlabel("time from pupil onset (s)")
        ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.suptitle("NREM → wake only (no REM → wake events). Mean ± SEM across days", fontsize=11, x=0.01, ha="left")
    fig.savefig(FIG / "3_cohorts_n2w.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "cohort_tests.csv", index=False)
    return pd.DataFrame(rows)


def fig4_arousal(ev):
    E2 = ev[ev.type == "N2W"].copy()
    res = []
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.8), sharex=True, constrained_layout=True)
    for row, (var, lab, sig0) in enumerate([("pupil_rise", "pupil rise", "pupil"),
                                            ("eye_0_15", "eye movement", "eye")]):
        d = E2[E2[var].notna()].copy()
        d["pct"] = d.groupby("day")[var].rank(pct=True)
        d["tert"] = np.digitize(d["pct"], [1 / 3, 2 / 3 + 1e-9])
        for ax, s in zip(axs[row], (sig0, "ca", "art")):
            for k in range(3):
                X = EV[s][d.index[d.tert == k].to_numpy()]
                if s == "art":
                    X = pd.DataFrame(X.T).rolling(3, center=True, min_periods=1).mean().to_numpy().T
                mu, se, n = msem(X)
                ax.plot(RG, mu, color=TERT_COL[k], lw=1.8, label=f"{TERT[k]} {lab} (n={n})")
                ax.fill_between(RG, mu - se, mu + se, color=TERT_COL[k], alpha=0.18, lw=0)
            shade(ax)
            ax.set_ylabel(SIG_LAB[s])
            if row == 1:
                ax.set_xlabel("time from pupil onset (s)")
            if s != sig0:
                v = d[["day", var, f"{s}_0_15"]].dropna()
                r = v.groupby("day")[[var, f"{s}_0_15"]].rank(pct=True)
                rho, p = stats.spearmanr(r[var], r[f"{s}_0_15"])
                res.append(dict(split=var, signal=s, n_events=len(v), n_days=v.day.nunique(), within_day_rho=rho, p=p))
                ax.set_title(f"within-day ρ = {rho:+.2f} (p = {p:.3f})", fontsize=10)
            else:
                ax.set_title(f"split by {lab} (thirds within each day; {d.day.nunique()} days)", fontsize=10)
            ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("Does a stronger wake-up give a bigger dilation? WT NREM → wake. Eye movement: DLC on 03.03/03.06, "
                 "pixel-based elsewhere, none on 2023.07.12", fontsize=10.5, x=0.01, ha="left")
    fig.savefig(FIG / "4_arousal.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(res).to_csv(OUT / "arousal_tests.csv", index=False)
    return pd.DataFrame(res)


EV = None


def main():
    global EV
    C.set_label_mode("manual")
    FIG.mkdir(parents=True, exist_ok=True)
    ev, E = load()
    EV = E
    ev.to_csv(OUT / "events.csv", index=False)
    fig1_heatmap(ev, E)
    fig2_per_day(ev, E)
    t3 = fig3_cohorts(ev)
    t4 = fig4_arousal(ev)
    pd.set_option("display.width", 200)
    print(ev.groupby(["cohort", "type"]).size().to_string())
    print(ev[ev.type == "N2W"].groupby("day").eye_src.agg(lambda s: ",".join(sorted(set(s)))).to_string())
    print(t3.round(4).to_string(index=False)); print(t4.round(3).to_string(index=False))
    print("->", FIG)


if __name__ == "__main__":
    main()
