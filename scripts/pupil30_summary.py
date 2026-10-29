"""Florence's request (2026-10-19): wake-up curves with a 30% pupil-dilation criterion,
RSC 2023 vs GCaMP8m, and whether REM duration affects the curve.

Criterion: raw pupil diameter (0.5-s median filtered) rises by >= 30% at wake-up:
    (max over 0..15 s - median over -20..0 s) / median over -20..0 s >= 0.30
Events: WT NREM -> wake (both cohorts) and REM -> wake (GCaMP8m only; RSC 2023 has no
REM), final labels, t0 = pupil onset, noisy cycles excluded (same as dilation_deepdive).

Figures
  1_curves      pupil (% change), Ca2+ dF/F0, arteriole dD/D0; one column per group;
                mean +- SEM across recording days; passing events only.
  2_rem_duration REM -> wake split at the median REM bout length (short vs long), and
                REM length vs 0-15 s response for each event.
  3_gcamp_with_without_rem  GCaMP8m: all wake-ups (NREM + REM) vs NREM -> wake only.

Usage:
    PRELIM_LABEL_SET=relabel_2026-10-09 venv/bin/python -B scripts/pupil30_summary.py
Output: data/RESULTS_pupil30_summary_2026-10-19/
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
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "figs_prelim"))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import deck_part1_figures as P1  # noqa: E402
import dilation_deepdive as DD  # noqa: E402
from rem_vs_nrem_wake import r2w_events  # noqa: E402

OUT = C.ROOT / "data" / "RESULTS_pupil30_summary_2026-10-19"
FIG = OUT / "figures"
THR = 30.0
RG = P1.RG
RESP = (0.0, 15.0)
GROUPS = [("RSC2023", "N2W", "RSC 2023\nNREM → wake", "#424242"),
          ("GCaMP8m", "N2W", "GCaMP8m\nNREM → wake", "#2E7D32"),
          ("GCaMP8m", "R2W", "GCaMP8m\nREM → wake", "#6A1B9A")]
GROUPS_COL = [((c, t), col) for c, t, _, col in GROUPS]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

_RAW = {}


def raw_pupil(key):
    if key not in _RAW:
        t, y = C.read_trace(key, "pupil")
        y = y.astype(float)
        y[y <= 0] = np.nan
        k = max(1, int(round(0.5 / np.nanmedian(np.diff(t))))) | 1
        yf = median_filter(np.where(np.isfinite(y), y, np.nanmedian(y)), size=k, mode="nearest")
        yf[~np.isfinite(y)] = np.nan
        _RAW[key] = (t, yf)
    return _RAW[key]


def pupil_pct(ev):
    """Per-event pupil % change curve on RG and the criterion value."""
    M = np.full((len(ev), len(RG)), np.nan)
    rise = np.full(len(ev), np.nan)
    for i, r in enumerate(ev.itertuples()):
        t, y = raw_pupil(r.key)
        b = np.nanmedian(y[(t >= r.t0 - 20) & (t < r.t0)])
        if not np.isfinite(b) or b <= 0:
            continue
        v = DD.interp_on(t, y, r.t0 + RG, max_gap=1.0)
        M[i] = 100 * (v - b) / b
        rise[i] = 100 * (np.nanmax(y[(t >= r.t0) & (t <= r.t0 + 15)]) - b) / b
    return M, rise


def load():
    C.set_label_mode("manual")
    keys = DD.kept_keys()
    ev, _ = DD.load_events(keys)
    r2w = r2w_events(keys)[["key", "t0", "pre_dur_s"]].rename(columns={"pre_dur_s": "rem_dur_s"})
    ev = ev.merge(r2w, on=["key", "t0"], how="left")
    pup, rise = pupil_pct(ev)
    ev["pupil_rise_pct"] = rise
    ev["passes"] = ev.pupil_rise_pct >= THR
    mats = {"pupil": pup, "ca": P1.epoch_matrix(ev, "ca"), "art": P1.epoch_matrix(ev, "art")}
    m = (RG >= RESP[0]) & (RG <= RESP[1])
    with np.errstate(all="ignore"):
        for s in ("ca", "art"):
            ev[f"{s}_peak"] = np.nanmax(mats[s][:, m], 1)
            ev[f"{s}_mean"] = np.nanmean(mats[s][:, m], 1)
    return ev, mats


def day_curves(ev, M, idx):
    d = ev.loc[idx]
    D = []
    for _, g in d.groupby("day"):
        with np.errstate(all="ignore"):
            D.append(np.nanmean(M[g.index.to_numpy()], 0))
    D = np.array(D)
    with np.errstate(all="ignore"):
        mu = np.nanmean(D, 0)
        se = np.nanstd(D, 0, ddof=1) / np.sqrt(len(D)) if len(D) > 1 else np.full_like(mu, np.nan)
    return mu, se, len(D)


def summary_table(ev):
    rows = []
    for coh, typ, lab, _ in GROUPS:
        a = ev[(ev.cohort == coh) & (ev.type == typ)]
        p = a[a.passes]
        dm = p.groupby("day")[["ca_peak", "ca_mean", "art_peak", "art_mean", "pupil_rise_pct"]].mean()
        rows.append(dict(group=lab.replace("\n", " "), events_total=len(a), events_pass=len(p), days=p.day.nunique(),
                         pupil_rise_median_pct=p.pupil_rise_pct.median(),
                         ca_peak=dm.ca_peak.mean(), ca_mean_0_15=dm.ca_mean.mean(),
                         art_peak=dm.art_peak.mean(), art_mean_0_15=dm.art_mean.mean(),
                         art_mean_all_events=a.groupby("day").art_mean.mean().mean()))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "summary_by_group.csv", index=False)
    return S


def fig_curves(ev, mats, S):
    rows = [("pupil", "pupil (% change)"), ("ca", "Ca²⁺ ΔF/F0 (%)"), ("art", "arteriole ΔD/D0 (%)")]
    fig, axs = plt.subplots(3, 3, figsize=(13, 9), sharex=True, constrained_layout=True)
    for j, (coh, typ, lab, col) in enumerate(GROUPS):
        idx = ev.index[(ev.cohort == coh) & (ev.type == typ) & ev.passes]
        for i, (s, ylab) in enumerate(rows):
            ax = axs[i, j]
            mu, se, nd = day_curves(ev, mats[s], idx)
            ax.plot(RG, mu, color=col, lw=2)
            ax.fill_between(RG, mu - se, mu + se, color=col, alpha=0.25, lw=0)
            ax.axvline(0, color="k", lw=0.8)
            ax.axhline(0, color="0.7", lw=0.8)
            if i == 0:
                ax.set_title(f"{lab}\n{len(idx)} of {int(S.events_total[j])} events pass, {nd} days",
                             fontsize=11, color=col, weight="bold")
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 2:
                ax.set_xlabel("time from pupil onset (s)")
    for i in (1, 2):                                   # Ca and arteriole on one scale per row
        lo = min(a.get_ylim()[0] for a in axs[i]); hi = max(a.get_ylim()[1] for a in axs[i])
        for a in axs[i]:
            a.set_ylim(lo, hi)
    fig.suptitle(f"Wake-ups with pupil dilation ≥ {THR:.0f}% (mean ± SEM across days). "
                 "Pupil y-axis differs per column.", fontsize=11, x=0.01, ha="left")
    fig.savefig(FIG / "1_curves.png", dpi=160)
    plt.close(fig)


def fig_rem(ev, mats):
    r = ev[(ev.type == "R2W") & ev.passes].dropna(subset=["rem_dur_s"]).copy()
    med = r.rem_dur_s.median()
    r["long"] = r.rem_dur_s > med
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    res = {"median_rem_s": med, "n": len(r)}
    for j, s, ylab in ((0, "pupil", "pupil (% change)"), (1, "ca", "Ca²⁺ ΔF/F0 (%)"), (2, "art", "arteriole ΔD/D0 (%)")):
        ax = axs[0, j]
        for lg, col, nm in ((False, "#CE93D8", "short"), (True, "#4A148C", "long")):
            g = r[r.long == lg]
            M = mats[s][g.index.to_numpy()]
            with np.errstate(all="ignore"):
                mu = np.nanmean(M, 0)
                se = np.nanstd(M, 0, ddof=1) / np.sqrt(len(g))
            rng = f"{g.rem_dur_s.min():.0f}–{g.rem_dur_s.max():.0f} s"
            ax.plot(RG, mu, color=col, lw=2, label=f"{nm} REM ({rng}, {len(g)} events)")
            ax.fill_between(RG, mu - se, mu + se, color=col, alpha=0.2, lw=0)
        ax.axvline(0, color="k", lw=0.8); ax.axhline(0, color="0.7", lw=0.8)
        ax.set_ylabel(ylab); ax.set_xlabel("time from pupil onset (s)")
        if j == 0:
            ax.legend(frameon=False, fontsize=8.5)
    for j, (y, ylab) in enumerate((("pupil_rise_pct", "pupil rise (%)"), ("ca_mean", "Ca²⁺ mean 0–15 s (%)"),
                                   ("art_mean", "arteriole mean 0–15 s (%)"))):
        ax = axs[1, j]
        for day, mk in zip(sorted(r.day.unique()), ("o", "s")):
            g = r[r.day == day]
            ax.scatter(g.rem_dur_s, g[y], marker=mk, s=40, color="#6A1B9A", alpha=0.8, label=day)
        rk = r.groupby("day")[["rem_dur_s", y]].rank(pct=True)
        rho, p = stats.spearmanr(rk.rem_dur_s, rk[y])
        rho_raw, p_raw = stats.spearmanr(r.rem_dur_s, r[y])
        res[f"{y}_rho_within_day"], res[f"{y}_p_within_day"] = rho, p
        res[f"{y}_rho_pooled"], res[f"{y}_p_pooled"] = rho_raw, p_raw
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_xlabel("REM bout length before waking (s)"); ax.set_ylabel(ylab)
        ax.set_title(f"within-day ρ = {rho:+.2f} (p = {p:.2f}); pooled ρ = {rho_raw:+.2f}", fontsize=10)
        if j == 0:
            ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle(f"GCaMP8m REM → wake (pupil ≥ {THR:.0f}%): split at median REM length {med:.0f} s. "
                 "Top: mean ± SEM across events. Events as units; 2 days only.", fontsize=11, x=0.01, ha="left")
    fig.savefig(FIG / "2_rem_duration.png", dpi=160)
    plt.close(fig)
    r[["day", "cycle", "rem_dur_s", "pupil_rise_pct", "ca_mean", "ca_peak", "art_mean", "art_peak"]].to_csv(
        OUT / "rem_to_wake_events.csv", index=False)
    return res


def fig_gcamp_mix(ev, mats):
    g = ev[(ev.cohort == "GCaMP8m") & ev.passes]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    out = {}
    for ax, s, ylab in ((axs[0], "ca", "Ca²⁺ ΔF/F0 (%)"), (axs[1], "art", "arteriole ΔD/D0 (%)")):
        for sel, col, nm in ((g.index, "#6A1B9A", "all wake-ups (NREM + REM)"),
                             (g.index[g.type == "N2W"], "#2E7D32", "NREM → wake only")):
            mu, se, nd = day_curves(ev, mats[s], sel)
            ax.plot(RG, mu, color=col, lw=2, label=f"{nm}: {len(sel)} events, {nd} days")
            ax.fill_between(RG, mu - se, mu + se, color=col, alpha=0.2, lw=0)
            m = (RG >= RESP[0]) & (RG <= RESP[1])
            out[f"{s}_{'all' if len(sel) == len(g) else 'n2w'}_mean_0_15"] = float(np.nanmean(mu[m]))
        ax.axvline(0, color="k", lw=0.8); ax.axhline(0, color="0.7", lw=0.8)
        ax.set_ylabel(ylab); ax.set_xlabel("time from pupil onset (s)")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle(f"GCaMP8m, pupil ≥ {THR:.0f}%: including REM → wake events vs NREM → wake only "
                 "(mean ± SEM across days)", fontsize=11, x=0.01, ha="left")
    fig.savefig(FIG / "3_gcamp_with_without_rem.png", dpi=160)
    plt.close(fig)
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    ev, mats = load()
    ev.to_csv(OUT / "events.csv", index=False)
    S = summary_table(ev)
    fig_curves(ev, mats, S)
    R = fig_rem(ev, mats)
    M = fig_gcamp_mix(ev, mats)
    pd.set_option("display.width", 220)
    print(S.round(1).to_string(index=False))
    print("REM duration:", {k: round(float(v), 3) for k, v in R.items()})
    print("GCaMP8m mix:", {k: round(v, 2) for k, v in M.items()})
    print("failing events:", ev[~ev.passes].groupby(["cohort", "type", "day"]).size().to_dict())
    print("->", OUT)


if __name__ == "__main__":
    main()
