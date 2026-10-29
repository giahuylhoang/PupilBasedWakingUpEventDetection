"""GRAB-NE with final manual labels (2026-10-21), for Florence's question
"what have you seen in the noradrenaline data, and should pupil be considered with NE?"

Data: Extra_NE, 13 cycles / 3 days (2025.09.09, 2025.09.11, 2026.03.10), labels from
data/labels_bucket_2026-10-09_1022 via C.set_label_mode("manual"). No Ca²⁺ in this cohort.

  1_state_levels   NE, pupil and arteriole level by state (vs cycle median), per day
  2_event_curves   NE / pupil / arteriole around NREM->wake, micro-arousal (MA, 5-15 s
                   Awake inside NREM), REM->wake, wake->NREM. t0 = pupil onset for
                   N2W / MA / R2W, label boundary for W2N. Mean +- SEM across days.
                   Pupil = robust z minus its -20..0 s median; NE dF/F0 and arteriole
                   dD/D0 use the -20..0 s median as F0 / D0.
  3_ne_vs_pupil    within each state, how closely NE and pupil track each other:
                   per-cycle Pearson r of 5-s bin means, and the NE-pupil
                   cross-correlation lag.
  4_ma_pupil       micro-arousals: does every NE surge come with a pupil dilation?
                   per-MA NE rise vs pupil rise (0-10 s).

Usage:
    PRELIM_LABEL_SET=relabel_2026-10-09 venv/bin/python -B scripts/ne_labelled_summary.py
Output: data/RESULTS_ne_labelled_2026-10-21/
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
import dilation_deepdive as DD  # noqa: E402
from rem_vs_nrem_wake import r2w_events  # noqa: E402

OUT = C.ROOT / "data" / "RESULTS_ne_labelled_2026-10-21"
FIG = OUT / "figures"
RG = np.arange(-30.0, 60.0 + 1e-9, 0.5)
PRE = (-20.0, 0.0)
STATES = ("Awake", "NREM", "REM")
ST_COL = {"Awake": "#F9A825", "NREM": "#1565C0", "REM": "#6A1B9A"}
EV_TYPES = [("N2W", "NREM → wake"), ("MA", "micro-arousal"), ("R2W", "REM → wake"), ("W2N", "wake → NREM")]
SIGS = [("ne", "NE ΔF/F0 (%)"), ("pupil", "pupil (z)"), ("art", "arteriole ΔD/D0 (%)")]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def ne_keys():
    return C.keys_for(["NE"])


def day_of(key):
    return C.parse_key(key).date


# ----------------------------------------------------------------- events
def load_events(keys):
    ev = C.response_events(keys)
    ev = ev[ev.type.isin(["N2W", "MA", "W2N"])][["type", "key", "t0"]]
    r = r2w_events(keys)
    if len(r):
        ev = pd.concat([ev, r[["type", "key", "t0"]]], ignore_index=True)
    ev["day"] = ev.key.map(day_of)
    return ev.reset_index(drop=True)


def epochs(ev, sig):
    M = np.full((len(ev), len(RG)), np.nan)
    for i, r in enumerate(ev.itertuples()):
        cyc = C.load_cycle(r.key)
        if sig not in cyc.sig:
            continue
        t, y = cyc.sig[sig]
        M[i] = DD.interp_on(t, y, r.t0 + RG)
    pm = (RG >= PRE[0]) & (RG < PRE[1])
    with np.errstate(all="ignore"):
        b = np.nanmedian(M[:, pm], 1, keepdims=True)
    if sig == "pupil":
        return M - b
    return 100 * (M - b) / (1 + b)          # signals are (x - median)/median, so 1 + b = x_pre / median


def day_mean(ev, M, idx):
    D = []
    for _, g in ev.loc[idx].groupby("day"):
        with np.errstate(all="ignore"):
            D.append(np.nanmean(M[g.index.to_numpy()], 0))
    D = np.array(D) if D else np.full((0, len(RG)), np.nan)
    with np.errstate(all="ignore"):
        mu = np.nanmean(D, 0)
        se = np.nanstd(D, 0, ddof=1) / np.sqrt(len(D)) if len(D) > 1 else np.full(len(RG), np.nan)
    return mu, se, len(D)


# ----------------------------------------------------------------- per-bin table
def bin_table(keys):
    rows = []
    for k in keys:
        cyc = C.load_cycle(k)
        L = np.asarray(cyc.labels, dtype=object)
        edges = np.arange(len(L) + 1) * C.BIN
        out = {"state": L}
        for s in ("ne", "pupil", "art"):
            if s not in cyc.sig:
                out[s] = np.full(len(L), np.nan)
                continue
            t, y = cyc.sig[s]
            idx = np.digitize(t, edges) - 1
            v = np.full(len(L), np.nan)
            for j in range(len(L)):
                m = (idx == j) & np.isfinite(y)
                if m.any():
                    v[j] = np.mean(y[m])
            out[s] = v
        d = pd.DataFrame(out)
        d["key"], d["day"], d["cycle"], d["bin"] = k, day_of(k), C.parse_key(k).cycle, np.arange(len(L))
        rows.append(d)
    B = pd.concat(rows, ignore_index=True)
    for s in ("ne", "pupil", "art"):
        B[s] = pd.to_numeric(B[s], errors="coerce").astype(float)
    return B


def fig_states(B):
    d = B[B.state.isin(STATES)]
    lev = d.groupby(["day", "state"])[["ne", "pupil", "art"]].median().reset_index()
    lev[["ne", "art"]] *= 100
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (s, lab) in zip(axs, (("ne", "NE vs cycle median (%)"), ("pupil", "pupil (robust z)"),
                                  ("art", "arteriole vs cycle median (%)"))):
        for day, g in lev.groupby("day"):
            g = g.set_index("state").reindex(STATES)
            ax.plot(range(3), g[s], "-o", color="0.5", lw=1, ms=4)
            ax.text(2.08 if np.isfinite(g[s].iloc[2]) else 1.08,
                    g[s].iloc[2] if np.isfinite(g[s].iloc[2]) else g[s].iloc[1], day, fontsize=7, va="center")
        for x, st in enumerate(STATES):
            ax.scatter([x] * len(lev[lev.state == st]), lev[lev.state == st][s], color=ST_COL[st], s=45, zorder=3)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_xticks(range(3), STATES)
        ax.set_xlim(-0.4, 2.8)
        ax.set_ylabel(lab)
    fig.suptitle("GRAB-NE days: median level in each state, per day (REM on 2025.09.11 only)", fontsize=11,
                 x=0.01, ha="left")
    fig.savefig(FIG / "1_state_levels.png", dpi=160)
    plt.close(fig)
    lev.to_csv(OUT / "state_levels_per_day.csv", index=False)
    return lev


def fig_events(ev, mats):
    fig, axs = plt.subplots(3, 4, figsize=(15, 8.5), sharex=True, constrained_layout=True)
    summ = []
    for j, (typ, tl) in enumerate(EV_TYPES):
        idx = ev.index[ev.type == typ]
        for i, (s, ylab) in enumerate(SIGS):
            ax = axs[i, j]
            mu, se, nd = day_mean(ev, mats[s], idx)
            col = {"ne": "#C62828", "pupil": "#424242", "art": "#1565C0"}[s]
            if nd:
                ax.plot(RG, mu, color=col, lw=2)
                ax.fill_between(RG, mu - se, mu + se, color=col, alpha=0.25, lw=0)
                m = (RG >= 0) & (RG <= 15)
                pk = np.nanargmax(np.abs(mu[m]))
                summ.append(dict(type=typ, signal=s, events=len(idx), days=nd, mean_0_15=np.nanmean(mu[m]),
                                 peak=mu[m][pk], peak_t=RG[m][pk]))
            ax.axvline(0, color="k", lw=0.8); ax.axhline(0, color="0.7", lw=0.8)
            if i == 0:
                ax.set_title(f"{tl}\n{len(idx)} events, {ev.loc[idx].day.nunique()} days", fontsize=11, weight="bold")
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 2:
                ax.set_xlabel("time from onset (s)")
    for i in range(3):
        lo = min(a.get_ylim()[0] for a in axs[i]); hi = max(a.get_ylim()[1] for a in axs[i])
        for a in axs[i]:
            a.set_ylim(lo, hi)
    fig.suptitle("GRAB-NE, final labels. Mean ± SEM across days (REM → wake: 1 day, so no SEM). "
                 "t0 = pupil onset (wake-ups, MA) or label boundary (wake → NREM).", fontsize=11, x=0.01, ha="left")
    fig.savefig(FIG / "2_event_curves.png", dpi=160)
    plt.close(fig)
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "event_curve_summary.csv", index=False)
    return S


def xcorr_lag(a, b, max_lag_bins=6):
    """Lag (bins) maximising corr(a[t], b[t+lag]); positive = b follows a."""
    best, bl = -np.inf, 0
    for L in range(-max_lag_bins, max_lag_bins + 1):
        x, y = (a[:len(a) - L], b[L:]) if L >= 0 else (a[-L:], b[:len(b) + L])
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 10:
            r = np.corrcoef(x[m], y[m])[0, 1]
            if r > best:
                best, bl = r, L
    return bl, best


def fig_ne_pupil(B):
    rows = []
    for k, g in B.groupby("key"):
        for st in ("all",) + STATES:
            h = g if st == "all" else g[g.state == st]
            m = (np.isfinite(h["ne"]) & np.isfinite(h["pupil"])).to_numpy()
            if m.sum() < 12:
                continue
            r = np.corrcoef(h["ne"].to_numpy()[m], h.pupil.to_numpy()[m])[0, 1]
            lag = np.nan
            if st == "all":
                lag, _ = xcorr_lag(g["ne"].to_numpy(), g.pupil.to_numpy())
            rows.append(dict(key=k, day=g.day.iloc[0], state=st, r=r, n_bins=int(m.sum()), lag_bins=lag))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "ne_pupil_correlation_per_cycle.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax = axs[0]
    for x, st in enumerate(("all",) + STATES):
        v = R[R.state == st]
        ax.scatter(np.full(len(v), x) + np.random.default_rng(x).uniform(-0.12, 0.12, len(v)), v.r, s=22,
                   color=ST_COL.get(st, "0.3"), alpha=0.8)
        dm = v.groupby("day").r.mean()
        ax.scatter([x + 0.3] * len(dm), dm, s=60, marker="D", color=ST_COL.get(st, "0.3"), edgecolor="k", lw=0.5)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xticks(range(4), ["whole cycle", "Awake", "NREM", "REM"])
    ax.set_ylabel("NE vs pupil, Pearson r (5-s bins)")
    ax.set_title("Each dot = one cycle; diamonds = day means", fontsize=10)
    ax = axs[1]
    v = R[R.state == "all"]
    ax.hist(v.lag_bins * C.BIN, bins=np.arange(-32.5, 33, 5), color="0.5", edgecolor="white")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("lag of pupil after NE (s)\n(positive = pupil follows NE)")
    ax.set_ylabel("cycles")
    ax.set_title(f"Best cross-correlation lag, median {np.median(v.lag_bins) * C.BIN:+.0f} s", fontsize=10)
    fig.suptitle("How closely does pupil track NE?", fontsize=11, x=0.01, ha="left")
    fig.savefig(FIG / "3_ne_vs_pupil.png", dpi=160)
    plt.close(fig)
    return R


def fig_ma(ev, mats):
    idx = ev.index[ev.type == "MA"]
    m = (RG >= 0) & (RG <= 10)
    with np.errstate(all="ignore"):
        ne = np.nanmax(mats["ne"][idx][:, m], 1)
        pu = np.nanmax(mats["pupil"][idx][:, m], 1)
        art = np.nanmean(mats["art"][idx][:, (RG >= 0) & (RG <= 15)], 1)
    d = pd.DataFrame(dict(day=ev.loc[idx, "day"].to_numpy(), key=ev.loc[idx, "key"].to_numpy(),
                          t0=ev.loc[idx, "t0"].to_numpy(), ne_rise=ne, pupil_rise=pu, art_mean=art))
    d.to_csv(OUT / "microarousal_events.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for day, mk in zip(sorted(d.day.unique()), "osD"):
        g = d[d.day == day]
        axs[0].scatter(g.pupil_rise, g.ne_rise, marker=mk, s=36, alpha=0.8, label=f"{day} ({len(g)})")
        axs[1].scatter(g.ne_rise, g.art_mean, marker=mk, s=36, alpha=0.8, label=day)
    v = d.dropna(subset=["ne_rise", "pupil_rise"])
    rk = v.groupby("day")[["ne_rise", "pupil_rise"]].rank(pct=True)
    rho, p = stats.spearmanr(rk.ne_rise, rk.pupil_rise)
    axs[0].set_title(f"NE rise vs pupil rise, within-day ρ = {rho:+.2f} (p = {p:.3f}, events as units)", fontsize=10)
    axs[0].set_xlabel("pupil rise, max 0–10 s (z)"); axs[0].set_ylabel("NE rise, max 0–10 s (%)")
    axs[0].legend(frameon=False, fontsize=8)
    v2 = d.dropna(subset=["ne_rise", "art_mean"])
    rk2 = v2.groupby("day")[["ne_rise", "art_mean"]].rank(pct=True)
    rho2, p2 = stats.spearmanr(rk2.ne_rise, rk2.art_mean)
    axs[1].axhline(0, color="0.8", lw=0.8)
    axs[1].set_title(f"NE rise vs arteriole 0–15 s, within-day ρ = {rho2:+.2f} (p = {p2:.3f})", fontsize=10)
    axs[1].set_xlabel("NE rise, max 0–10 s (%)"); axs[1].set_ylabel("arteriole ΔD/D0, mean 0–15 s (%)")
    fig.suptitle("Micro-arousals (5–15 s Awake inside NREM), each dot = one event", fontsize=11, x=0.01, ha="left")
    fig.savefig(FIG / "4_ma_pupil.png", dpi=160)
    plt.close(fig)
    return dict(n=len(d), rho_ne_pupil=rho, p_ne_pupil=p, rho_ne_art=rho2, p_ne_art=p2,
                ne_rise_median=float(np.nanmedian(d.ne_rise)), pupil_rise_median=float(np.nanmedian(d.pupil_rise)),
                frac_ne_up=float(np.mean(d.ne_rise > 0)), frac_pupil_gt05=float(np.mean(d.pupil_rise > 0.5)))


def main():
    C.set_label_mode("manual")
    FIG.mkdir(parents=True, exist_ok=True)
    keys = ne_keys()
    B = bin_table(keys)
    comp = B.groupby("day").state.value_counts().unstack(fill_value=0) * C.BIN / 60
    ev = load_events(keys)
    mats = {s: epochs(ev, s) for s, _ in SIGS}
    ev.to_csv(OUT / "events.csv", index=False)
    lev = fig_states(B)
    S = fig_events(ev, mats)
    R = fig_ne_pupil(B)
    M = fig_ma(ev, mats)
    pd.set_option("display.width", 220)
    print("cycles:", len(keys), "| minutes by state:\n", comp.round(1))
    print("events:", ev.groupby(["type", "day"]).size().unstack(fill_value=0))
    print(lev.round(2).to_string(index=False))
    print(S.round(2).to_string(index=False))
    print(R.groupby("state").agg(r_median=("r", "median"), r_min=("r", "min"), r_max=("r", "max"), cycles=("r", "size"),
                                 lag_median_bins=("lag_bins", "median")).round(2))
    print("MA:", {k: round(v, 3) if isinstance(v, float) else v for k, v in M.items()})
    print("->", OUT)


if __name__ == "__main__":
    main()
