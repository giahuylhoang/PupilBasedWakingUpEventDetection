"""Dilating vs non-dilating wake-ups in WT: what separates them, and is it cycle?

Events: final labels (PRELIM_LABEL_SET=relabel_2026-10-09), manual only, WT.
  N2W  common.response_events type N2W   (NREM >= 30 s then Awake >= 20 s)
  R2W  rem_vs_nrem_wake.r2w_events       (REM >= 20 s then Awake >= 20 s)
  t0 = pupil onset. Same normalisation as the findings decks
  (deck_part1_figures): ΔD/D0, D0 = median of -20..0 s; ΔF/F0 the same.
Excluded as noisy (Grant, 2026-10-14): 2026.03.03 cycle 13; 2026.03.06 cycles 13-16.

Class: an event "dilates" if its mean arteriole ΔD/D0 over 0..15 s > +THR %.
"Cycle" = Florence's numbered 15-min recording block; higher number = later in
the session (cycle 2 = first block, mouse falling asleep).

Per-event features compared between classes:
  cycle, cycle order within the day, minutes into the cycle, length of the sleep
  bout before waking, sleep (NREM+REM) earlier in the same cycle and earlier in
  the day, REM in the 3 min before, pupil rise, pupil level before waking,
  Ca2+ response, vessel level before waking (vs the cycle median), vessel noise.

Tests: events as units with a within-day permutation (class labels shuffled
inside each recording day), so a feature that only differs between days does
not count. Cycle trend: rank correlation of the response with cycle number,
within day.

Usage:
    PRELIM_LABEL_SET=relabel_2026-10-09 venv/bin/python -B scripts/dilation_split.py [--thr 1.0]
Also: vessel size per sleep state (vs cycle median) per cycle and day.

Output: data/RESULTS_dilation_split_2026-10-14/ (thr_<x>/ for a non-default --thr)
"""
from __future__ import annotations

import argparse
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
import deck_part1_figures as P1  # noqa: E402
from rem_vs_nrem_wake import r2w_events  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--thr", type=float, default=1.0, help="dilation threshold, %% mean ΔD/D0 over 0-15 s")
ap.add_argument("--n-perm", type=int, default=5000)
ARGS = ap.parse_args()

OUT = C.ROOT / "data" / "RESULTS_dilation_split_2026-10-14"
if ARGS.thr != 1.0:
    OUT = OUT / f"thr_{ARGS.thr:g}"
FIG = OUT / "figures"
RG, PRE = P1.RG, P1.PRE
EXCLUDE = {("2026.03.03", 13), ("2026.03.06", 13), ("2026.03.06", 14), ("2026.03.06", 15), ("2026.03.06", 16)}
COH_COL = {"RSC2023": "#424242", "GCaMP8m": "#2E7D32"}
COH_LAB = {"RSC2023": "WT RSC 2023", "GCaMP8m": "WT GCaMP8m 2025–26"}
CLS_COL = {"dilating": "#C62828", "non-dilating": "#1565C0"}
RNG = np.random.default_rng(20261014)

FEATURES = [
    ("cycle", "cycle number"),
    ("cycle_order", "cycle order in the day (1 = first)"),
    ("t_in_cycle_min", "minutes into the cycle"),
    ("pre_sleep_bout_s", "sleep bout before waking (s)"),
    ("sleep_before_in_cycle_min", "sleep earlier in this cycle (min)"),
    ("sleep_before_in_day_min", "sleep earlier in the day, recorded cycles (min)"),
    ("rem_prior_3min_s", "REM in the 3 min before (s)"),
    ("pupil_rise", "pupil rise 0–15 s (z)"),
    ("pupil_pre", "pupil level before waking (z)"),
    ("ca_mean", "Ca²⁺ ΔF/F0 0–15 s (%)"),
    ("art_pre_level", "vessel before waking vs cycle median (%)"),
    ("art_noise", "vessel noise before waking (%, SD of steps)"),
]


# ----------------------------------------------------------------- events + features
def load():
    keys = C.keys_for(["WT"])
    n = C.response_events(keys)
    n = n[n["type"] == "N2W"][["type", "key", "mouse", "t0", "pre_dur_s"]]
    r = r2w_events(keys)[["type", "key", "mouse", "t0", "pre_dur_s"]]
    ev = pd.concat([n, r], ignore_index=True)
    ev["cohort"] = np.where(ev["key"].str.startswith("Extra_GCaMP8m"), "GCaMP8m", "RSC2023")
    ev["day"] = ev["mouse"].str.split("_").str[-1]
    ev["cycle"] = ev["key"].str.extract(r"_cycle(\d+)$")[0].astype(int)
    ev["excluded"] = [(d, c) in EXCLUDE for d, c in zip(ev.day, ev.cycle)]
    ev = ev.reset_index(drop=True)

    E = {s: P1.epoch_matrix(ev, s) for s in ("pupil", "ca", "art")}
    m = (RG >= 0) & (RG <= 15)
    pre = (RG >= PRE[0]) & (RG < PRE[1])
    with np.errstate(all="ignore"):
        ev["art_mean"] = np.nanmean(E["art"][:, m], 1)
        ev["ca_mean"] = np.nanmean(E["ca"][:, m], 1)
        ev["pupil_rise"] = np.nanmax(E["pupil"][:, m], 1)
        ev["art_noise"] = np.nanstd(np.diff(E["art"][:, pre], axis=1), 1)

    # features from labels and raw (cycle-median-normalised) traces
    order = {}
    for d, g in ev.groupby("day"):
        cyc_all = sorted({int(k.split("_cycle")[-1]) for k in C.keys_for(["WT"]) if d in k})
        order[d] = {c: i + 1 for i, c in enumerate(cyc_all)}
    sleep_per_cycle = {}
    for k in C.keys_for(["WT"]):
        L = C.load_labels(k)
        sleep_per_cycle[k] = np.array([x in ("NREM", "REM") for x in L])
    rows = []
    for r_ in ev.itertuples():
        L = C.load_labels(r_.key)
        i0 = int(r_.t0 // C.BIN)
        asleep = sleep_per_cycle[r_.key]
        earlier = [k for k in sleep_per_cycle if r_.day in k and int(k.split("_cycle")[-1]) < r_.cycle]
        cyc = C.load_cycle(r_.key)
        out = dict(
            cycle_order=order[r_.day][r_.cycle],
            t_in_cycle_min=r_.t0 / 60,
            pre_sleep_bout_s=r_.pre_dur_s,
            sleep_before_in_cycle_min=asleep[:i0].sum() * C.BIN / 60,
            sleep_before_in_day_min=sum(sleep_per_cycle[k].sum() for k in earlier) * C.BIN / 60,
            rem_prior_3min_s=sum(x == "REM" for x in L[max(0, i0 - 36):i0]) * C.BIN,
        )
        # pupil level before waking (cycle robust-z, not baseline-subtracted)
        t, y = cyc.sig["pupil"]
        w = (t >= r_.t0 + PRE[0]) & (t < r_.t0 + PRE[1])
        out["pupil_pre"] = np.nanmedian(y[w]) if w.any() else np.nan
        # vessel before waking relative to the whole-cycle median
        if "art" in cyc.sig:
            t, y = cyc.sig["art"]
            w = (t >= r_.t0 + PRE[0]) & (t < r_.t0 + PRE[1])
            out["art_pre_level"] = 100 * np.nanmedian(y[w]) if w.any() else np.nan
        rows.append(out)
    ev = pd.concat([ev, pd.DataFrame(rows)], axis=1)
    ev["cls"] = np.where(ev.art_mean > ARGS.thr, "dilating", "non-dilating")
    ev.loc[ev.art_mean.isna(), "cls"] = np.nan
    return ev, E


# ----------------------------------------------------------------- stats
def perm_within_day(d, feat):
    """Difference (dilating - non-dilating) in mean within-day percentile rank of
    feat; p from shuffling class labels inside each day."""
    v = d[["day", "cls", feat]].dropna()
    if v.cls.nunique() < 2 or len(v) < 6:
        return np.nan, np.nan, len(v)
    v = v.assign(r=v.groupby("day")[feat].rank(pct=True))
    is_d = (v.cls == "dilating").to_numpy()
    rr = v.r.to_numpy()
    obs = rr[is_d].mean() - rr[~is_d].mean()
    groups = [np.flatnonzero((v.day == dd).to_numpy()) for dd in v.day.unique()]
    null = np.empty(ARGS.n_perm)
    lab = is_d.copy()
    for b in range(ARGS.n_perm):
        for gi in groups:
            lab[gi] = RNG.permutation(is_d[gi])
        if lab.all() or (~lab).all():
            null[b] = 0.0
            continue
        null[b] = rr[lab].mean() - rr[~lab].mean()
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (ARGS.n_perm + 1)
    return obs, p, len(v)


def compare(ev):
    rows = []
    for (coh, typ), d in ev[~ev.excluded].groupby(["cohort", "type"]):
        for f, lab in FEATURES:
            a = d.loc[d.cls == "dilating", f].dropna(); b = d.loc[d.cls == "non-dilating", f].dropna()
            diff, p, nn = perm_within_day(d, f)
            rows.append(dict(cohort=coh, type=typ, feature=f, label=lab, n_dil=len(a), n_non=len(b),
                             median_dil=a.median() if len(a) else np.nan,
                             median_non=b.median() if len(b) else np.nan,
                             rank_diff_dil_minus_non=diff, p_perm_within_day=p))
    T = pd.DataFrame(rows)
    T["q_bh"] = np.nan
    for key, idx in T.groupby(["cohort", "type"]).groups.items():
        ok = T.loc[idx, "p_perm_within_day"].notna()
        T.loc[idx[ok.to_numpy()], "q_bh"] = C.bh(T.loc[idx[ok.to_numpy()], "p_perm_within_day"].to_numpy())
    T.to_csv(OUT / "class_feature_tests.csv", index=False)
    return T


def cycle_trend(ev):
    rows = []
    for (coh, typ), d in ev[~ev.excluded].groupby(["cohort", "type"]):
        for f in ("art_mean", "ca_mean", "pupil_rise"):
            v = d[["day", "cycle", f]].dropna()
            if v.day.nunique() == 0 or len(v) < 6:
                continue
            r = v.groupby("day")[["cycle", f]].rank(pct=True)
            rho, p = stats.spearmanr(r.cycle, r[f])
            rows.append(dict(cohort=coh, type=typ, signal=f, n_events=len(v), n_days=v.day.nunique(),
                             within_day_rho=rho, p=p))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "cycle_trend.csv", index=False)
    return T


def cohort_after_exclusion(ev):
    rows = []
    for excl in (False, True):
        d = ev[(ev.type == "N2W") & (~ev.excluded if excl else True)]
        dm = d.groupby(["cohort", "day"]).art_mean.mean().reset_index()
        a = dm[dm.cohort == "RSC2023"].art_mean.to_numpy(); b = dm[dm.cohort == "GCaMP8m"].art_mean.to_numpy()
        r = C.group_test(a, b)
        rows.append(dict(noisy_cycles_excluded=excl, rsc_days=len(a), gcamp_days=len(b), rsc_mean=a.mean(),
                         gcamp_mean=b.mean(), gcamp_events=int((d.cohort == "GCaMP8m").sum()),
                         welch_p=r["welch_p"], mw_p=r["mw_p"]))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "cohort_n2w_after_exclusion.csv", index=False)
    return T


# ----------------------------------------------------------------- figures
def fig_distribution(ev):
    d = ev[~ev.excluded & ev.art_mean.notna()]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True, constrained_layout=True)
    for ax, typ in zip(axs, ("N2W", "R2W")):
        x = 0
        ticks = []
        for coh in ("RSC2023", "GCaMP8m"):
            for day in sorted(d[(d.cohort == coh)].day.unique()):
                v = d[(d.cohort == coh) & (d.day == day) & (d.type == typ)]
                if not len(v):
                    continue
                ax.scatter(np.full(len(v), x) + RNG.uniform(-0.18, 0.18, len(v)), v.art_mean,
                           c=[CLS_COL[c] for c in v.cls], s=26, alpha=0.85)
                ticks.append((x, f"{day}\n({len(v)})", COH_COL[coh]))
                x += 1
        ax.axhline(ARGS.thr, color="0.4", ls="--", lw=1)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=8, rotation=0)
        for lbl, t in zip(ax.get_xticklabels(), ticks):
            lbl.set_color(t[2])
        ax.set_title({"N2W": "NREM → wake", "R2W": "REM → wake"}[typ], fontsize=12, weight="bold")
    axs[0].set_ylabel("arteriole ΔD/D0, mean 0–15 s (%)")
    fig.suptitle(f"Each wake-up, coloured by class (red = dilating, > +{ARGS.thr:g}%). Grey day labels = RSC 2023, "
                 "green = GCaMP8m. Noisy cycles excluded.", fontsize=10.5, x=0.01, ha="left")
    fig.savefig(FIG / "1_distribution.png", dpi=150)
    plt.close(fig)


def fig_traces(ev, E):
    d = ev[~ev.excluded & ev.cls.notna()]
    combos = [("RSC2023", "N2W"), ("GCaMP8m", "N2W"), ("GCaMP8m", "R2W")]
    fig, axs = plt.subplots(3, 3, figsize=(15, 9.5), sharex=True, constrained_layout=True)
    for row, (coh, typ) in enumerate(combos):
        dd = d[(d.cohort == coh) & (d.type == typ)]
        for col, s in enumerate(("pupil", "ca", "art")):
            ax = axs[row, col]
            M = E[s] if s != "art" else pd.DataFrame(E["art"].T).rolling(3, center=True, min_periods=1).mean().to_numpy().T
            for c in ("dilating", "non-dilating"):
                idx = dd.index[dd.cls == c].to_numpy()
                if not len(idx):
                    continue
                mu, se, nn = P1.msem(M[idx])
                ax.plot(RG, mu, color=CLS_COL[c], lw=1.8, label=f"{c} (n={nn}, {dd.loc[idx, 'day'].nunique()} days)")
                ax.fill_between(RG, mu - se, mu + se, color=CLS_COL[c], alpha=0.18, lw=0)
            P1.shade(ax)
            ax.set_ylabel(P1.SIG_LAB[s])
            if col == 0:
                ax.set_title(f"{COH_LAB[coh]}, {'NREM' if typ == 'N2W' else 'REM'} → wake", fontsize=11,
                             weight="bold", color=COH_COL[coh], loc="left")
            if row == 2:
                ax.set_xlabel("time from pupil onset (s)")
            ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.savefig(FIG / "2_class_traces.png", dpi=150)
    plt.close(fig)


def fig_cycle(ev):
    d = ev[ev.art_mean.notna()]
    days = [(c, dd) for c in ("RSC2023", "GCaMP8m") for dd in sorted(d[d.cohort == c].day.unique())]
    fig, axs = plt.subplots(1, len(days), figsize=(2.1 * len(days), 4.4), sharey=True, constrained_layout=True)
    for ax, (coh, day) in zip(axs, days):
        v = d[d.day == day]
        for typ, mk in (("N2W", "o"), ("R2W", "^")):
            w = v[v.type == typ]
            ok = ~w.excluded
            ax.scatter(w.cycle[ok] + RNG.uniform(-0.15, 0.15, ok.sum()), w.art_mean[ok], marker=mk, s=28,
                       c=[CLS_COL[c] for c in w.cls[ok]], alpha=0.85)
            ax.scatter(w.cycle[~ok] + RNG.uniform(-0.15, 0.15, (~ok).sum()), w.art_mean[~ok], marker=mk, s=28,
                       facecolors="none", edgecolors="0.6")
        means = v[~v.excluded & (v.type == "N2W")].groupby("cycle").art_mean.mean()
        ax.plot(means.index, means.values, color="k", lw=1.5)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_title(day, fontsize=10, color=COH_COL[coh], weight="bold")
        ax.set_xlabel("cycle")
        cyc = sorted(v.cycle.unique())
        ax.set_xticks(cyc)
    axs[0].set_ylabel("arteriole ΔD/D0, mean 0–15 s (%)")
    fig.suptitle("Wake-up response by cycle. Circles = NREM → wake, triangles = REM → wake; red = dilating, blue = not; "
                 "open grey = excluded noisy cycle. Black line = NREM → wake cycle mean.", fontsize=10, x=0.01, ha="left")
    fig.savefig(FIG / "3_by_cycle.png", dpi=150)
    plt.close(fig)


def fig_features(ev, T):
    d = ev[~ev.excluded & ev.cls.notna()]
    combos = [("RSC2023", "N2W"), ("GCaMP8m", "N2W"), ("GCaMP8m", "R2W")]
    fig, axs = plt.subplots(len(combos), len(FEATURES), figsize=(2.05 * len(FEATURES), 7.6), constrained_layout=True)
    for row, (coh, typ) in enumerate(combos):
        dd = d[(d.cohort == coh) & (d.type == typ)]
        for col, (f, lab) in enumerate(FEATURES):
            ax = axs[row, col]
            for x, c in enumerate(("dilating", "non-dilating")):
                v = dd.loc[dd.cls == c, f].dropna()
                ax.scatter(np.full(len(v), x) + RNG.uniform(-0.2, 0.2, len(v)), v, s=10, color=CLS_COL[c], alpha=0.6)
                if len(v):
                    ax.plot([x - 0.3, x + 0.3], [v.median()] * 2, color="k", lw=2)
            t = T[(T.cohort == coh) & (T.type == typ) & (T.feature == f)]
            p = t.p_perm_within_day.iloc[0] if len(t) else np.nan
            ax.set_title(("" if not np.isfinite(p) else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"),
                         fontsize=8.5, color="#C62828" if np.isfinite(p) and p < 0.05 else "0.3")
            ax.set_xticks([0, 1], ["dil", "non"], fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
            if row == 0:
                ax.text(0.5, 1.22, lab, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, wrap=True)
            if col == 0:
                ax.set_ylabel(f"{COH_LAB[coh]}\n{'NREM' if typ == 'N2W' else 'REM'} → wake", fontsize=9,
                              color=COH_COL[coh], weight="bold")
    fig.savefig(FIG / "4_features.png", dpi=150)
    plt.close(fig)


def state_levels():
    """Vessel size in each state, relative to the cycle median (%), per cycle and per day."""
    rows = []
    for k in C.keys_for(["WT"]):
        info = C.parse_key(k)
        if (info.date, info.cycle) in EXCLUDE:
            continue
        cyc = C.load_cycle(k)
        if "art" not in cyc.sig:
            continue
        t, y = cyc.sig["art"]
        L = C.load_labels(k)
        b = (t // C.BIN).astype(int)
        st = np.array([L[j] if j < len(L) else None for j in b], dtype=object)
        for s in ("Awake", "NREM", "REM"):
            m = st == s
            if m.sum() > 50:
                rows.append(dict(cohort="GCaMP8m" if k.startswith("Extra_GCaMP8m") else "RSC2023", day=info.date,
                                 cycle=info.cycle, state=s, level_pct=100 * np.nanmedian(y[m]), n_samples=int(m.sum())))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "vessel_by_state_per_cycle.csv", index=False)
    D = d.groupby(["cohort", "day", "state"]).level_pct.mean().unstack()
    D["NREM_minus_Awake"] = D["NREM"] - D["Awake"]
    D = D.reset_index()
    D.to_csv(OUT / "vessel_by_state_per_day.csv", index=False)
    a = D[D.cohort == "RSC2023"].NREM_minus_Awake.to_numpy(); g = D[D.cohort == "GCaMP8m"].NREM_minus_Awake.to_numpy()
    r = C.group_test(a, g)

    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    x = 0; ticks = []
    for coh in ("RSC2023", "GCaMP8m"):
        for _, row in D[D.cohort == coh].iterrows():
            v = [row.Awake, row.NREM] + ([row.REM] if np.isfinite(row.get("REM", np.nan)) else [])
            ax.plot(np.arange(len(v)) * 0.28 + x, v, "-o", color=COH_COL[coh], lw=1.5, ms=6)
            for j, (vv, lab) in enumerate(zip(v, ("W", "N", "R"))):
                ax.text(x + j * 0.28, vv, f" {lab}", fontsize=7, va="center", color="0.3")
            ticks.append((x + 0.14, row.day, COH_COL[coh]))
            x += 1
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks], rotation=35, fontsize=8)
    for lbl, t in zip(ax.get_xticklabels(), ticks):
        lbl.set_color(t[2])
    ax.set_ylabel("vessel size vs cycle median (%)")
    ax.set_title(f"Vessel size in Awake (W), NREM (N), REM (R), per day\nNREM − Awake: RSC {a.mean():+.1f}%, "
                 f"GCaMP8m {g.mean():+.1f}% (Welch p = {r['welch_p']:.3f}, days)", fontsize=10)
    fig.savefig(FIG / "5_vessel_by_state.png", dpi=160)
    plt.close(fig)
    return D, r


def main():
    C.set_label_mode("manual")
    FIG.mkdir(parents=True, exist_ok=True)
    ev, E = load()
    ev.to_csv(OUT / "events.csv", index=False)
    T = compare(ev)
    Tc = cycle_trend(ev)
    Tx = cohort_after_exclusion(ev)
    fig_distribution(ev)
    fig_traces(ev, E)
    fig_cycle(ev)
    fig_features(ev, T)
    Ds, rs = state_levels()
    pd.set_option("display.width", 220)
    keep = ev[~ev.excluded]
    print("events kept / excluded:", int((~ev.excluded).sum()), int(ev.excluded.sum()))
    print(pd.crosstab([keep.cohort, keep.type], keep.cls, margins=True))
    print(pd.crosstab([keep.cohort, keep.day], [keep.type, keep.cls]))
    print(T[["cohort", "type", "feature", "n_dil", "n_non", "median_dil", "median_non",
             "rank_diff_dil_minus_non", "p_perm_within_day", "q_bh"]].round(3).to_string(index=False))
    print(Tc.round(3).to_string(index=False))
    print(Tx.round(4).to_string(index=False))
    print(Ds.round(2).to_string(index=False)); print("NREM-Awake RSC vs GCaMP8m:", {k: round(v, 4) for k, v in rs.items()})
    for coh, d in ev[(ev.type == "N2W") & ~ev.excluded].groupby("cohort"):
        v = d[["day", "ca_mean", "art_mean"]].dropna()
        r = v.groupby("day")[["ca_mean", "art_mean"]].rank(pct=True)
        print(coh, "Ca vs arteriole within-day rho = %.2f, p = %.3f, n = %d" % (*stats.spearmanr(r.ca_mean, r.art_mean), len(v)))
    print("->", OUT)


if __name__ == "__main__":
    main()
