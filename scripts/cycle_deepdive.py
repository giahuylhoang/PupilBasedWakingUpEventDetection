"""Cycle deep dive (2026-10-16): what changes across recording cycles, and does it explain
the GCaMP8m wake-up dip?

Same events / exclusions as scripts/dilation_deepdive.py. One row per cycle, then each
measure plotted against cycle number with one line per day:
  N2W and R2W arteriole response, N2W Ca²⁺ response, NREM - Awake vessel level,
  REM level, raw vessel size, % of the cycle in NREM / REM / Awake.
Stats: for each cohort, Spearman of cycle number vs measure on within-day ranks (cycles as
units), and the 2025.12.18 day alone (the only GCaMP8m day that starts early, cycles 6-12).

Usage:
    PRELIM_LABEL_SET=relabel_2026-10-09 venv/bin/python -B scripts/cycle_deepdive.py
Output: data/RESULTS_cycle_deepdive_2026-10-16/
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

OUT = C.ROOT / "data" / "RESULTS_cycle_deepdive_2026-10-16"
FIG = OUT / "figures"
DD.OUT = OUT                                   # cycle_table() writes its CSV here, not in the deepdive folder
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

MEASURES = [
    ("n2w_resp", "NREM → wake arteriole (%)", True),
    ("r2w_resp", "REM → wake arteriole (%)", True),
    ("n2w_ca", "NREM → wake Ca²⁺ (%)", True),
    ("nrem_minus_awake", "vessel NREM − Awake (%)", True),
    ("drift_pct_over_cycle", "vessel drift over the cycle (%)", True),
    ("equiv_diam_px", "vessel size (equiv. diameter, px)", False),
    ("pct_nrem", "% of cycle in NREM", False),
    ("pct_rem", "% of cycle in REM", False),
    ("pct_awake", "% of cycle Awake", False),
]


def add_cycle_extras(T, ev):
    ca = ev[ev.type == "N2W"].groupby("key")["ca_resp"].mean().rename("n2w_ca")
    T = T.merge(ca, left_on="key", right_index=True, how="left")
    comp = []
    for k in T.key:
        L = np.asarray(C.load_cycle(k).labels, dtype=object)
        n = max(1, sum(x is not None for x in L))
        comp.append({s: 100 * np.sum(L == s) / n for s in ("NREM", "REM", "Awake")})
    comp = pd.DataFrame(comp)
    T["pct_nrem"], T["pct_rem"], T["pct_awake"] = comp.NREM.values, comp.REM.values, comp.Awake.values
    return T


def within_day_cycle_rho(d, m):
    v = d[["day", "cycle", m]].dropna()
    v = v[v.groupby("day")[m].transform("size") >= 3]          # need >= 3 cycles in a day to rank
    if len(v) < 5:
        return np.nan, np.nan, len(v)
    r = v.groupby("day")[["cycle", m]].rank(pct=True)
    rho, p = stats.spearmanr(r.cycle, r[m])
    return rho, p, len(v)


def fig_cycles(T):
    fig, axs = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    rows = []
    for ax, (m, lab, zero) in zip(axs.flat, MEASURES):
        for coh in ("RSC2023", "GCaMP8m"):
            d = T[T.cohort == coh]
            days = sorted(d.day.unique())
            cm = plt.get_cmap("Greys" if coh == "RSC2023" else "Greens")
            for i, day in enumerate(days):
                g = d[d.day == day].sort_values("cycle").dropna(subset=[m])
                if not len(g):
                    continue
                col = cm(0.45 + 0.5 * i / max(1, len(days) - 1))
                ax.plot(g.cycle, g[m], "-o", color=col, ms=5, lw=1.4, label=day)
            rho, p, n = within_day_cycle_rho(d, m)
            rows.append(dict(measure=m, label=lab, cohort=coh, rho_within_day=rho, p=p, n_cycles=n))
        g = T[(T.day == "2025.12.18")].dropna(subset=[m])
        if len(g) >= 4:
            rho, p = stats.spearmanr(g.cycle, g[m])
            rows.append(dict(measure=m, label=lab, cohort="GCaMP8m 2025.12.18 only", rho_within_day=rho, p=p,
                             n_cycles=len(g)))
        if zero:
            ax.axhline(0, color="0.7", lw=0.8)
        ax.axvspan(5.5, 10.5, color="#FFF3E0", zorder=-1)
        r = pd.DataFrame(rows)
        r = r[r.measure == m].set_index("cohort")
        def fmt(c):
            if c not in r.index or not np.isfinite(r.loc[c, "rho_within_day"]):
                return "–"
            return f"{r.loc[c, 'rho_within_day']:+.2f} (p={r.loc[c, 'p']:.2f})"
        ax.set_title(f"{lab}\nρ vs cycle: RSC {fmt('RSC2023')}; GCaMP8m {fmt('GCaMP8m')}", fontsize=10)
        ax.set_xlabel("cycle number")
    axs.flat[0].legend(frameon=False, fontsize=7.5, ncol=2, loc="best")
    fig.suptitle("Per cycle, one line per day (grey = RSC 2023, green = GCaMP8m). Orange band = cycles 6–10, "
                 "where both cohorts have data. ρ = within-day rank correlation, cycles as units.",
                 fontsize=11, x=0.01, ha="left")
    fig.savefig(FIG / "1_by_cycle.png", dpi=150)
    plt.close(fig)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "cycle_trends.csv", index=False)
    return S


def fig_overlap(ev):
    """Events in cycles 6-10 only (where both cohorts were recorded), per day."""
    d = ev[(ev.type == "N2W") & ev.cycle.between(6, 10)]
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    g = d.groupby(["cohort", "day"]).agg(m=("art_resp", "mean"), n=("art_resp", "size"),
                                         cyc=("cycle", lambda c: f"c{c.min()}–{c.max()}")).reset_index()
    g = g.sort_values(["cohort", "day"], ascending=[False, True]).reset_index(drop=True)
    for i, r in g.iterrows():
        col = DD.COH_COL[r.cohort]
        x = d[(d.day == r.day)].art_resp
        ax.scatter(np.full(len(x), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(x)), x, s=14, color=col,
                   alpha=0.5)
        ax.plot([i - 0.3, i + 0.3], [r.m] * 2, color=col, lw=3)
        ax.text(i, -10.3, f"{r.n} ev\n{r.cyc}", ha="center", fontsize=8, color="0.3")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xticks(range(len(g)), g.day, rotation=30, fontsize=9)
    for t, c in zip(ax.get_xticklabels(), g.cohort):
        t.set_color(DD.COH_COL[c])
    ax.set_ylim(-11, 16)
    ax.set_ylabel("NREM → wake arteriole ΔD/D0, 0–15 s (%)")
    ax.set_title("Only cycles 6–10: each day's events (dots) and mean (bar)", fontsize=11)
    fig.savefig(FIG / "2_overlap_cycles_6_10.png", dpi=160)
    plt.close(fig)
    g.to_csv(OUT / "overlap_cycles_6_10_per_day.csv", index=False)
    return g


def main():
    C.set_label_mode("manual")
    FIG.mkdir(parents=True, exist_ok=True)
    keys = DD.kept_keys()
    ev, _ = DD.load_events(keys)
    ev.to_csv(OUT / "events.csv", index=False)
    T = add_cycle_extras(DD.cycle_table(keys, ev), ev)
    T.to_csv(OUT / "cycle_table.csv", index=False)
    S = fig_cycles(T)
    g = fig_overlap(ev)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string(index=False))
    print(g.round(2).to_string(index=False))
    print(T.sort_values(["cohort", "day", "cycle"])[["cohort", "day", "cycle", "n_n2w", "n2w_resp", "n_r2w", "r2w_resp",
          "n2w_ca", "nrem_minus_awake", "rem_level", "equiv_diam_px", "pct_nrem", "pct_rem", "pct_awake"]]
          .round(1).to_string(index=False))
    print("->", OUT)


if __name__ == "__main__":
    main()
