#!/usr/bin/env python3
"""One-slide figure: two ways of defining micro-arousals (MA), compared.

  Method 1  label-based: short Awake bout (5-15 s) in the manual sleep labels,
            NREM >= 10 s before and after; onset anchored to pupil dilation.
            (common.response_events, as in Fig 3 / Fig 8, v2 labels.)
  Method 2  pupil-detector: raw pupil-event detector output (May 2026), events
            3-15 s long with NREM in the 10 s before and after
            (scripts/pupil_events_ma.py).

Panels: definitions | pupil and arteriole event averages (scramble vs KD) for each
method | key numbers. Numbers are read from the result tables, never typed in.

Output -> data/RESULTS_slides_2026-10-02/MA_methods_comparison.{png,pdf}
    ./venv/bin/python -B scripts/slide_ma_methods.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figs_prelim import common as C  # noqa: E402

OUT = ROOT / "data" / "RESULTS_slides_2026-10-02"
V2T = ROOT / "data" / "RESULTS_figures_prelim_2026-10-01_v2" / "tables"
PE = ROOT / "data" / "RESULTS_pupil_event_MA_2026-10-02" / "tables"
GROUPS = ("SCR", "KD")


def day_curve(ev, sig):
    rg, M = C.epochs_for(ev.reset_index(drop=True), sig)
    pm = pd.DataFrame(M).groupby(ev.mouse.values).mean().dropna(how="all")
    mu = pm.mean().to_numpy()
    se = pm.std(ddof=1).to_numpy() / np.sqrt(len(pm))
    return rg, mu, se, len(pm)


def draw_eta(ax, events_by_group, sig, ylabel, title=None, note=None):
    for g in GROUPS:
        e = events_by_group[g]
        rg, mu, se, nd = day_curve(e, sig)
        ax.plot(rg, mu, color=C.GROUP_COL[g], lw=1.8, label=f"{C.GROUP_LABEL[g]}  ({nd} days, {len(e)} events)")
        ax.fill_between(rg, mu - se, mu + se, color=C.GROUP_COL[g], alpha=.18, lw=0)
    ax.axvspan(*C.BASE_WIN, color="0.88", lw=0, zorder=0)
    ax.axvspan(*C.RESP_WIN, color="#FFF3E0", lw=0, zorder=0)
    ax.axvline(0, color="k", lw=.7)
    ax.axhline(0, color="0.6", lw=.5)
    ax.set_xlim(-15, 30)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if note:
        ax.text(0.98, 0.95, note, transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=.6))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    C.set_label_mode("manual+ml")

    # ---- method 1: label-based MAs (v2 definition) ----
    keys = C.keys_for(list(GROUPS))
    ev1 = C.response_events(keys)
    ev1 = ev1[ev1.type == "MA"]
    m1 = {g: ev1[ev1.group == g] for g in GROUPS}

    # ---- method 2: pupil-detector MAs ----
    E = pd.read_csv(PE / "all_detector_events_classified.csv")
    ev2 = E[(E.context == "MA") & E.group.isin(GROUPS)].assign(t0=lambda d: d.Start)
    m2 = {g: ev2[ev2.group == g] for g in GROUPS}

    # ---- numbers from the result tables ----
    g1 = pd.read_csv(V2T / "fig8_F_MA_group_test.csv").set_index("metric")
    k2 = pd.read_csv(PE / "MA_KD_vs_SCR.csv")
    k2 = k2[k2.labels == "manual+ml"].set_index("measure")
    s2 = pd.read_csv(PE / "MA_vs_surrogate_tests.csv")
    agree = pd.read_csv(PE / "MA_vs_manual_labels.csv", index_col=0)
    found = pd.read_csv(PE / "label_MAs_found_by_detector.csv", index_col=0)
    rate = pd.read_csv(PE / "MA_rate_per_day.csv")

    a1, a2 = g1.loc["art_auc"], k2.loc["art_auc"]
    n_in_nrem = int(agree.loc["All", "NREM throughout"]) if "NREM throughout" in agree.columns else np.nan
    n_tot = int(agree.loc["All", "All"])
    surr_min_q = s2.t_q_bh.min()
    r_scr = rate[rate.group == "SCR"].MA_per_h_NREM.mean()
    r_kd = rate[rate.group == "KD"].MA_per_h_NREM.mean()
    found_scr, found_kd = found.loc["SCR", "frac_with_pupil_event"], found.loc["KD", "frac_with_pupil_event"]

    # ---- figure ----
    fig = plt.figure(figsize=(13.33, 7.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1, 1, 1.45], height_ratios=[1, 1],
                          wspace=0.36, hspace=0.32, left=0.03, right=0.985, top=0.86, bottom=0.09)
    fig.suptitle("Two ways to define micro-arousals give different events", fontsize=17, fontweight="bold",
                 x=0.03, ha="left", y=0.975)
    fig.text(0.03, 0.915, "IP3R2 scramble vs KD.  Event averages aligned to onset (0 s); grey = baseline, "
             "orange = 0–15 s response window; mean ± SEM across recording days.",
             fontsize=10, color="0.25")

    # definitions
    axd = fig.add_subplot(gs[:, 0]); axd.axis("off")
    txt = [
        ("Method 1 · from sleep labels", "#1565C0", [
            "Awake 5–15 s in manual labels",
            "≥10 s NREM before and after",
            "Onset = pupil dilation",
        ]),
        ("Method 2 · from pupil detector", "#6A1B9A", [
            "Automated pupil-rise detector",
            "Event lasts 3–15 s",
            "NREM 10 s before and after",
            "Labels only give sleep context",
        ]),
    ]
    y = 0.98
    for head, col, lines in txt:
        axd.text(0, y, head, fontsize=12, fontweight="bold", color=col, va="top", transform=axd.transAxes)
        y -= 0.075
        for ln in lines:
            axd.text(0.03, y, "•  " + ln, fontsize=10,
                     va="top", transform=axd.transAxes)
            y -= 0.058
        y -= 0.06
    axd.text(0, y, "Why they differ", fontsize=12.5, fontweight="bold", color="0.2", va="top",
             transform=axd.transAxes)
    y -= 0.075
    for ln in (f"Detector catches {100*found_scr:.0f}% / {100*found_kd:.0f}%",
               "(scramble / KD) of label MAs,",
               "but as long events (end = end",
               "of whisker burst), so they fail",
               "the 3–15 s rule."):
        axd.text(0.03, y, ln, fontsize=10, va="top", transform=axd.transAxes)
        y -= 0.052

    # event averages
    draw_eta(fig.add_subplot(gs[0, 1]), m1, "pupil", "pupil (z)", "Method 1 · labels")
    draw_eta(fig.add_subplot(gs[0, 2]), m2, "pupil", "pupil (z)", "Method 2 · pupil detector")
    ax = fig.add_subplot(gs[1, 1])
    draw_eta(ax, m1, "art", "arteriole ΔD/D₀",
             note=f"KD vs scramble\narteriole AUC p = {a1.welch_p:.3f}\n(effect size g = {a1.hedges_g:.1f})")
    ax.set_xlabel("time from onset (s)", fontsize=9)
    ax = fig.add_subplot(gs[1, 2])
    draw_eta(ax, m2, "art", "arteriole ΔD/D₀",
             note=f"KD vs scramble\narteriole AUC p = {a2.welch_p:.2f}")
    ax.set_xlabel("time from onset (s)", fontsize=9)
    fig.axes[1].legend(fontsize=8, loc="upper left", frameon=False)

    # key numbers
    axk = fig.add_subplot(gs[:, 3]); axk.axis("off")
    rows = [
        ("", "Method 1", "Method 2"),
        ("Events\nscramble / KD", f"{len(m1['SCR'])} / {len(m1['KD'])}", f"{len(m2['SCR'])} / {len(m2['KD'])}"),
        ("Labelled Awake?", "yes\n(by definition)", f"no: {n_in_nrem}/{n_tot}\nlabelled NREM"),
        ("Pupil response", "clear rise", "barely rises"),
        ("Beats random\nNREM moments?", "not yet tested", f"no difference\n(all q ≥ {surr_min_q:.2f})"),
        ("Rate per h NREM\nscramble / KD", "—", f"{r_scr:.0f} / {r_kd:.0f}"),
        ("KD vs scramble\narteriole response", f"KD absent\n(p = {a1.welch_p:.3f})", f"no difference\n(p = {a2.welch_p:.2f})"),
    ]
    col_x = (0.0, 0.40, 0.71)          # label column, method 1, method 2
    head_y = 0.97
    axk.text(col_x[1] + 0.13, head_y, "Method 1", ha="center", va="top", fontsize=10.5, fontweight="bold",
             color="white", transform=axk.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", fc="#1565C0", ec="none"))
    axk.text(col_x[2] + 0.14, head_y, "Method 2", ha="center", va="top", fontsize=10.5, fontweight="bold",
             color="white", transform=axk.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", fc="#6A1B9A", ec="none"))
    y = 0.885
    step = 0.103
    for i, (lab, v1, v2) in enumerate(rows[1:]):
        if i % 2 == 0:
            axk.add_patch(plt.Rectangle((0, y - step + 0.012), 1.0, step, transform=axk.transAxes,
                                        color="#F5F5F5", lw=0, zorder=0))
        axk.text(col_x[0] + 0.01, y - step / 2 + 0.012, lab, ha="left", va="center", fontsize=9,
                 color="0.2", transform=axk.transAxes)
        hl = "KD" in lab and "arteriole" in lab
        axk.text(col_x[1] + 0.13, y - step / 2 + 0.012, v1, ha="center", va="center", fontsize=9.3,
                 transform=axk.transAxes, fontweight="bold" if hl else "normal",
                 color="#C62828" if hl else "k")
        axk.text(col_x[2] + 0.14, y - step / 2 + 0.012, v2, ha="center", va="center", fontsize=9.3,
                 transform=axk.transAxes)
        y -= step
    axk.text(0.0, 1.035, "Key numbers", fontsize=12, fontweight="bold", transform=axk.transAxes)
    axk.text(0.0, 0.0, "Takeaway: label-based MAs are behavioural\n"
             "arousals, and the KD arteriole deficit appears\n"
             "only there. Short detector events inside NREM\n"
             "look like small pupil fluctuations in sleep.",
             fontsize=9.5, va="bottom", transform=axk.transAxes,
             bbox=dict(boxstyle="round,pad=0.4", fc="#FFF8E1", ec="#FFB300", lw=.8))

    fig.text(0.985, 0.012, "Preliminary · unit = recording day · KD bins without manual labels filled by "
             "a pupil/whisker classifier", fontsize=7.5, color="0.45", ha="right")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"MA_methods_comparison.{ext}", dpi=200)
    print("wrote", OUT / "MA_methods_comparison.png")


if __name__ == "__main__":
    main()
