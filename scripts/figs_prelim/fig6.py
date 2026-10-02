"""Figure 6 — IP3R2 KD vs scramble sleep architecture.

Groups: SCR (IP3R2 scramble, all manual labels) vs KD (IP3R2 KD; 12 of 19
cycles carry ML-filled labels from the pupil+whisker state model). Every
quantitative panel is run twice:
  primary    C.set_label_mode('manual+ml')  -> KD 6 days / 19 cyc
  sensitivity C.set_label_mode('manual')     -> KD 3 days / 7 cyc
SCR is manual-only in both modes (no ML applied to SCR). See
scripts/figs_prelim/AGENT_BRIEF.md and common.py for shared conventions.

Panels:
  6A/6B/6C  not possible in this repo (schematic / histology / startle) ->
            placeholder panel.
  6D        example scoring traces, one SCR + one KD (manual) cycle.
  6E        % time per state, stacked bar per mouse + group summary.
  6F        bout duration / bouts-per-hour (Awake, NREM) + MA/h.
  6G        motion/arousal controls (whisker>2 fraction, pupil z SD,
            MA frequency, N2W frequency).
  BH correction applied jointly across all 6E-6G group tests, per label mode.

Run: ./venv/bin/python -B -B scripts/figs_prelim/fig6.py   (from repo root)
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.figs_prelim import common as C

MODES = ("manual+ml", "manual")
MODE_LABEL = {"manual+ml": "manual+ML", "manual": "manual-only"}
MODE_MARKER = {"manual+ml": "o", "manual": "o"}  # filled vs hollow set via facecolor
GROUPS = ("SCR", "KD")


# --------------------------------------------------------------------------- #
# data collection helpers
# --------------------------------------------------------------------------- #
def cycle_frame(mode: str) -> pd.DataFrame:
    """One row per cycle: state fractions, bout/event summaries, motion stats."""
    C.set_label_mode(mode)
    rows = []
    for g in GROUPS:
        for k in C.keys_for(g):
            cyc = C.load_cycle(k)
            lab = np.asarray(cyc.labels, dtype=object)
            n = len(lab)
            if n == 0:
                continue
            dur_h = cyc.duration / 3600.0
            frac = {s: float(np.mean(lab == s)) for s in ("Awake", "NREM", "REM", "Unclear")}
            bt = C.bouts(lab)
            ev = C.events(lab)

            def bout_stats(state):
                d = bt.loc[bt.state == state, "dur_s"]
                return (float(d.median()) if len(d) else np.nan,
                        float(len(d)) / dur_h if dur_h > 0 else np.nan)

            awake_med, awake_rate = bout_stats("Awake")
            nrem_med, nrem_rate = bout_stats("NREM")
            ma_rate = float((ev.type == "MA").sum()) / dur_h if dur_h > 0 else np.nan
            n2w_rate = float((ev.type == "N2W").sum()) / dur_h if dur_h > 0 else np.nan

            whisk_hi = pupil_sd = np.nan
            if "whisk" in cyc.sig:
                _, wy = cyc.sig["whisk"]
                wy = wy[np.isfinite(wy)]
                if wy.size:
                    whisk_hi = float(np.mean(wy > 2))
            if "pupil" in cyc.sig:
                _, py = cyc.sig["pupil"]
                py = py[np.isfinite(py)]
                if py.size:
                    pupil_sd = float(np.std(py))

            rows.append(dict(
                key=k, mouse=cyc.info.mouse, group=cyc.info.group,
                date=cyc.info.date, cycle=cyc.info.cycle,
                label_source=C.label_source(k, mode), dur_h=dur_h,
                pct_awake=100 * frac["Awake"], pct_nrem=100 * frac["NREM"],
                pct_rem=100 * frac["REM"], pct_unclear=100 * frac["Unclear"],
                awake_bout_med_s=awake_med, nrem_bout_med_s=nrem_med,
                awake_bouts_per_h=awake_rate, nrem_bouts_per_h=nrem_rate,
                ma_per_h=ma_rate, n2w_per_h=n2w_rate,
                whisk_frac_hi=whisk_hi, pupil_z_sd=pupil_sd,
            ))
    df = pd.DataFrame(rows)
    df["mode"] = mode
    return df


def mouse_frame(cyc_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per mouse (recording day); carries ml_any flag."""
    value_cols = ["pct_awake", "pct_nrem", "pct_rem", "pct_unclear",
                  "awake_bout_med_s", "nrem_bout_med_s", "awake_bouts_per_h",
                  "nrem_bouts_per_h", "ma_per_h", "n2w_per_h",
                  "whisk_frac_hi", "pupil_z_sd"]
    g = cyc_df.groupby(["group", "mouse"], as_index=False)[value_cols].mean()
    ml_any = cyc_df.groupby(["group", "mouse"])["label_source"].apply(
        lambda s: bool((s != "manual").any())).reset_index(name="ml_any")
    n_cyc = cyc_df.groupby(["group", "mouse"]).size().reset_index(name="n_cyc")
    g = g.merge(ml_any, on=["group", "mouse"]).merge(n_cyc, on=["group", "mouse"])
    return g


# --------------------------------------------------------------------------- #
# 6A/6B/6C placeholder
# --------------------------------------------------------------------------- #
def panel_placeholder():
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.axis("off")
    msg = (
        "Fig. 6A — viral/genetic strategy schematic: NOT POSSIBLE\n"
        "   (no schematic asset in this repo; needs AAV construct / shRNA / timeline diagram)\n\n"
        "Fig. 6B — expression / histology image: NOT POSSIBLE\n"
        "   (no histology or imaging-region images in this repo)\n\n"
        "Fig. 6C — startle-evoked astrocyte Ca2+ validation: NOT POSSIBLE\n"
        "   (no startle-stimulus recordings exist anywhere in data/; only spontaneous\n"
        "    sleep/wake cycles are present, so scramble-vs-KD evoked Ca2+ responses\n"
        "    to a startle cannot be computed from available data)"
    )
    ax.text(0.02, 0.98, msg, va="top", ha="left", fontsize=10.5, family="monospace",
            transform=ax.transAxes)
    ax.set_title("Figure 6A–6C — not reproducible from repo data", fontsize=12, loc="left")
    fig.text(0.01, 0.01, C.PRELIM_STAMP, fontsize=6, color="gray")
    return fig


# --------------------------------------------------------------------------- #
# 6D example traces
# --------------------------------------------------------------------------- #
def pick_example(group: str, prefer_manual: bool) -> str | None:
    keys = C.keys_for(group)
    if not keys:
        return None
    if prefer_manual:
        man = [k for k in keys if C.label_source(k) == "manual"]
        pool = man if man else keys
    else:
        pool = keys
    # prefer a cycle with a decent mix of states and longest coverage
    best, best_score = None, -1
    for k in pool:
        cyc = C.load_cycle(k)
        lab = np.asarray(cyc.labels, dtype=object)
        if lab.size == 0 or "pupil" not in cyc.sig or "whisk" not in cyc.sig:
            continue
        n_states = len(set(x for x in lab if x is not None))
        cov = np.mean([x is not None for x in lab])
        score = n_states + cov
        if score > best_score:
            best, best_score = k, score
    return best or pool[0]


def panel_6D():
    C.set_label_mode("manual+ml")  # irrelevant for SCR; for KD we force manual example below
    scr_key = pick_example("SCR", prefer_manual=True)
    kd_key = pick_example("KD", prefer_manual=True)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for row, (grp, key) in enumerate([("SCR", scr_key), ("KD", kd_key)]):
        cyc = C.load_cycle(key)
        lab = np.asarray(cyc.labels, dtype=object)
        src = C.label_source(key)
        axp, axw, axs = axes[row]
        for ax, sig, name in ((axp, "pupil", "pupil"), (axw, "whisk", "whisker")):
            if sig in cyc.sig:
                t, y = cyc.sig[sig]
                ax.plot(t, y, color=C.SIG_COL[sig], lw=0.6)
                C.shade_states(ax, lab, alpha=0.15)
                ax.set_ylabel(C.SIG_LABEL[sig])
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(0, cyc.duration)
        # state ribbon
        for b in C.bouts(lab).itertuples():
            if b.state is None:
                continue
            axs.axhspan(0, 1, xmin=b.t_start / cyc.duration,
                       xmax=(b.t_start + b.dur_s) / cyc.duration,
                       color=C.STATE_COL.get(b.state, "#DDDDDD"), lw=0)
        axs.set_yticks([])
        axs.set_xlabel("time (s)")
        axs.set_xlim(0, cyc.duration)
        axp.set_title(f"{C.GROUP_LABEL[grp]} — {key}  [{src}]", loc="left", fontsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=C.STATE_COL[s]) for s in C.STATES]
    fig.legend(handles, C.STATES, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Fig. 6D — example scoring traces (pupil z, whisker z, state ribbon)\n"
                 f"KD example label source: {C.label_source(kd_key)} (preferred manual)",
                 fontsize=11)
    fig.text(0.01, -0.06, C.PRELIM_STAMP, fontsize=6, color="gray")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    return fig


# --------------------------------------------------------------------------- #
# 6E: % time per state
# --------------------------------------------------------------------------- #
def panel_6E(cyc_frames: dict, mouse_frames: dict):
    fig, axes = plt.subplots(1, len(MODES), figsize=(13, 5.2), sharey=True)
    stats_rows = []
    for mi, mode in enumerate(MODES):
        ax = axes[mi]
        md = mouse_frames[mode].copy()
        md = md.sort_values(["group", "mouse"]).reset_index(drop=True)
        states = ["pct_awake", "pct_nrem", "pct_rem", "pct_unclear"]
        state_names = ["Awake", "NREM", "REM", "Unclear"]
        colors = [C.STATE_COL["Awake"], C.STATE_COL["NREM"], C.STATE_COL["REM"], C.STATE_COL["Unclear"]]
        x = np.arange(len(md))
        bottom = np.zeros(len(md))
        for col, name, color in zip(states, state_names, colors):
            vals = md[col].to_numpy()
            ax.bar(x, vals, bottom=bottom, color=color, label=name, width=0.7,
                  edgecolor="white", linewidth=0.5)
            bottom += vals
        xt_labels = [f"{m}{'*' if ml else ''}" for m, ml in zip(md["mouse"].str.slice(-10), md["ml_any"])]
        ax.set_xticks(x)
        ax.set_xticklabels(xt_labels, rotation=60, ha="right", fontsize=7)
        for xi, grp in zip(x, md["group"]):
            ax.text(xi, 102, grp, ha="center", fontsize=7, color=C.GROUP_COL[grp], fontweight="bold")
        ax.set_ylim(0, 108)
        ax.set_title(f"{MODE_LABEL[mode]}  (* = ML-filled day)", fontsize=10)
        if mi == 0:
            ax.set_ylabel("% of cycle time")
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)

        # group summary stats (mouse-level) for % awake, % nrem
        for value, label in (("pct_awake", "%Awake"), ("pct_nrem", "%NREM")):
            a = md.loc[md.group == "SCR", value]
            b = md.loc[md.group == "KD", value]
            r = C.group_test(a, b)
            r.update(mode=mode, metric=label)
            stats_rows.append(r)

    fig.suptitle("Fig. 6E — % time per state per mouse (SCR vs KD), both label modes", fontsize=12)
    fig.text(0.01, 0.01, C.PRELIM_STAMP, fontsize=6, color="gray")
    fig.text(0.01, -0.02, C.n_note(pd.concat(cyc_frames.values())), fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.02, 0.88, 0.93))
    return fig, pd.DataFrame(stats_rows)


# --------------------------------------------------------------------------- #
# 6F bout duration / rates and 6G motion controls
# --------------------------------------------------------------------------- #
def _dot_panel(ax, mouse_df, value, title, ylabel, cyc_df=None, cyc_value=None):
    """Mouse-level dots (prominent, filled=manual-only mouse, hollow if ML used),
    with faint cycle-level points behind if provided."""
    order = {"SCR": 0, "KD": 1}
    jitter = {"SCR": -0.12, "KD": 0.12}
    if cyc_df is not None and cyc_value is not None:
        for _, r in cyc_df.iterrows():
            x = order[r.group] + jitter[r.group] + np.random.uniform(-0.04, 0.04)
            ax.scatter(x, r[cyc_value], s=10, color=C.GROUP_COL[r.group], alpha=0.18, zorder=1)
    for _, r in mouse_df.iterrows():
        x = order[r.group] + jitter[r.group]
        face = C.GROUP_COL[r.group] if not r.ml_any else "white"
        ax.scatter(x, r[value], s=70, edgecolor=C.GROUP_COL[r.group], facecolor=face,
                  linewidth=1.6, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SCR", "KD"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_title(title, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9)


def panel_6FG(cyc_frames: dict, mouse_frames: dict):
    specs_F = [
        ("awake_bout_med_s", "Awake bout duration\n(mouse median, s)", "s"),
        ("nrem_bout_med_s", "NREM bout duration\n(mouse median, s)", "s"),
        ("awake_bouts_per_h", "Awake bouts / h", "bouts/h"),
        ("nrem_bouts_per_h", "NREM bouts / h", "bouts/h"),
        ("ma_per_h", "Micro-arousals / h", "MA/h"),
    ]
    specs_G = [
        ("whisk_frac_hi", "Frac. time whisker z>2", "fraction"),
        ("pupil_z_sd", "Pupil z SD", "z SD"),
        ("ma_per_h", "MA frequency / h", "MA/h"),
        ("n2w_per_h", "N2W frequency / h", "N2W/h"),
    ]
    n_rows = 2
    n_cols = max(len(specs_F), len(specs_G))
    fig, axes = plt.subplots(n_rows * len(MODES), n_cols, figsize=(3.1 * n_cols, 3.0 * n_rows * len(MODES)))

    stats_rows = []
    lmm_frames = []
    for mi, mode in enumerate(MODES):
        md = mouse_frames[mode]
        cd = cyc_frames[mode]
        for ri, (specs, label) in enumerate([(specs_F, "6F"), (specs_G, "6G")]):
            row_idx = mi * n_rows + ri
            for ci in range(n_cols):
                ax = axes[row_idx, ci]
                if ci >= len(specs):
                    ax.axis("off")
                    continue
                value, title, ylabel = specs[ci]
                _dot_panel(ax, md, value, f"[{label}] {title}\n{MODE_LABEL[mode]}", ylabel,
                          cyc_df=cd, cyc_value=value if value in cd.columns else None)
                a = md.loc[md.group == "SCR", value]
                b = md.loc[md.group == "KD", value]
                r = C.group_test(a, b)
                r.update(mode=mode, panel=label, metric=value)
                stats_rows.append(r)
                # cycle-level LMM: value ~ C(group), random intercept mouse
                sub = cd[["group", "mouse", value]].dropna().rename(columns={value: "value"})
                lm = C.lmm(sub, "value", "C(group)")
                if lm is not None:
                    lm = lm.copy()
                    lm["mode"], lm["panel"], lm["metric"] = mode, label, value
                    lmm_frames.append(lm)
                else:
                    lmm_frames.append(pd.DataFrame([dict(
                        term="NOT_FITTED", mode=mode, panel=label, metric=value,
                        note="too few mice/obs")]))

    fig.suptitle("Fig. 6F-G — bout architecture (6F) and motion/arousal controls (6G), "
                 "SCR vs KD, both label modes\nFilled dot = all-manual day; hollow dot = "
                 "ML-filled day; faint dots = cycle-level", fontsize=11)
    fig.text(0.01, 0.005, C.PRELIM_STAMP, fontsize=6, color="gray")
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    stats_df = pd.DataFrame(stats_rows)
    lmm_df = pd.concat(lmm_frames, ignore_index=True) if lmm_frames else pd.DataFrame()
    return fig, stats_df, lmm_df


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    np.random.seed(0)
    C.ensure_dirs()

    cyc_frames, mouse_frames = {}, {}
    for mode in MODES:
        cd = cycle_frame(mode)
        cyc_frames[mode] = cd
        mouse_frames[mode] = mouse_frame(cd)
    C.set_label_mode("manual+ml")  # restore default global state

    # --- placeholder (6A-6C) ---
    fig_ph = panel_placeholder()
    C.save_fig(fig_ph, "fig6_ABC_placeholder")
    plt.close(fig_ph)

    # --- 6D ---
    fig_d = panel_6D()
    C.save_fig(fig_d, "fig6_D_examples")
    plt.close(fig_d)

    # --- 6E ---
    fig_e, stats_e = panel_6E(cyc_frames, mouse_frames)
    C.save_fig(fig_e, "fig6_E-G_architecture_E")
    plt.close(fig_e)

    # --- 6F/6G combined figure (per brief: fig6_E-G_architecture) ---
    fig_fg, stats_fg, lmm_fg = panel_6FG(cyc_frames, mouse_frames)
    C.save_fig(fig_fg, "fig6_E-G_architecture")
    plt.close(fig_fg)

    # --- combine stats, BH across 6E-6G group tests, per mode ---
    stats_all = pd.concat([stats_e.assign(panel="6E"), stats_fg], ignore_index=True)
    stats_all["q_bh"] = np.nan
    for mode in MODES:
        m = stats_all["mode"] == mode
        stats_all.loc[m, "q_bh"] = C.bh(stats_all.loc[m, "welch_p"].to_numpy())
    C.save_table(stats_all, "fig6_group_tests_6E-G")
    C.save_table(lmm_fg, "fig6_lmm_6F-G")

    # per-mouse and per-cycle tables for transparency
    for mode in MODES:
        tag = mode.replace("+", "")
        C.save_table(cyc_frames[mode], f"fig6_cycle_table_{tag}")
        C.save_table(mouse_frames[mode], f"fig6_mouse_table_{tag}")

    write_notes(cyc_frames, mouse_frames, stats_all, lmm_fg)
    print_summary(stats_all)


def write_notes(cyc_frames, mouse_frames, stats_all, lmm_fg):
    md = mouse_frames["manual+ml"]
    md_man = mouse_frames["manual"]

    def fmt_group_test(df, metric, mode):
        r = df[(df.metric == metric) & (df["mode"] == mode)]
        if r.empty:
            return "n/a"
        r = r.iloc[0]
        return (f"SCR {r.mean_a:.1f} vs KD {r.mean_b:.1f}, Hedges g={r.hedges_g:.2f}, "
                f"Welch p={r.welch_p:.3f}, MW p={r.mw_p:.3f}, q(BH)={r.q_bh:.3f} "
                f"(n_SCR={r.n_a}, n_KD={r.n_b} days)")

    lines = []
    lines.append("# Figure 6 — IP3R2 KD vs scramble sleep architecture\n")
    lines.append(
        "**Label modes.** Every quantitative panel was run in both "
        "`manual+ml` (primary; KD 6 days/19 cyc, 12 cyc ML-filled) and "
        "`manual` (sensitivity; KD 3 days/7 cyc). SCR is manual-only in both "
        "(n=4 days/16 cyc). ML day `IP3R2_KD_2023.08.26` is excluded from "
        "predictions (held-out kappa 0.07) and contributes manual labels only "
        "in both modes. The other ML-filled KD day pairs: "
        "`IP3R2_KD_2023.08.25` manual (kappa ~0.60 in cv_per_day.csv for a "
        "comparable KD day), `IP3R2_KD_2024.02.18`, `IP3R2_KD_2024.06.01`, "
        "`IP3R2_KD_2025.10.25`, `IP3R2_KD_2025.11.01` ML-filled/partial.\n")
    lines.append(
        "## Panels\n"
        "- **6A/6B/6C — NOT POSSIBLE** (`fig6_ABC_placeholder.png`): 6A (viral/genetic "
        "strategy schematic) and 6B (expression/histology image) require schematic "
        "and histology assets that do not exist in this repo. 6C (startle-evoked "
        "astrocyte Ca2+ validation) requires startle-stimulus recordings; only "
        "spontaneous sleep/wake cycles exist in `data/`, so no startle-response "
        "comparison can be computed.\n"
        "- **6D** (`fig6_D_examples.png`): one example SCR cycle and one example KD "
        "cycle (KD chosen from manually-labelled days where possible), showing pupil "
        "z, whisker z and the state ribbon (Awake/NREM/REM/Unclear).\n"
        "- **6E** (`fig6_E-G_architecture_E.png`): % time Awake/NREM/REM/Unclear, "
        "stacked bar per mouse (day), both label modes side by side; KD days with any "
        "ML-filled bins are flagged with `*` on the x-axis.\n"
        "- **6F/6G** (`fig6_E-G_architecture.png`): bout duration (mouse median), "
        "bouts/h for Awake and NREM, MA/h (6F) and motion/arousal controls — fraction "
        "of time whisker z>2, pupil z SD, MA/h, N2W/h (6G). Filled markers = day is "
        "all-manual; hollow markers = day has ML-filled cycles. Faint dots behind = "
        "cycle-level values.\n")
    lines.append(
        "## Statistics\n"
        "`C.group_test` (Welch t + Mann-Whitney + Hedges g, SCR=a vs KD=b) computed "
        "per mouse-level metric, per label mode, for every 6E-6G measure. "
        "Benjamini-Hochberg correction applied jointly across all 6E-6G tests "
        "**within each label mode** (`fig6_group_tests_6E-G.csv`, column `q_bh`). "
        "A cycle-level linear mixed model `value ~ C(group)` with a random "
        "intercept per mouse was also fit for every 6F/6G metric "
        "(`fig6_lmm_6F-G.csv`); rows marked `NOT_FITTED` indicate too few mice/obs "
        "to fit (expected for the manual-only KD sensitivity analysis, n=3 KD days, "
        "n=4 SCR days = 7 total mice, borderline for `statsmodels` mixedlm).\n")

    lines.append("## Headline numbers (mouse-level, SCR vs KD)\n")
    for mode in MODES:
        lines.append(f"### {MODE_LABEL[mode]}\n")
        lines.append(f"- %NREM: {fmt_group_test(stats_all, '%NREM', mode)}")
        lines.append(f"- %Awake: {fmt_group_test(stats_all, '%Awake', mode)}")
        lines.append(f"- Awake bout duration (median, s): {fmt_group_test(stats_all, 'awake_bout_med_s', mode)}")
        lines.append(f"- NREM bout duration (median, s): {fmt_group_test(stats_all, 'nrem_bout_med_s', mode)}")
        lines.append(f"- MA/h: {fmt_group_test(stats_all, 'ma_per_h', mode)}")
        lines.append("")

    lines.append(
        "## Caveats\n"
        "- Unit = recording day (`mouse`); n is small (SCR 4 days, KD 6 or 3 days "
        "depending on mode) — Wilcoxon/Mann-Whitney cannot reach small p at this n "
        "(`wilcoxon_min_p` logic in common.py applies to paired tests; here "
        "Mann-Whitney has an analogous floor).\n"
        "- 12 of 19 KD cycles (across 4 of 6 ML-eligible days) rely on ML-predicted "
        "labels from a pupil+whisker model; held-out validation kappa was ~0.56-0.67 "
        "on comparable days (see `cv_per_day.csv`) but was not independently measured "
        "on the specific ML-filled days used here (no manual ground truth exists for "
        "them, by construction) — one day with known-bad validation (kappa 0.07, "
        "`IP3R2_KD_2023.08.26`) is excluded from ML and keeps only its manual labels "
        "in all analyses.\n"
        "- Because the ML-filled days have NO manual labels at all (0 manual bins in "
        "`predicted_cycles.csv` for 2024.02.18 / 2024.06.01 / 2025.10.25, and partial "
        "for 2025.11.01), the 'manual' sensitivity mode is NOT simply a subset of the "
        "same days with fewer bins — it drops those days from KD entirely, so KD n "
        "drops from 6 to 3 days between modes. Any difference between modes could "
        "reflect either (a) label-mode effects, or (b) which specific days are "
        "included; this figure cannot separate the two.\n"
        "- mean whisker z is not reported (meaningless after within-cycle z, per "
        "brief); whisker-based motion is instead summarised as fraction of time "
        "whisker z>2 (6G).\n"
        "- REM-like bouts are rare in both groups; REM% is shown descriptively in 6E "
        "only and not separately tested.\n"
        "- `IP3R2_KD_2025.11.01` has very few manual bins (29 of 180 for cycle 9; "
        "cycles 10/11 are 0 manual) per `predicted_cycles.csv`, and `cv_per_day.csv` "
        "shows this day's raw-rule kappa is undefined (NaN) at n_bins=4 in one row — "
        "treat its manual-mode contribution as thin.\n")
    lines.append(
        "## Reproduce\n"
        "`./venv/bin/python -B scripts/figs_prelim/fig6.py` from the repo root.\n")

    (C.OUT / "notes" / "fig6.md").parent.mkdir(parents=True, exist_ok=True)
    (C.OUT / "notes" / "fig6.md").write_text("\n".join(lines))


def print_summary(stats_all):
    print("\n=== Figure 6 summary ===")
    for mode in MODES:
        print(f"\n--- {mode} ---")
        sub = stats_all[stats_all["mode"] == mode]
        print(sub[["panel", "metric", "mean_a", "mean_b", "hedges_g", "welch_p", "q_bh"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
