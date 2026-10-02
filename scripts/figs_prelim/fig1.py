"""Figure 1 — behavioural state classification + sleep architecture, WT only.

Panels (per AGENT_BRIEF.md / FIGURE_PLAN_text.txt, Figure 1 section):
  1C  example state-classification traces (pupil z, eye-speed z, whisker z,
      state ribbon) over an ~8-min window chosen programmatically for having
      the most state transitions among WT cycles that have sig['eye'],
      contain wake + NREM, and >=1 MA.
  1D  (own figure) classification decision table (text panel) + ML
      validation (per-day kappa bars + overall acc/balanced-acc/kappa),
      reading data/RESULTS_state_ml_2026-10-01/tables/cv_per_day.csv and
      cv_overall.csv for rule == 'smoothed_3bin'. No retraining.
  1E  stacked bar per mouse (day) of % time Awake/NREM/REM (Unclear and
      unlabelled bins excluded from the denominator) + group mean bar.
  1F  bout duration by state: one dot per mouse = median bout duration (s),
      individual bouts faint behind, log y-axis.
  1G  occurrence per hour per mouse: Awake bouts/h, NREM bouts/h, MA/h.
      REM-like events reported as a total count in text/notes only.
  1H  transition matrix (Awake / NREM / REM / MA as its own class): counts
      and row-normalised probabilities.

Outputs (data/RESULTS_figures_prelim_2026-10-01/):
  figures/fig1_main.png|pdf        (1C, 1E, 1F, 1G, 1H)
  figures/fig1_1D_logic_and_validation.png|pdf
  tables/fig1_example_window.csv
  tables/fig1_time_in_state.csv
  tables/fig1_bout_duration_mouse.csv
  tables/fig1_bout_duration_bouts.csv
  tables/fig1_occurrence_per_hour.csv
  tables/fig1_transition_matrix_counts.csv
  tables/fig1_transition_matrix_probs.csv
  tables/fig1_ml_validation_per_day.csv
  tables/fig1_ml_validation_overall.csv
  notes/fig1.md

Run (from repo root):
  ./venv/bin/python -B scripts/figs_prelim/fig1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.figs_prelim import common as C  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

GROUP = "WT"
ML_TABLE_DIR = C.ROOT / "data" / "RESULTS_state_ml_2026-10-01" / "tables"
RULE = "smoothed_3bin"
NOTES_PATH = C.OUT / "notes" / "fig1.md"

MA_COL = "#F9A825"  # distinct from STATE_COL REM to avoid confusion in 1H / ribbon marks
AROUSAL_MARK_COL = "#000000"


# --------------------------------------------------------------------------- #
# 1C — pick the example window programmatically
# --------------------------------------------------------------------------- #
def pick_example_window(keys):
    """Among WT cycles with sig['eye'], Awake+NREM present and >=1 MA, choose
    the (key, start_bin, n_bins) 5-8 min window with the most label
    transitions, requiring >=1 MA event inside the window and no missing
    labels inside it."""
    candidates = []
    for k in keys:
        cyc = C.load_cycle(k)
        if "eye" not in cyc.sig:
            continue
        labs = cyc.labels
        if "Awake" not in labs or "NREM" not in labs:
            continue
        ev = C.events(labs)
        ma = ev[ev.type == "MA"]
        if ma.empty:
            continue
        n = len(labs)
        for win_bins in (int(5 * 60 / C.BIN), int(6 * 60 / C.BIN),
                         int(7 * 60 / C.BIN), int(8 * 60 / C.BIN)):
            for s in range(0, n - win_bins + 1):
                w = labs[s:s + win_bins]
                if any(l is None for l in w):
                    continue
                t0, t1 = s * C.BIN, (s + win_bins) * C.BIN
                ma_in = ma[(ma.t0 >= t0) & (ma.t0 < t1)]
                if ma_in.empty:
                    continue
                if "Awake" not in w or "NREM" not in w:
                    continue
                trans = sum(1 for i in range(1, len(w)) if w[i] != w[i - 1])
                candidates.append((trans, k, s, win_bins, len(ma_in)))
    if not candidates:
        return None
    candidates.sort(key=lambda r: r[0], reverse=True)
    best = candidates[0]
    return dict(key=best[1], start_bin=best[2], n_bins=best[3],
                n_transitions=best[0], n_ma_in_window=best[4])


def plot_1c(ax_pupil, ax_eye, ax_whisk, ax_ribbon, win):
    key = win["key"]
    cyc = C.load_cycle(key)
    labs = cyc.labels
    t0 = win["start_bin"] * C.BIN
    t1 = (win["start_bin"] + win["n_bins"]) * C.BIN
    ev = C.event_table([key])
    ev_win = ev[(ev.t0 >= t0) & (ev.t0 < t1)]

    for ax, name in ((ax_pupil, "pupil"), (ax_eye, "eye"), (ax_whisk, "whisk")):
        t, y = cyc.sig[name]
        m = (t >= t0) & (t < t1)
        ax.plot(t[m] - t0, y[m], color=C.SIG_COL[name], lw=0.8)
        ax.set_ylabel(C.SIG_LABEL[name], fontsize=8)
        ax.tick_params(labelsize=7)
        C.shade_states(ax, labs, t_offset=t0, alpha=0.12)
        ax.set_xlim(0, t1 - t0)
    ax_pupil.set_xticklabels([])
    ax_eye.set_xticklabels([])

    # state ribbon
    bts = C.bouts(labs)
    bts = bts[(bts.t_start + bts.dur_s > t0) & (bts.t_start < t1)]
    for b in bts.itertuples():
        col = C.STATE_COL.get(b.state, "#FFFFFF")
        start = max(b.t_start, t0) - t0
        end = min(b.t_start + b.dur_s, t1) - t0
        ax_ribbon.axvspan(start, end, color=col, lw=0)
    ax_ribbon.set_xlim(0, t1 - t0)
    ax_ribbon.set_yticks([])
    ax_ribbon.set_ylim(0, 1)
    ax_ribbon.set_xlabel("time in window (s)", fontsize=8)
    ax_ribbon.tick_params(labelsize=7)

    # mark MA and N2W onsets
    for r in ev_win.itertuples():
        x = r.t0 - t0
        if r.type == "MA":
            ax_ribbon.axvline(x, color=AROUSAL_MARK_COL, lw=1.1, ls="--")
            ax_ribbon.annotate("MA", (x, 1.02), xycoords=("data", "axes fraction"),
                               fontsize=6.5, ha="center", color=AROUSAL_MARK_COL)
        elif r.type == "N2W":
            ax_ribbon.axvline(x, color="#1565C0", lw=1.1, ls=":")
            ax_ribbon.annotate("N2W", (x, 1.02), xycoords=("data", "axes fraction"),
                               fontsize=6.5, ha="center", color="#1565C0")

    from matplotlib.patches import Patch
    handles = [Patch(color=C.STATE_COL[s], label=s) for s in ("Awake", "NREM", "REM")]
    ax_ribbon.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.65),
                     fontsize=6.5, ncol=3, frameon=False)
    ax_pupil.set_title(f"1C  {key}  t={t0:.0f}-{t1:.0f}s  "
                       f"({win['n_transitions']} transitions, {win['n_ma_in_window']} MA)",
                       fontsize=8, loc="left")
    return ev_win


# --------------------------------------------------------------------------- #
# 1E — % time in state per mouse/day
# --------------------------------------------------------------------------- #
def time_in_state_table(keys):
    rows = []
    for k in keys:
        cyc = C.load_cycle(k)
        labs = cyc.labels
        valid = [l for l in labs if l is not None and l != "Unclear"]
        n_valid = len(valid)
        if n_valid == 0:
            continue
        for s in C.STATES:
            pct = 100.0 * sum(1 for l in valid if l == s) / n_valid
            rows.append(dict(key=k, mouse=cyc.info.mouse, group=cyc.info.group,
                             state=s, pct_time=pct, n_valid_bins=n_valid))
    df = pd.DataFrame(rows)
    # collapse cycles -> mouse/day (mean of cycle-level %, weighted isn't needed:
    # average within mouse as required by stats discipline)
    day = (df.groupby(["group", "mouse", "state"], as_index=False)
             .agg(pct_time=("pct_time", "mean"), n_cycles=("key", "nunique")))
    return df, day


def plot_1e(ax, day_df):
    mice = sorted(day_df.mouse.unique())
    states = list(C.STATES)
    bottoms = np.zeros(len(mice))
    for s in states:
        vals = np.array([day_df[(day_df.mouse == m) & (day_df.state == s)]
                         .pct_time.sum() for m in mice])
        ax.bar(np.arange(len(mice)), vals, bottom=bottoms, color=C.STATE_COL[s],
              label=s, width=0.7)
        bottoms += vals
    # group mean bar
    mean_vals = {s: day_df[day_df.state == s].pct_time.mean() for s in states}
    gx = len(mice) + 1.0
    bottom = 0
    for s in states:
        ax.bar([gx], [mean_vals[s]], bottom=bottom, color=C.STATE_COL[s], width=0.7)
        bottom += mean_vals[s]
    ax.axvline(len(mice) - 0.5 + 0.5, color="k", lw=0.6, ls=":")

    def _short(m):
        # e.g. "Extra_RestrosplenialCortex_2023.05.05" -> "RSC\n05.05"
        m2 = m.replace("Extra_", "")
        parts = m2.rsplit("_", 1)
        cohort = parts[0].replace("GCaMP8m", "GCaMP").replace("RestrosplenialCortex", "RSC")
        date = parts[1][5:] if len(parts) > 1 else ""  # drop year, keep MM.DD
        return f"{cohort}\n{date}"

    ax.set_xticks(list(range(len(mice))) + [gx])
    ax.set_xticklabels([_short(m) for m in mice] + ["WT\nmean"],
                       fontsize=6.5, rotation=0, ha="center")
    ax.set_xlim(-0.7, gx + 0.7)
    ax.set_ylabel("% scored time", fontsize=8)
    ax.set_ylim(0, 112)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.17),
             frameon=False, ncol=3)
    ax.set_title(f"1E  % time per state (n={len(mice)} days)", fontsize=8, loc="left", pad=18)


# --------------------------------------------------------------------------- #
# 1F — bout duration by state
# --------------------------------------------------------------------------- #
def bout_duration_tables(keys):
    rows = []
    for k in keys:
        cyc = C.load_cycle(k)
        b = C.bouts(cyc.labels)
        b = b[b.state.isin(C.STATES)].copy()
        b["key"] = k
        b["mouse"] = cyc.info.mouse
        b["group"] = cyc.info.group
        rows.append(b)
    bouts_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    mouse_df = (bouts_df.groupby(["group", "mouse", "state"], as_index=False)
               .agg(median_dur_s=("dur_s", "median"), n_bouts=("dur_s", "size")))
    return bouts_df, mouse_df


def plot_1f(ax, bouts_df, mouse_df):
    states = list(C.STATES)
    rng = np.random.default_rng(0)
    for i, s in enumerate(states):
        sub_b = bouts_df[bouts_df.state == s]
        if not sub_b.empty:
            jitter = rng.uniform(-0.12, 0.12, size=len(sub_b))
            ax.scatter(np.full(len(sub_b), i) + jitter, sub_b.dur_s,
                      color=C.STATE_COL[s], alpha=0.12, s=8, lw=0, zorder=1)
        sub_m = mouse_df[mouse_df.state == s]
        if not sub_m.empty:
            jitter = rng.uniform(-0.08, 0.08, size=len(sub_m))
            ax.scatter(np.full(len(sub_m), i) + jitter, sub_m.median_dur_s,
                      color=C.STATE_COL[s], edgecolor="black", linewidth=0.5,
                      s=38, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, fontsize=8)
    ax.set_ylabel("bout duration (s, log)", fontsize=8)
    ax.tick_params(labelsize=7)
    n_days = mouse_df.mouse.nunique()
    ax.set_title(f"1F  bout duration\n(dot=mouse median, n={n_days} d; REM descriptive)",
                fontsize=7.3, loc="left")


# --------------------------------------------------------------------------- #
# 1G — occurrence per hour per mouse
# --------------------------------------------------------------------------- #
def occurrence_table(keys):
    rows = []
    for k in keys:
        cyc = C.load_cycle(k)
        labs = cyc.labels
        n_valid = sum(1 for l in labs if l is not None)
        hours = (n_valid * C.BIN) / 3600.0
        if hours <= 0:
            continue
        b = C.bouts(labs)
        ev = C.events(labs)
        n_awake_bouts = (b.state == "Awake").sum()
        n_nrem_bouts = (b.state == "NREM").sum()
        n_ma = (ev.type == "MA").sum()
        rows.append(dict(key=k, mouse=cyc.info.mouse, group=cyc.info.group,
                         hours_scored=hours,
                         awake_bouts_per_h=n_awake_bouts / hours,
                         nrem_bouts_per_h=n_nrem_bouts / hours,
                         ma_per_h=n_ma / hours))
    df = pd.DataFrame(rows)
    day = (df.groupby(["group", "mouse"], as_index=False)
          .agg(hours_scored=("hours_scored", "sum"),
               awake_bouts_per_h=("awake_bouts_per_h", "mean"),
               nrem_bouts_per_h=("nrem_bouts_per_h", "mean"),
               ma_per_h=("ma_per_h", "mean")))
    return df, day


def plot_1g(ax, day_df):
    metrics = [("awake_bouts_per_h", "Awake\nbouts/h", "#1565C0"),
              ("nrem_bouts_per_h", "NREM\nbouts/h", "#C62828"),
              ("ma_per_h", "MA/h", "#F9A825")]
    rng = np.random.default_rng(1)
    for i, (col, label, color) in enumerate(metrics):
        vals = day_df[col].to_numpy()
        jitter = rng.uniform(-0.1, 0.1, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=color, s=30,
                  edgecolor="black", linewidth=0.4, zorder=3)
        ax.scatter([i], [np.nanmean(vals)], color=color, marker="_", s=400,
                  linewidth=2.2, zorder=4)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], fontsize=7.5)
    ax.set_xlim(-0.6, len(metrics) - 0.4)
    ax.set_ylabel("events / hour", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"1G  occurrence per mouse\n(n={len(day_df)} days)", fontsize=7.3, loc="left")


# --------------------------------------------------------------------------- #
# 1H — transition matrix (Awake / NREM / REM / MA as its own class)
# --------------------------------------------------------------------------- #
def transition_matrix(keys):
    """Build transitions between bouts, but re-labelling any Awake bout that
    satisfies the MA definition (1-3 bins, flanked by NREM>=2 bins each side)
    as its own class 'MA' before counting bout-to-bout transitions."""
    classes = ["Awake", "NREM", "REM", "MA"]
    counts = pd.DataFrame(0, index=classes, columns=classes, dtype=float)
    for k in keys:
        cyc = C.load_cycle(k)
        b = C.bouts(cyc.labels)
        b = b[b.state.isin(C.STATES)].reset_index(drop=True)
        if len(b) < 2:
            continue
        tagged = list(b.state)
        for i in range(len(b)):
            if b.state[i] != "Awake" or not (1 <= b.n_bins[i] <= 3):
                continue
            prev_ok = i > 0 and b.state[i - 1] == "NREM" and b.n_bins[i - 1] >= 2
            next_ok = i < len(b) - 1 and b.state[i + 1] == "NREM" and b.n_bins[i + 1] >= 2
            if prev_ok and next_ok:
                tagged[i] = "MA"
        for i in range(len(tagged) - 1):
            a, c = tagged[i], tagged[i + 1]
            if a in classes and c in classes:
                counts.loc[a, c] += 1
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    return counts, probs


def plot_1h(ax, probs, counts):
    classes = list(probs.index)
    M = probs.to_numpy()
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=7.5)
    ax.set_yticklabels(classes, fontsize=7.5)
    ax.set_xlabel("to", fontsize=8)
    ax.set_ylabel("from", fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            p = M[i, j]
            c = counts.to_numpy()[i, j]
            if np.isnan(p):
                continue
            txt = f"{p:.2f}\n(n={int(c)})"
            color = "white" if p > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=color)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label("row-normalised prob.", fontsize=7)
    ax.set_title("1H  bout-to-bout transition matrix", fontsize=8, loc="left")


# --------------------------------------------------------------------------- #
# 1D — classification logic text panel + ML validation
# --------------------------------------------------------------------------- #
def plot_1d_logic(ax):
    ax.axis("off")
    ax.set_title("1D  manual state-scoring logic (5-s bins)", fontsize=9, loc="left")
    lines = [
        "Each 5-s bin is scored manually from pupil, eye-movement and whisker",
        "video, without EEG/EMG — labels are behavioural correlates, hence the",
        "cautious terms 'NREM-like' and 'REM-like' used throughout.",
        "",
        "Awake        active whisking and/or large, frequent pupil/eye movements;",
        "             pupil often constricted or dynamic.",
        "NREM-like    whisking and eye movement minimal/absent; pupil large and",
        "             stable (behavioural quiescence).",
        "REM-like     NREM-like body quiescence (whisker low) but with phasic",
        "             pupil/eye movements and small pupil diameter.",
        "Unclear      ambiguous bin; excluded from % time, bout and event stats.",
        "",
        "Events (from the bout sequence of 5-s labels; common.py):",
        "  MA   micro-arousal: Awake bout of 1-3 bins (5-15 s) flanked by",
        "       NREM-like bouts of >=2 bins (>=10 s) on both sides.",
        "  N2W  NREM-to-wake: NREM-like bout >=6 bins (>=30 s) followed by an",
        "       Awake bout >=4 bins (>=20 s, i.e. longer than an MA).",
        "  W2N  wake-to-NREM (semi-control): Awake >=6 bins then NREM-like",
        "       >=6 bins.",
        "  REM  onset of any REM-like bout (descriptive only; rare).",
        "Bin resolution is 5 s, so onset timing is not interpretable below that.",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes, fontsize=7.6,
           va="top", ha="left", family="monospace")


def plot_1d_validation(ax_bar, ax_text):
    per_day = pd.read_csv(ML_TABLE_DIR / "cv_per_day.csv")
    overall = pd.read_csv(ML_TABLE_DIR / "cv_overall.csv")
    pd_r = per_day[per_day["rule"] == RULE].copy()
    ov_r = overall[(overall["rule"] == RULE) & (overall["scope"] == "all held-out days")]

    pd_r = pd_r.sort_values(["group", "day"])
    colors = [C.GROUP_COL.get(g, "#777777") for g in pd_r["group"]]
    x = np.arange(len(pd_r))
    ax_bar.bar(x, pd_r["kappa"], color=colors)
    ax_bar.set_xticks(x)
    short_labels = [d.split("_", 1)[-1].replace("Restrosplenial", "RSC_") for d in pd_r["day"]]
    ax_bar.set_xticklabels(short_labels, fontsize=6, rotation=90, ha="center")
    ax_bar.set_ylabel("Cohen's kappa", fontsize=8)
    ax_bar.axhline(0, color="k", lw=0.5)
    ax_bar.tick_params(labelsize=7)
    from matplotlib.patches import Patch
    groups_present = list(pd_r["group"].unique())
    handles = [Patch(color=C.GROUP_COL.get(g, "#777"), label=C.GROUP_LABEL.get(g, g))
              for g in groups_present]
    ax_bar.legend(handles=handles, fontsize=6.5, loc="lower right", frameon=False)

    if not ov_r.empty:
        r = ov_r.iloc[0]
        title = (f"ML validation (leave-one-day-out, rule='{RULE}'): "
                f"acc={r['accuracy']:.2f}, bal.acc={r['balanced_acc']:.2f}, "
                f"kappa={r['kappa']:.2f}  (n={int(r['n_bins'])} held-out bins)"
                if "n_bins" in r else
                f"ML validation (leave-one-day-out, rule='{RULE}'): "
                f"acc={r['accuracy']:.2f}, bal.acc={r['balanced_acc']:.2f}, "
                f"kappa={r['kappa']:.2f}")
    else:
        title = f"ML validation (rule='{RULE}'): overall row not found"
    ax_bar.set_title(title, fontsize=7.8, loc="left")

    ax_text.axis("off")
    note_lines = [
        "ML model: pupil+whisker features predicting manual labels, evaluated by",
        "leave-one-recording-day-out cross-validation (scripts/predict_states_ml.py).",
        "Used here only to validate that behavioural signals carry state information;",
        "predictions are not retrained and fill unlabelled IP3R2-KD bins elsewhere",
        f"(ML_EXCLUDED_DAYS = {sorted(C.ML_EXCLUDED_DAYS)}; never used for those days).",
    ]
    ax_text.text(0.0, 0.95, "\n".join(note_lines), transform=ax_text.transAxes,
               fontsize=7, va="top", ha="left")
    return pd_r, ov_r


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    C.ensure_dirs()
    keys = C.keys_for(GROUP)

    # ---- 1C ----
    win = pick_example_window(keys)
    if win is None:
        raise RuntimeError("No WT cycle with eye signal, wake+NREM and an MA found.")
    win_df = pd.DataFrame([dict(
        key=win["key"], start_s=win["start_bin"] * C.BIN,
        end_s=(win["start_bin"] + win["n_bins"]) * C.BIN,
        n_transitions=win["n_transitions"], n_ma_in_window=win["n_ma_in_window"])])
    C.save_table(win_df, "fig1_example_window")

    # ---- 1E ----
    time_cycle_df, time_day_df = time_in_state_table(keys)
    C.save_table(time_day_df, "fig1_time_in_state")

    # ---- 1F ----
    bouts_df, bout_mouse_df = bout_duration_tables(keys)
    C.save_table(bout_mouse_df, "fig1_bout_duration_mouse")
    C.save_table(bouts_df[["key", "mouse", "group", "state", "dur_s"]],
                "fig1_bout_duration_bouts")

    # ---- 1G ----
    occ_cycle_df, occ_day_df = occurrence_table(keys)
    C.save_table(occ_day_df, "fig1_occurrence_per_hour")

    # ---- REM total (descriptive) ----
    ev_all = C.event_table(keys)
    rem_total = int((ev_all.type == "REM").sum())

    # ---- 1H ----
    trans_counts, trans_probs = transition_matrix(keys)
    C.save_table(trans_counts.reset_index().rename(columns={"index": "from"}),
                "fig1_transition_matrix_counts")
    C.save_table(trans_probs.reset_index().rename(columns={"index": "from"}),
                "fig1_transition_matrix_probs")

    # =================== fig1_main (1C, 1E, 1F, 1G, 1H) ===================
    fig = plt.figure(figsize=(16, 11.5))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1.4],
                          width_ratios=[1.3, 1], hspace=0.9, wspace=0.35,
                          top=0.93, bottom=0.05)

    gs_left = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[:, 0],
                                               height_ratios=[1, 1, 1, 1.1], hspace=0.55)
    ax_pupil = fig.add_subplot(gs_left[0])
    ax_eye = fig.add_subplot(gs_left[1], sharex=ax_pupil)
    ax_whisk = fig.add_subplot(gs_left[2], sharex=ax_pupil)
    ax_ribbon = fig.add_subplot(gs_left[3], sharex=ax_pupil)
    plot_1c(ax_pupil, ax_eye, ax_whisk, ax_ribbon, win)

    gs_right = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[:, 1],
                                                height_ratios=[1.3, 1, 1], hspace=1.3, wspace=0.7)
    ax_1h = fig.add_subplot(gs_right[0, :])
    plot_1h(ax_1h, trans_probs, trans_counts)

    ax_1e = fig.add_subplot(gs_right[1, :])
    plot_1e(ax_1e, time_day_df)

    ax_1f = fig.add_subplot(gs_right[2, 0])
    ax_1g = fig.add_subplot(gs_right[2, 1])
    plot_1f(ax_1f, bouts_df, bout_mouse_df)
    plot_1g(ax_1g, occ_day_df)

    fig.suptitle(f"Figure 1. State classification & sleep architecture — WT (RSC)\n"
                f"{C.PRELIM_STAMP}", fontsize=9, y=1.01)
    C.save_fig(fig, "fig1_main")
    plt.close(fig)

    # =================== fig1_1D_logic_and_validation ===================
    fig2 = plt.figure(figsize=(12, 8.5))
    gs2 = gridspec.GridSpec(2, 2, figure=fig2, height_ratios=[1.5, 1],
                           width_ratios=[1.1, 1], hspace=0.9, wspace=0.4)
    ax_logic = fig2.add_subplot(gs2[:, 0])
    plot_1d_logic(ax_logic)
    ax_bar = fig2.add_subplot(gs2[0, 1])
    ax_text = fig2.add_subplot(gs2[1, 1])
    pd_r, ov_r = plot_1d_validation(ax_bar, ax_text)
    C.save_table(pd_r, "fig1_ml_validation_per_day")
    C.save_table(ov_r, "fig1_ml_validation_overall")
    fig2.suptitle(f"Figure 1D. Classification logic & ML validation\n{C.PRELIM_STAMP}",
                 fontsize=9, y=0.995)
    C.save_fig(fig2, "fig1_1D_logic_and_validation")
    plt.close(fig2)

    # =================== notes ===================
    mean_pct_nrem = time_day_df[time_day_df.state == "NREM"].pct_time.mean()
    median_awake_bout = bout_mouse_df[bout_mouse_df.state == "Awake"].median_dur_s.median()
    median_nrem_bout = bout_mouse_df[bout_mouse_df.state == "NREM"].median_dur_s.median()
    mean_ma_per_h = occ_day_df.ma_per_h.mean()
    n_days = len(set(C.load_cycle(k).info.mouse for k in keys))
    n_cyc = len(keys)

    notes = f"""# Figure 1 — state classification & sleep architecture (WT only)

Reproduce: `./venv/bin/python -B scripts/figs_prelim/fig1.py` (from repo root).

n = {n_days} WT recording days, {n_cyc} cycles ({C.n_note(pd.DataFrame([dict(group='WT', mouse=C.load_cycle(k).info.mouse, key=k) for k in keys]))}).

## Panels

- **1C** example traces: `{win['key']}`, window {win['start_bin']*C.BIN:.0f}-{(win['start_bin']+win['n_bins'])*C.BIN:.0f} s
  ({win['n_bins']*C.BIN/60:.1f} min), chosen as the window with the most label transitions
  ({win['n_transitions']}) among WT cycles that have `sig['eye']`, contain Awake+NREM, and
  >=1 MA inside the window ({win['n_ma_in_window']} MA onsets shown). Tracks: pupil z, eye-speed z,
  whisker z, plus a state ribbon with MA (black dashed) and N2W (blue dotted) onsets marked.
  No REM-like bin fell in this particular window (see 1G/REM note below).
- **1D** (own figure `fig1_1D_logic_and_validation`): left panel is a text decision table
  describing the 5-s manual scoring rule (pupil / eye / whisker) and the MA/N2W/W2N/REM
  event definitions from `common.py`, using cautious 'NREM-like'/'REM-like' language.
  Right panels: per-day kappa (leave-one-day-out ML cross-validation, rule='smoothed_3bin',
  colour = group, all 4 groups shown since the CV table isn't WT-only) and the overall
  acc/balanced-acc/kappa for 'all held-out days' in the panel title. Source: read directly
  from `data/RESULTS_state_ml_2026-10-01/tables/cv_per_day.csv` and `cv_overall.csv`; nothing
  retrained.
- **1E** % scored time (Unclear and unlabelled bins excluded from the denominator) in
  Awake/NREM/REM, one stacked bar per WT day plus a WT group-mean bar. Mean %NREM across
  WT days = **{mean_pct_nrem:.1f}%**.
- **1F** bout duration by state: dot = mouse-day median bout (s), individual bouts shown
  faint behind, log y-axis. Median (across days) of the per-day median bout duration:
  Awake = **{median_awake_bout:.1f} s**, NREM = **{median_nrem_bout:.1f} s**. REM bouts are
  shown descriptively only (few days have any REM-like bouts at all).
- **1G** occurrence/hour per WT day: Awake bouts/h, NREM bouts/h, MA/h (dots = days,
  bar = mean). Mean MA/h across WT days = **{mean_ma_per_h:.2f}**. REM-like events are not
  shown per-hour (too rare per day); see REM total below.
- **1H** bout-to-bout transition matrix among Awake / NREM / REM / **MA treated as its own
  class** (an Awake bout is re-labelled MA when it satisfies the MA definition — 1-3 bins
  flanked by >=2-bin NREM bouts — before transitions are counted). Counts and row-normalised
  probabilities shown in each cell.

## Headline numbers
- Mean % time NREM-like across WT days: **{mean_pct_nrem:.1f}%**
- Median Awake bout / NREM-like bout duration (day-level median of medians):
  **{median_awake_bout:.1f} s / {median_nrem_bout:.1f} s**
- Mean micro-arousals per hour (WT): **{mean_ma_per_h:.2f}/h**
- Total REM-like onsets across WT dataset (descriptive only): **{rem_total}**

## What could not be made / caveats
- REM-like bouts are rare in WT (only a few days contain any), so 1F/1G treat REM
  descriptively only, as instructed; no per-mouse REM bout-rate panel is shown.
- The example window in 1C has no REM-like bin; eye-motion signal (`sig['eye']`) is only
  available for 9/30 WT cycles (2026.03.03 and 2026.03.06 GCaMP8m sessions), so the window
  search was restricted to those 9 cycles as required by the brief.
- 1D's ML-validation bars include all 4 cohorts (WT/SCR/KD/NE) because `cv_per_day.csv` is
  not split by figure group; WT bars are coloured per `C.GROUP_COL['WT']` and are
  distinguishable via the legend.
- Per the statistical-discipline rules, 1E/1F/1G/1H all use the recording day ("mouse") as
  the unit; cycle- or bout-level points are shown faint/behind only, never as independent n.
- No `C.lmm` fit attempted for Figure 1 panels: these are purely descriptive sleep-architecture
  summaries (%time, bout duration, occurrence, transitions), not condition-vs-condition
  comparisons, so a fixed-effect LMM is not the natural tool; done in Figures 2/3 where state
  is a within-mouse covariate against a biological signal (will be requested from those agents).

## Files
Figures: `fig1_main.png/pdf`, `fig1_1D_logic_and_validation.png/pdf`
Tables: `fig1_example_window.csv`, `fig1_time_in_state.csv`, `fig1_bout_duration_mouse.csv`,
`fig1_bout_duration_bouts.csv`, `fig1_occurrence_per_hour.csv`,
`fig1_transition_matrix_counts.csv`, `fig1_transition_matrix_probs.csv`,
`fig1_ml_validation_per_day.csv`, `fig1_ml_validation_overall.csv`
"""
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text(notes)

    print("=== Figure 1 done ===")
    print(f"example window: {win}")
    print(f"mean % NREM (WT days): {mean_pct_nrem:.1f}%")
    print(f"median Awake / NREM bout (day-median): {median_awake_bout:.1f} / {median_nrem_bout:.1f} s")
    print(f"mean MA/h (WT): {mean_ma_per_h:.2f}")
    print(f"REM-like total count (WT): {rem_total}")


if __name__ == "__main__":
    main()
