"""Wake-up (N2W) and micro-arousal (MA) arteriole responses per group, on the
2026-10-08 labels (all KD + scramble manually labelled; GCaMP8m re-labelled).

Run with:  PRELIM_LABEL_SET=relabel_2026-10-08 ./venv/bin/python -B scripts/n2w_groups_relabel.py

Answers:
  1. Is there an arteriole dilation at wake-up in WT, scramble, KD (and NE)?
     Per event: arteriole AUC 0-15 s (baseline -5..0 s) and level change
     mean(+10..+40) - mean(-30..-10) (% of cycle median). Per day: mean event
     value MINUS the mean of the same measure at random NREM moments of that day
     (>= 20 s from any event label onset, NREM from -10 to +5 s; i.e. a moment
     inside sleep where nothing was labelled to happen). That
     surrogate removes drift / slow-oscillation bias. Test: day-level
     differences vs 0 (one-sample t and Wilcoxon), unit = recording day.
  2. KD vs scramble: Welch t on the day-level, surrogate-corrected values.
  3. Same for label MAs (method 1) - the random-NREM test still owed.
  4. Chronology (Grant): every day's wake-up response against recording date.
Manual labels only. Outputs: data/RESULTS_groups_relabel_2026-10-08/
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

sys.path.insert(0, str(Path(__file__).resolve().parent / "figs_prelim"))
import common as C  # noqa: E402

OUT = C.ROOT / "data" / ("RESULTS_groups_relabel_" + C.LABEL_SET.split("_", 1)[1])   # one folder per label set
GROUPS = ("WT", "SCR", "KD", "NE")
N_SURR = 200
RNG = np.random.default_rng(20261008)
PRE, POST = (-30.0, -10.0), (10.0, 40.0)


def measures(t, y, t0s):
    """AUC 0-15 s (baseline -5..0) and level change, for a vector of onsets."""
    auc, lvl = [], []
    for t0 in t0s:
        b = (t >= t0 - 5) & (t < t0)
        r = (t >= t0) & (t <= t0 + 15)
        a = (t >= t0 + PRE[0]) & (t < t0 + PRE[1])
        p = (t > t0 + POST[0]) & (t <= t0 + POST[1])
        ok = b.sum() > 3 and r.sum() > 10 and np.isfinite(y[b]).any() and np.isfinite(y[r]).any()
        auc.append(np.trapz(np.nan_to_num(y[r] - np.nanmean(y[b])), t[r]) if ok else np.nan)
        ok = np.isfinite(y[a]).sum() > 10 and np.isfinite(y[p]).sum() > 10
        lvl.append(100 * (np.nanmean(y[p]) - np.nanmean(y[a])) if ok else np.nan)
    return np.array(auc), np.array(lvl)


def surrogate_t0s(key, all_t0, n):
    lab = C.load_labels(key)
    dur = len(lab) * C.BIN
    cand = []
    for t0 in RNG.uniform(35, dur - 45, n * 20):
        b0, b1 = int((t0 - 10) // C.BIN), int((t0 + 5) // C.BIN)
        if b0 < 0 or b1 >= len(lab) or any(lab[i] != "NREM" for i in range(b0, b1 + 1)):
            continue
        if all_t0.size and np.min(np.abs(all_t0 - t0)) < 20:
            continue
        cand.append(t0)
        if len(cand) >= n:
            break
    return np.array(cand)


def day_rows(ev_all):
    """Per day and event type: real mean, surrogate mean, difference."""
    rows, events = [], []
    for day, d in ev_all.groupby("mouse"):
        for typ in ("N2W", "MA"):
            e = d[d["type"] == typ]
            real_auc, real_lvl, s_auc, s_lvl, ca_auc = [], [], [], [], []
            for key, ek in e.groupby("key"):
                cyc = C.load_cycle(key)
                if "art" not in cyc.sig:
                    continue
                t, y = cyc.sig["art"]
                a, lv = measures(t, y, ek["t0"].to_numpy())
                real_auc += list(a); real_lvl += list(lv)
                if "ca" in cyc.sig:
                    tc, yc = cyc.sig["ca"]
                    ca_auc += list(measures(tc, yc, ek["t0"].to_numpy())[0])
                for t0, aa, ll in zip(ek["t0"], a, lv):
                    events.append(dict(day=day, group=cyc.info.group, type=typ, key=key, t0=t0,
                                       art_auc=aa, art_level=ll))
            if not real_auc:
                continue
            # surrogate pool from every labelled cycle of the day (same weighting per cycle)
            for key in d["key"].unique():
                cyc = C.load_cycle(key)
                if "art" not in cyc.sig:
                    continue
                t, y = cyc.sig["art"]
                st = surrogate_t0s(key, ev_all.loc[ev_all.key == key, "t0_label"].to_numpy(), N_SURR // 4)
                a, lv = measures(t, y, st)
                s_auc += list(a); s_lvl += list(lv)
            info = C.parse_key(d["key"].iloc[0])
            rows.append(dict(day=day, group=info.group, date=info.date, type=typ, n_events=len(real_auc),
                             n_surr=int(np.isfinite(s_auc).sum()),
                             art_auc=np.nanmean(real_auc), art_auc_surr=np.nanmean(s_auc),
                             art_level=np.nanmedian(real_lvl), art_level_surr=np.nanmedian(s_lvl),
                             frac_down=np.mean(np.array(real_lvl)[np.isfinite(real_lvl)] < 0),
                             ca_auc=np.nanmean(ca_auc) if ca_auc else np.nan))
    D = pd.DataFrame(rows)
    D["art_auc_corr"] = D["art_auc"] - D["art_auc_surr"]
    D["art_level_corr"] = D["art_level"] - D["art_level_surr"]
    return D, pd.DataFrame(events)


def one_sample(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return dict(n_days=len(x), mean=np.mean(x) if len(x) else np.nan, t_p=np.nan, wilcoxon_p=np.nan,
                    days_positive=int((x > 0).sum()))
    return dict(n_days=len(x), mean=x.mean(), sem=x.std(ddof=1) / np.sqrt(len(x)),
                t_p=stats.ttest_1samp(x, 0).pvalue, wilcoxon_p=stats.wilcoxon(x).pvalue,
                days_positive=int((x > 0).sum()))


def main():
    C.set_label_mode("manual")
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    keys = C.keys_for(list(GROUPS))
    ev = C.response_events(keys)
    ev = ev[ev["type"].isin(["N2W", "MA"])]
    D, E = day_rows(ev)
    D.to_csv(OUT / "per_day.csv", index=False)
    E.to_csv(OUT / "events.csv", index=False)

    # 1 + 3: response vs 0 (surrogate-corrected) per group and type
    T1 = []
    for (g, typ), d in D.groupby(["group", "type"]):
        for m in ("art_auc_corr", "art_level_corr", "ca_auc"):
            T1.append(dict(group=g, type=typ, metric=m, **one_sample(d[m])))
    T1 = pd.DataFrame(T1)
    T1["q_bh"] = np.nan
    m = T1["t_p"].notna()
    T1.loc[m, "q_bh"] = C.bh(T1.loc[m, "t_p"].to_numpy())
    T1.to_csv(OUT / "response_vs_zero.csv", index=False)

    # 2: KD vs scramble
    T2 = []
    for typ in ("N2W", "MA"):
        for mtr in ("art_auc_corr", "art_level_corr", "ca_auc", "n_events"):
            a = D[(D.group == "SCR") & (D.type == typ)][mtr].dropna()
            b = D[(D.group == "KD") & (D.type == typ)][mtr].dropna()
            if len(a) >= 2 and len(b) >= 2:
                r = C.group_test(a.to_numpy(), b.to_numpy())
                T2.append(dict(type=typ, metric=mtr, scr_days=len(a), kd_days=len(b),
                               scr_mean=a.mean(), kd_mean=b.mean(), **{k: r[k] for k in ("hedges_g", "welch_p", "mw_p")}))
    T2 = pd.DataFrame(T2)
    T2.to_csv(OUT / "kd_vs_scr.csv", index=False)

    # 4: chronology figure
    d = D[D.type == "N2W"].copy()
    d["date_dt"] = pd.to_datetime(d["date"], format="%Y.%m.%d")
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True)
    for ax, mtr, lab in [(axs[0], "art_level_corr", "wake-up arteriole level change\n(% of median, minus random NREM)"),
                         (axs[1], "art_auc_corr", "wake-up arteriole AUC 0-15 s\n(minus random NREM)")]:
        for g, dd in d.groupby("group"):
            ax.scatter(dd["date_dt"], dd[mtr], s=20 + 3 * dd["n_events"], color=C.GROUP_COL[g],
                       label=f"{C.GROUP_LABEL[g]} ({len(dd)} days)", alpha=0.85, edgecolor="k", lw=0.4)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_ylabel(lab); ax.legend(fontsize=8, frameon=False)
        for r in d.itertuples():
            if r.group == "WT" and r.date.startswith("2026.03.03"):
                ax.annotate("03.03", (r.date_dt, getattr(r, mtr)), fontsize=7, xytext=(4, -8),
                            textcoords="offset points")
    fig.suptitle(f"Wake-up arteriole response by recording date (all groups, labels {C.LABEL_SET}; dot size = events)",
                 fontsize=11, x=0.01, ha="left")
    fig.savefig(OUT / "figures" / "chronology_n2w.png", dpi=150)
    plt.close(fig)

    # 4b: chronology test. Pre-specified: Spearman of day value vs date (Grant's
    # hypothesis of a change over time). The early/late split dates are post hoc
    # and reported only descriptively.
    CT = []
    for subset, dd in [("all groups", d), ("excluding KD", d[d.group != "KD"])]:
        for mtr in ("art_auc_corr", "art_level_corr"):
            r, pv = stats.spearmanr(dd["date_dt"].astype("int64"), dd[mtr])
            row = dict(subset=subset, metric=mtr, n_days=len(dd), spearman_rho=r, spearman_p=pv)
            for cut in ("2024-03-01", "2025-01-01"):
                a, b = dd[dd.date_dt < cut][mtr], dd[dd.date_dt >= cut][mtr]
                row[f"before_{cut}"] = f"{int((a > 0).sum())}/{len(a)} positive, mean {a.mean():+.2f}"
                row[f"after_{cut}"] = f"{int((b > 0).sum())}/{len(b)} positive, mean {b.mean():+.2f}"
            CT.append(row)
    pd.DataFrame(CT).to_csv(OUT / "chronology_tests.csv", index=False)
    print(pd.DataFrame(CT).round(3).to_string(index=False))

    # group summary figure
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, (typ, mtr, lab) in zip(axs, [("N2W", "art_level_corr", "wake-up: arteriole level change (%)"),
                                         ("N2W", "art_auc_corr", "wake-up: arteriole AUC"),
                                         ("MA", "art_auc_corr", "micro-arousal: arteriole AUC")]):
        for i, g in enumerate(GROUPS):
            v = D[(D.group == g) & (D.type == typ)][mtr].dropna()
            ax.scatter(np.full(len(v), i) + RNG.uniform(-0.08, 0.08, len(v)), v, color=C.GROUP_COL[g], s=30, zorder=3)
            if len(v):
                ax.hlines(v.mean(), i - 0.25, i + 0.25, color="k", lw=2)
                row = T1[(T1.group == g) & (T1.type == typ) & (T1.metric == mtr)]
                p = row["t_p"].iloc[0] if len(row) else np.nan
                ax.text(i, ax.get_ylim()[1] if False else v.max(), f"n={len(v)}\np={p:.3f}" if np.isfinite(p) else f"n={len(v)}",
                        ha="center", va="bottom", fontsize=7)
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_xticks(range(len(GROUPS)), [C.GROUP_LABEL[g] for g in GROUPS], fontsize=8)
        ax.set_title(lab + "\n(each dot = day, minus random NREM)", fontsize=9)
    fig.savefig(OUT / "figures" / "groups_summary.png", dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 230)
    print(D[["group", "date", "type", "n_events", "n_surr", "art_level", "art_level_surr", "art_level_corr",
             "art_auc_corr", "frac_down", "ca_auc"]].sort_values(["type", "group", "date"]).round(3).to_string(index=False))
    print("\n== response vs 0 (day level)\n", T1.round(4).to_string(index=False))
    print("\n== KD vs scramble\n", T2.round(4).to_string(index=False))
    print("->", OUT)


if __name__ == "__main__":
    main()
