#!/usr/bin/env python3
"""Micro-arousals from the pupil-based event detector, classified with manual labels.

Idea (Huy, 2026-10-02): the May pipeline (and the August re-run) detected many
short pupil events that were later thrown away by the 10-s duration filter. An
event that is short and sits inside NREM is very likely a micro-arousal (MA).
This script recovers those events, classifies them by sleep context using the
manual labels, and characterises them.

SOURCE
  data/events/<key>_detected_events.csv (May 2026, raw detector output, every
  group): Start = pupil-slope onset (back-tracked), End = end of the following
  whisker burst (or Start + 5 s if no whisking).
  Same-Start duplicates are collapsed to the shortest End, as in the August
  canonical set (scripts/build_canonical_unfiltered_events.py).
  The detector's Type (from *_detected_categories.csv) is attached for reporting
  only and NEVER used to select events: Type 3 means "arteriole dilated", so
  filtering on it would build the arteriole result in.

CONTEXT CLASSIFICATION (labels: Florence's manual labels via scripts/figs_prelim/
common.py; IP3R2 KD bins without manual labels are filled by the ML model, and
every KD result is repeated manual-only)
  pre   = label of the 10 s before Start      (NREM if >= 80% of labelled bins)
  post  = label of the 10 s after End
  post20= label of the 20 s after End (for wake-ups)
  MA        3 <= duration <= 15 s, pre NREM, post NREM      <- Huy's rule
  wake-up   pre NREM, post20 Awake (>= 80%)
  in-wake   pre Awake
  other     everything else (mixed, Unclear, REM, unlabelled)

FOR EACH PUPIL-DETECTED MA
  * what Florence's labels say inside it (did she mark it Awake = a label MA?)
  * event-triggered pupil / whisker / arteriole / Ca2+ (or NE), aligned to Start,
    baseline -5..0 s, response 0..15 s (common.epoch / response_metrics)
  * a SURROGATE control: random NREM time points in the same cycle, matched on
    pre-onset pupil (the detector rejects onsets with raised pupil, so an
    unmatched null is biased). An MA response is only real if it beats this.
  * KD vs scramble: mouse-level Welch + Mann-Whitney + Hedges g, mixed model
    value ~ group + (1|day) with a degenerate-fit flag, BH across tests.

Outputs -> data/RESULTS_pupil_event_MA_2026-10-02/ (new folder, refuses to overwrite)
    ./venv/bin/python -B scripts/pupil_events_ma.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figs_prelim import common as C  # noqa: E402

EV_DIR = ROOT / "data" / "events"
OUT = ROOT / "data" / "RESULTS_pupil_event_MA_2026-10-02"
MA_MIN, MA_MAX = 3.0, 15.0
CTX_S, POST_WAKE_S, CTX_FRAC = 10.0, 20.0, 0.80
SURR_PER_EVENT, SURR_EXCL_S, SURR_PUPIL_TOL = 20, 20.0, 0.25
GROUPS = ("WT", "SCR", "KD", "NE")
RNG = np.random.default_rng(20261002)


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #
def load_detector_events() -> pd.DataFrame:
    rows = []
    for f in sorted(EV_DIR.glob("*_detected_events.csv")):
        key = f.name[: -len("_detected_events.csv")]
        try:
            info = C.parse_key(key)
        except ValueError:
            continue
        if info.group not in GROUPS:
            continue
        d = pd.read_csv(f)
        if d.empty:
            continue
        d = d.sort_values(["Start", "End"]).groupby("Start", as_index=False).first()  # same-Start -> shortest End
        cat = EV_DIR / f"{key}_detected_categories.csv"
        if cat.exists():
            c = pd.read_csv(cat)
            d = d.merge(c[["Start", "End", "Type"]], on=["Start", "End"], how="left")
        else:
            d["Type"] = np.nan
        d["key"], d["mouse"], d["group"] = key, info.mouse, info.group
        rows.append(d)
    E = pd.concat(rows, ignore_index=True)
    E["duration"] = E["End"] - E["Start"]
    return E


def frac_state(labels, t0, t1, state):
    """Fraction of the LABELLED bins overlapping [t0, t1) that carry `state`."""
    b0, b1 = int(np.floor(max(t0, 0) / C.BIN)), int(np.ceil(t1 / C.BIN))
    seg = [v for v in labels[b0:b1] if v is not None]
    if not seg or len(seg) < 0.5 * max(b1 - b0, 1):
        return np.nan
    return float(np.mean([v == state for v in seg]))


def classify(E: pd.DataFrame) -> pd.DataFrame:
    out = E.copy()
    pre, post, post20, inside_awake, src = [], [], [], [], []
    for r in out.itertuples():
        lab = C.load_labels(r.key)
        pre.append(frac_state(lab, r.Start - CTX_S, r.Start, "NREM"))
        post.append(frac_state(lab, r.End, r.End + CTX_S, "NREM"))
        post20.append(frac_state(lab, r.End, r.End + POST_WAKE_S, "Awake"))
        inside_awake.append(frac_state(lab, r.Start, max(r.End, r.Start + C.BIN), "Awake"))
        src.append(C.label_source(r.key))
    out["pre_nrem"], out["post_nrem"], out["post20_awake"] = pre, post, post20
    out["inside_labelled_awake"], out["label_source"] = inside_awake, src
    pre_awake = 1 - out["pre_nrem"]

    def cls(r):
        if not np.isfinite(r.pre_nrem):
            return "unlabelled"
        if r.pre_nrem >= CTX_FRAC and MA_MIN <= r.duration <= MA_MAX \
                and np.isfinite(r.post_nrem) and r.post_nrem >= CTX_FRAC:
            return "MA"
        if r.pre_nrem >= CTX_FRAC and np.isfinite(r.post20_awake) and r.post20_awake >= CTX_FRAC:
            return "wake-up"
        if 1 - r.pre_nrem >= CTX_FRAC:
            return "in-wake"
        return "other"
    out["context"] = out.apply(cls, axis=1)
    return out


# --------------------------------------------------------------------------- #
# responses + surrogate
# --------------------------------------------------------------------------- #
def signals_for(group):
    return ("pupil", "whisk", "art", "ne") if group == "NE" else ("pupil", "whisk", "art", "ca")


def responses(ev: pd.DataFrame) -> pd.DataFrame:
    e = ev.assign(t0=ev["Start"]).reset_index(drop=True)
    out = e[["key", "mouse", "group", "t0", "duration", "Type", "label_source",
             "inside_labelled_awake"]].copy()
    for s in ("pupil", "whisk", "art", "ca", "ne"):
        rg, M = C.epochs_for(e, s)
        pk, auc, tpk = C.response_metrics(rg, M)
        out[f"{s}_peak"], out[f"{s}_auc"], out[f"{s}_tpk"] = pk, auc, tpk
        out[f"{s}_pre"] = [np.nan] * len(e)
    # pre-onset pupil level (raw z, -5..0 s), for surrogate matching
    pre = []
    for r in e.itertuples():
        cyc = C.load_cycle(r.key)
        t, y = cyc.sig.get("pupil", (None, None))
        pre.append(np.nanmean(y[(t >= r.t0 - 5) & (t < r.t0)]) if t is not None else np.nan)
    out["pupil_pre"] = pre
    return out


def surrogate_times(key, real_t0s, all_t0s, n):
    """Random NREM time points: 10 s of NREM before and after, >= 20 s from any detected event."""
    lab = C.load_labels(key)
    cand = []
    for b in range(2, len(lab) - 2):
        t = b * C.BIN
        if frac_state(lab, t - CTX_S, t, "NREM") == 1.0 and frac_state(lab, t, t + CTX_S, "NREM") == 1.0:
            if np.all(np.abs(np.asarray(all_t0s) - t) > SURR_EXCL_S):
                cand.append(t + RNG.uniform(0, C.BIN))
    if not cand:
        return []
    return list(RNG.choice(cand, size=min(n, len(cand) * 3), replace=True))


def surrogate_control(ma: pd.DataFrame, E_all: pd.DataFrame) -> pd.DataFrame:
    """For each real MA, the mean response of pupil-matched surrogate NREM points."""
    rows = []
    for key, s in ma.groupby("key"):
        all_t0 = E_all.loc[E_all.key == key, "Start"].to_numpy()
        pool = surrogate_times(key, s.t0.to_numpy(), all_t0, SURR_PER_EVENT * 5 * len(s))
        if not pool:
            continue
        P = pd.DataFrame(dict(key=key, t0=pool, type="SURR"))
        P["mouse"], P["group"] = s.mouse.iloc[0], s.group.iloc[0]
        P["Start"], P["duration"], P["Type"] = P["t0"], np.nan, np.nan
        P["label_source"], P["inside_labelled_awake"] = s.label_source.iloc[0], np.nan
        R = responses(P)
        for r in s.itertuples():
            ok = np.abs(R.pupil_pre - r.pupil_pre) <= SURR_PUPIL_TOL
            m = R[ok].head(SURR_PER_EVENT) if ok.sum() >= 3 else R.iloc[
                np.argsort(np.abs(R.pupil_pre - r.pupil_pre).to_numpy())[:SURR_PER_EVENT]]
            row = dict(key=key, mouse=r.mouse, group=r.group, t0=r.t0, n_surr=len(m),
                       pupil_matched=bool(ok.sum() >= 3))
            for s_ in ("pupil", "whisk", "art", "ca", "ne"):
                for met in ("peak", "auc"):
                    row[f"{s_}_{met}_real"] = getattr(r, f"{s_}_{met}")
                    row[f"{s_}_{met}_surr"] = m[f"{s_}_{met}"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def lmm_group(d, value):
    d = d.dropna(subset=[value]).copy()
    if d.mouse.nunique() < 3 or d.group.nunique() < 2:
        return np.nan, np.nan, np.nan
    d["kd"] = (d.group == "KD").astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = smf.mixedlm(f"{value} ~ kd", d, groups=d.mouse).fit(reml=True)
            ratio = float(r.cov_re.iloc[0, 0]) / float(r.scale)
            return r.params["kd"], r.pvalues["kd"], ratio
        except Exception:  # noqa: BLE001
            return np.nan, np.nan, np.nan


def kd_vs_scr(R: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    d = R[R.group.isin(["SCR", "KD"])]
    for sig in ("pupil", "whisk", "art", "ca"):
        for met in ("peak", "auc"):
            v = f"{sig}_{met}"
            m = d.groupby(["group", "mouse"])[v].mean().reset_index()
            gt = C.group_test(m[m.group == "SCR"][v], m[m.group == "KD"][v])
            coef, p, ratio = lmm_group(d, v)
            rows.append(dict(labels=label, measure=v, scr_mean=gt["mean_a"], kd_mean=gt["mean_b"],
                             days_scr=gt["n_a"], days_kd=gt["n_b"],
                             events_scr=int((d.group == "SCR").sum()), events_kd=int((d.group == "KD").sum()),
                             hedges_g=gt["hedges_g"], welch_p=gt["welch_p"], mw_p=gt["mw_p"],
                             lmm_kd_minus_scr=coef, lmm_p=p, mouse_var_over_resid=ratio,
                             lmm_degenerate=bool(np.isfinite(ratio) and ratio < C.LMM_DEGENERATE_RATIO)))
    T = pd.DataFrame(rows)
    T["welch_q_bh"] = C.bh(T.welch_p.to_numpy())
    T["lmm_q_bh"] = C.bh(np.where(T.lmm_degenerate, np.nan, T.lmm_p))
    return T


def surrogate_tests(S: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in GROUPS:
        s = S[S.group == g]
        if s.empty:
            continue
        for sig in signals_for(g):
            for met in ("peak", "auc"):
                a, b = f"{sig}_{met}_surr", f"{sig}_{met}_real"
                m = s.groupby("mouse")[[a, b]].mean().dropna()
                r = C.paired_test(m[a], m[b])
                rows.append(dict(group=g, measure=f"{sig}_{met}", n_days=r["n"], n_events=len(s),
                                 surrogate=r["mean_a"], real=r["mean_b"], real_minus_surr=r["mean_diff"],
                                 dz=r["dz"], t_p=r["t_p"], wilcoxon_p=r["wilcoxon_p"],
                                 wilcoxon_floor=r["wilcoxon_min_p"],
                                 days_real_gt_surr=int((m[b] > m[a]).sum())))
    T = pd.DataFrame(rows)
    T["t_q_bh"] = C.bh(T.t_p.to_numpy())
    return T


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def eta_by_group(ma, fname, title):
    sig_rows = ("pupil", "whisk", "art", "ca_or_ne")
    fig, axes = plt.subplots(1, len(sig_rows), figsize=(16, 4.8))   # side by side, near-square panels
    for ax, s in zip(axes, sig_rows):
        for g in GROUPS:
            e = ma[ma.group == g].assign(t0=lambda x: x.Start)
            sig = ("ne" if g == "NE" else "ca") if s == "ca_or_ne" else s
            if e.empty:
                continue
            rg, M = C.epochs_for(e.reset_index(drop=True), sig)
            pm = pd.DataFrame(M).groupby(e.mouse.values).mean()
            if pm.dropna(how="all").empty:
                continue
            mu = pm.mean().to_numpy(); se = pm.std(ddof=1).to_numpy() / np.sqrt(len(pm))
            ax.plot(rg, mu, color=C.GROUP_COL[g], lw=1.5,
                    label=f"{C.GROUP_LABEL[g]} ({e.mouse.nunique()} d, {len(e)} ev)" + (" [NE]" if sig == "ne" else ""))
            ax.fill_between(rg, mu - se, mu + se, color=C.GROUP_COL[g], alpha=.15, lw=0)
        ax.axvspan(*C.BASE_WIN, color="0.88", lw=0, zorder=0)
        ax.axvspan(*C.RESP_WIN, color="#FFF3E0", lw=0, zorder=0)
        ax.axvline(0, color="k", lw=.6)
        ax.set_title({"pupil": "pupil (z)", "whisk": "whisker (z)", "art": "arteriole ΔD/D₀",
                      "ca_or_ne": "Ca²⁺ / NE ΔF/F₀"}[s], fontsize=10)
        ax.set_xlabel("time from pupil onset (s)", fontsize=8)
        ax.set_xlim(-20, 40)
    axes[0].legend(fontsize=6.5, loc="upper right", frameon=False)
    axes[0].set_ylabel("mean ± SEM across days", fontsize=8)
    fig.suptitle(title + "\n" + C.PRELIM_STAMP, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def surrogate_fig(S, T):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, g in zip(axes, GROUPS):
        s = S[S.group == g]
        sigs = signals_for(g)
        for j, sig in enumerate(sigs):
            a, b = f"{sig}_peak_surr", f"{sig}_peak_real"
            m = s.groupby("mouse")[[a, b]].mean().dropna()
            sc = m[a].abs().max() if len(m) and m[a].abs().max() > 0 else 1
            for _, r in m.iterrows():
                ax.plot([j - .15, j + .15], [r[a] / sc, r[b] / sc], color="0.6", lw=.8)
            ax.scatter([j - .15] * len(m), m[a] / sc, color="0.5", s=18, zorder=3)
            ax.scatter([j + .15] * len(m), m[b] / sc, color=C.GROUP_COL[g], s=22, zorder=3)
            row = T[(T.group == g) & (T.measure == f"{sig}_peak")]
            if len(row):
                ax.text(j, ax.get_ylim()[1] if False else 1.15, f"p={row.t_p.iloc[0]:.3f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(sigs))); ax.set_xticklabels(sigs, fontsize=8)
        ax.set_ylim(-0.6, 1.3)
        ax.set_title(f"{C.GROUP_LABEL[g]}: MA peak vs pupil-matched\nNREM surrogate (grey); per-day, scaled", fontsize=8)
    fig.suptitle("Is the MA response real? Real pupil-detected MAs vs random NREM moments with the same "
                 "pre-onset pupil\n" + C.PRELIM_STAMP, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "MA_vs_surrogate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def overview_fig(E):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    tab = pd.crosstab(E.group, E.context).reindex(index=list(GROUPS))
    cols = ["MA", "wake-up", "in-wake", "other", "unlabelled"]
    tab = tab.reindex(columns=[c for c in cols if c in tab.columns], fill_value=0)
    bottom = np.zeros(len(tab))
    colmap = {"MA": "#8E24AA", "wake-up": "#1565C0", "in-wake": "#90CAF9", "other": "#BDBDBD", "unlabelled": "#EEEEEE"}
    for c in tab.columns:
        ax.bar(range(len(tab)), tab[c], bottom=bottom, color=colmap[c], label=c)
        bottom += tab[c].to_numpy()
    ax.set_xticks(range(len(tab))); ax.set_xticklabels([C.GROUP_LABEL[g] for g in tab.index], fontsize=8)
    ax.set_ylabel("detector events"); ax.legend(fontsize=7)
    ax.set_title("All pupil-detector events by sleep context", fontsize=9)
    ax = axes[1]
    for g in GROUPS:
        d = E[(E.group == g)].duration
        ax.hist(d.clip(upper=40), bins=np.arange(0, 41, 1), histtype="step", color=C.GROUP_COL[g],
                label=C.GROUP_LABEL[g], lw=1.2)
    ax.axvspan(MA_MIN, MA_MAX, color="#8E24AA", alpha=.08)
    ax.axvline(10, color="k", ls=":", lw=1)
    ax.text(10.3, ax.get_ylim()[1] * .9, "old 10-s filter", fontsize=7)
    ax.set_xlabel("event duration (s)"); ax.set_ylabel("events"); ax.legend(fontsize=7)
    ax.set_title("Durations (shaded = MA range 3-15 s)", fontsize=9)
    ax = axes[2]
    ma = E[E.context == "MA"]
    lab = pd.cut(ma.inside_labelled_awake, [-.01, 0.0, .5, 1.0], labels=["labelled NREM throughout",
                                                                        "partly Awake", "mostly Awake"])
    t = pd.crosstab(ma.group, lab).reindex(index=list(GROUPS)).fillna(0)
    t.plot(kind="bar", stacked=True, ax=ax, color=["#C62828", "#FFB74D", "#1565C0"], legend=True, rot=0)
    ax.set_xticklabels([C.GROUP_LABEL[g] for g in t.index], fontsize=8)
    ax.legend(fontsize=7); ax.set_ylabel("pupil-detected MAs")
    ax.set_title("What the manual labels say inside each MA", fontsize=9)
    fig.suptitle("Pupil-detector events (May 2026 raw set) classified with manual sleep labels\n"
                 + C.PRELIM_STAMP, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "overview_events_by_context.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; not overwriting")
    (OUT / "figures").mkdir(parents=True)
    (OUT / "tables").mkdir()
    C.set_label_mode("manual+ml")

    E = classify(load_detector_events())
    E.to_csv(OUT / "tables" / "all_detector_events_classified.csv", index=False)

    # rates per hour of NREM
    nrem_h = {}
    for k in E.key.unique():
        lab = C.load_labels(k)
        nrem_h[k] = np.sum([v == "NREM" for v in lab]) * C.BIN / 3600
    E["nrem_h"] = E.key.map(nrem_h)
    rate = (E[E.context == "MA"].groupby(["group", "mouse"]).size()
            / E.groupby(["group", "mouse"]).apply(lambda s: s.drop_duplicates("key").nrem_h.sum())).rename(
        "MA_per_h_NREM").reset_index()
    rate.to_csv(OUT / "tables" / "MA_rate_per_day.csv", index=False)
    ctx = pd.crosstab(E.group, E.context, margins=True)
    ctx.to_csv(OUT / "tables" / "context_counts.csv")

    MA = E[E.context == "MA"].copy()
    R = responses(MA)
    R.to_csv(OUT / "tables" / "MA_event_responses.csv", index=False)

    # label agreement
    agree = MA.assign(labelled=pd.cut(MA.inside_labelled_awake, [-.01, 0.0, .5, 1.0],
                                      labels=["NREM throughout", "partly Awake", "mostly Awake"]))
    agree_t = pd.crosstab(agree.group, agree.labelled, margins=True)
    agree_t.to_csv(OUT / "tables" / "MA_vs_manual_labels.csv")
    lab_ma = C.event_table(C.keys_for(list(GROUPS)))
    lab_ma = lab_ma[lab_ma.type == "MA"]
    hit = []
    for r in lab_ma.itertuples():
        e = E[(E.key == r.key) & (E.Start <= r.t0 + C.BIN + 3) & (E.End >= r.t0 - 3)]
        hit.append(len(e) > 0)
    lab_ma = lab_ma.assign(has_pupil_event=hit)
    found = lab_ma.groupby("group").has_pupil_event.agg(["size", "mean"]).rename(
        columns={"size": "label_MAs", "mean": "frac_with_pupil_event"})
    found.to_csv(OUT / "tables" / "label_MAs_found_by_detector.csv")

    # surrogate
    S = surrogate_control(R.assign(Start=R.t0), E)
    S.to_csv(OUT / "tables" / "MA_surrogate_pairs.csv", index=False)
    ST = surrogate_tests(S)
    ST.to_csv(OUT / "tables" / "MA_vs_surrogate_tests.csv", index=False)

    # KD vs SCR, both label modes
    KS = [kd_vs_scr(R, "manual+ml")]
    C.set_label_mode("manual")
    E_m = classify(load_detector_events())
    R_m = responses(E_m[E_m.context == "MA"])
    KS.append(kd_vs_scr(R_m, "manual only"))
    rate_m = (E_m[E_m.context == "MA"].groupby(["group", "mouse"]).size()).rename("n_MA").reset_index()
    C.set_label_mode("manual+ml")
    KS = pd.concat(KS, ignore_index=True)
    KS.to_csv(OUT / "tables" / "MA_KD_vs_SCR.csv", index=False)
    rg = rate.merge(rate, how="left")
    rt = C.group_test(rate[rate.group == "SCR"].MA_per_h_NREM, rate[rate.group == "KD"].MA_per_h_NREM)

    # figures
    overview_fig(E)
    eta_by_group(MA, "MA_eta_by_group.png", "Pupil-detected micro-arousals (3-15 s, NREM before and after), "
                                             "aligned to pupil onset; mean ± SEM across days")
    surrogate_fig(S, ST)

    # ---------- report ----------
    pd.set_option("display.width", 220)
    print(f"detector events: {len(E)} in {E.key.nunique()} cycles\n")
    print("=== context counts ===\n" + ctx.to_string())
    print("\n=== MA rate per hour of NREM (per day) ===")
    print(rate.groupby("group").MA_per_h_NREM.describe()[["count", "mean", "min", "max"]].round(1).to_string())
    print(f"   KD vs SCR MA rate: SCR {rt['mean_a']:.1f} vs KD {rt['mean_b']:.1f}/h, Welch p={rt['welch_p']:.3f}")
    print("\n=== what manual labels say inside pupil-detected MAs ===\n" + agree_t.to_string())
    print("\n=== label-defined MAs that have a detector event ===\n" + found.round(2).to_string())
    print("\n=== MA response vs pupil-matched NREM surrogate (per-day paired) ===")
    print(ST[["group", "measure", "n_days", "n_events", "surrogate", "real", "days_real_gt_surr",
              "t_p", "t_q_bh", "wilcoxon_p"]].round(4).to_string(index=False))
    print("\n=== KD vs SCR (pupil-detected MAs) ===")
    print(KS[["labels", "measure", "events_scr", "events_kd", "days_scr", "days_kd", "scr_mean", "kd_mean",
              "hedges_g", "welch_p", "welch_q_bh", "lmm_p", "lmm_degenerate"]].round(4).to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
