"""Deep dive (2026-10-16): why GCaMP8m vessels dip at wake-up while RSC 2023 vessels dilate.

Builds on scripts/dilation_split.py. Same events (final labels, WT, NREM->wake and
REM->wake, t0 = pupil onset) and the same noisy-cycle exclusion
(2026.03.03 c13; 2026.03.06 c13-16).

  A  levels    vessel size around wake-up vs the cycle median (no pre-wake
               baseline), -120..+120 s: where does the vessel sit in sleep vs wake?
  B  cycles    per cycle: NREM-minus-Awake vessel level (excluding -30..+60 s
               around every wake-up) vs that cycle's mean wake-up response. If waking simply returns the vessel to its awake
               size, response ~ -(NREM - Awake).
  C  events    pre-wake level vs response, within day; LMEs
               response ~ cohort [+ pre_level | + cycle NREM-Awake] + (1 | day)
               to ask whether the cohort difference survives once the sleep-vs-wake
               vessel level is accounted for.
  D  behaviour whisking after waking (0-15 s), by cohort and vs response.
  E  signal    per-cycle measurement quality: raw area, awake robust CV,
               sample-to-sample noise, in-cycle drift, jumps, NREM slow
               vasomotion power (0.01-0.1 Hz).

Usage:
    PRELIM_LABEL_SET=relabel_2026-10-09 venv/bin/python -B scripts/dilation_deepdive.py
Output: data/RESULTS_dilation_deepdive_2026-10-16/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as ss
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "figs_prelim"))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import deck_part1_figures as P1  # noqa: E402
from rem_vs_nrem_wake import r2w_events  # noqa: E402

OUT = C.ROOT / "data" / "RESULTS_dilation_deepdive_2026-10-16"
FIG = OUT / "figures"
EXCLUDE = {("2026.03.03", 13), ("2026.03.06", 13), ("2026.03.06", 14), ("2026.03.06", 15), ("2026.03.06", 16)}
COH_COL = {"RSC2023": "#424242", "GCaMP8m": "#2E7D32"}
COH_LAB = {"RSC2023": "WT RSC 2023", "GCaMP8m": "WT GCaMP8m 2025–26"}
LONG = np.arange(-120.0, 120.0 + 1e-9, 0.5)
PRE, RESP = (-20.0, 0.0), (0.0, 15.0)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def cohort_of(key):
    return "GCaMP8m" if key.startswith("Extra_GCaMP8m") else "RSC2023"


def kept_keys():
    out = []
    for k in C.keys_for(["WT"]):
        i = C.parse_key(k)
        if (i.date, i.cycle) not in EXCLUDE:
            out.append(k)
    return out


def interp_on(t, y, grid, max_gap=1.5):
    fin = np.isfinite(y)
    if fin.sum() < 10:
        return np.full(len(grid), np.nan)
    tf, yf = t[fin], y[fin]
    v = np.interp(grid, tf, yf, left=np.nan, right=np.nan)
    j = np.clip(np.searchsorted(tf, grid), 1, len(tf) - 1)
    v[np.minimum(np.abs(tf[j] - grid), np.abs(tf[j - 1] - grid)) > max_gap] = np.nan
    return v


def state_at(L, t):
    i = (np.asarray(t) // C.BIN).astype(int)
    return np.array([L[j] if 0 <= j < len(L) else None for j in i], dtype=object)


# ----------------------------------------------------------------- events
def load_events(keys):
    n = C.response_events(keys)
    n = n[n["type"] == "N2W"][["type", "key", "mouse", "t0"]]
    r = r2w_events(keys)[["type", "key", "mouse", "t0"]]
    ev = pd.concat([n, r], ignore_index=True)
    ev["cohort"] = ev["key"].map(cohort_of)
    ev["day"] = ev["mouse"].str.split("_").str[-1]
    ev["cycle"] = ev["key"].str.extract(r"_cycle(\d+)$")[0].astype(int)
    ev = ev.reset_index(drop=True)
    art = P1.epoch_matrix(ev, "art")                              # pre-baseline ΔD/D0 (%)
    rg = P1.RG
    m, pm = (rg >= RESP[0]) & (rg <= RESP[1]), (rg >= PRE[0]) & (rg < PRE[1])
    with np.errstate(all="ignore"):
        ev["art_resp"] = np.nanmean(art[:, m], 1)
        ev["ca_resp"] = np.nanmean(P1.epoch_matrix(ev, "ca")[:, m], 1)
    lev = np.full((len(ev), len(LONG)), np.nan)
    wh = np.full(len(ev), np.nan)
    whpre = np.full(len(ev), np.nan)
    for i, r_ in enumerate(ev.itertuples()):
        cyc = C.load_cycle(r_.key)
        if "art" in cyc.sig:
            t, y = cyc.sig["art"]
            lev[i] = 100 * interp_on(t, y, r_.t0 + LONG)
        if "whisk" in cyc.sig:
            t, y = cyc.sig["whisk"]
            w = (t >= r_.t0 + RESP[0]) & (t < r_.t0 + RESP[1])
            wp = (t >= r_.t0 + PRE[0]) & (t < r_.t0 + PRE[1])
            wh[i] = np.nanmean(y[w]) if w.any() else np.nan
            whpre[i] = np.nanmean(y[wp]) if wp.any() else np.nan
    lp = (LONG >= PRE[0]) & (LONG < PRE[1])
    with np.errstate(all="ignore"):
        ev["pre_level"] = np.nanmedian(lev[:, lp], 1)               # vessel before waking vs cycle median (%)
        ev["post_level"] = np.nanmean(lev[:, (LONG >= 30) & (LONG <= 90)], 1)
    ev["whisk_resp"], ev["whisk_pre"] = wh, whpre
    ev["whisk_rise"] = ev.whisk_resp - ev.whisk_pre
    return ev, lev


# ----------------------------------------------------------------- per-cycle tables
def cycle_table(keys, ev):
    rows = []
    for k in keys:
        cyc = C.load_cycle(k)
        if "art" not in cyc.sig:
            continue
        info = C.parse_key(k)
        t, y = cyc.sig["art"]
        _, raw = C.read_trace(k, "arteriole-diameter")
        st = state_at(cyc.labels, t)
        # state levels away from the wake-ups themselves (drop -30..+60 s around every
        # N2W / R2W pupil onset), so the response is not part of its own predictor
        near = np.zeros(len(t), bool)
        for t0 in ev.loc[ev.key == k, "t0"]:
            near |= (t >= t0 - 30) & (t <= t0 + 60)
        st = np.where(near, None, st)
        lv = {}
        for s in ("Awake", "NREM", "REM"):
            msk = (st == s) & np.isfinite(y)
            lv[s] = 100 * np.nanmedian(y[msk]) if msk.sum() > 50 else np.nan
        fs = 1 / np.nanmedian(np.diff(t))
        yy = 100 * y
        dy = np.diff(yy)
        st_all = state_at(cyc.labels, t)
        aw = (st_all == "Awake") & np.isfinite(yy)
        awake_cv = 100 * stats.median_abs_deviation(y[aw], nan_policy="omit") * 1.4826 if aw.sum() > 50 else np.nan
        fin = np.isfinite(yy)
        slope = np.polyfit(t[fin], yy[fin], 1)[0] * (t[fin][-1] - t[fin][0]) if fin.sum() > 50 else np.nan
        # NREM slow vasomotion: Welch PSD on finite NREM runs >= 60 s
        nrem_pow = np.nan
        nm = (st_all == "NREM") & fin
        if nm.sum() * (1 / fs) >= 60:
            segs, cur = [], []
            for j, ok in enumerate(nm):
                if ok:
                    cur.append(j)
                elif cur:
                    segs.append(cur); cur = []
            if cur:
                segs.append(cur)
            P = []
            for sg in segs:
                if len(sg) / fs >= 60:
                    f, p = ss.welch(ss.detrend(yy[sg]), fs=fs, nperseg=min(len(sg), int(60 * fs)))
                    band = (f >= 0.01) & (f <= 0.1)
                    P.append(np.trapezoid(p[band], f[band]))
            if P:
                nrem_pow = float(np.mean(P))
        e = ev[ev.key == k]
        rows.append(dict(
            key=k, cohort=cohort_of(k), day=info.date, cycle=info.cycle,
            raw_area_median=np.nanmedian(raw), equiv_diam_px=np.sqrt(np.nanmedian(raw) / np.pi) * 2,
            awake_level=lv["Awake"], nrem_level=lv["NREM"], rem_level=lv["REM"],
            nrem_minus_awake=lv["NREM"] - lv["Awake"],
            awake_robust_cv_pct=awake_cv, step_noise_pct=np.nanstd(dy),
            jumps_gt10pct_per_min=np.sum(np.abs(dy) > 10) / (len(yy) / fs / 60),
            drift_pct_over_cycle=slope, nrem_slow_power=nrem_pow,
            n_n2w=int((e.type == "N2W").sum()), n2w_resp=e.loc[e.type == "N2W", "art_resp"].mean(),
            n_r2w=int((e.type == "R2W").sum()), r2w_resp=e.loc[e.type == "R2W", "art_resp"].mean(),
        ))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "cycle_table.csv", index=False)
    return T


def within_day_rho(d, x, y):
    v = d[["day", x, y]].dropna()
    r = v.groupby("day")[[x, y]].rank(pct=True)
    rho, p = stats.spearmanr(r[x], r[y])
    return rho, p, len(v)


# ----------------------------------------------------------------- figures
def fig_A_levels(ev, lev):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, constrained_layout=True)
    for ax, typ in zip(axs, ("N2W", "R2W")):
        for coh in ("RSC2023", "GCaMP8m"):
            d = ev[(ev.type == typ) & (ev.cohort == coh)]
            if not len(d):
                continue
            D = []
            for day, g in d.groupby("day"):
                X = lev[g.index.to_numpy()]
                with np.errstate(all="ignore"):
                    D.append(np.nanmean(X, 0))
            D = np.array(D)
            with np.errstate(all="ignore"):
                mu = np.nanmean(D, 0)
                se = np.nanstd(D, 0, ddof=1) / np.sqrt(len(D)) if len(D) > 1 else np.zeros_like(mu)
            ax.plot(LONG, pd.Series(mu).rolling(5, center=True, min_periods=1).mean(), color=COH_COL[coh], lw=2,
                    label=f"{COH_LAB[coh]} ({len(D)} days, {len(d)} events)")
            ax.fill_between(LONG, mu - se, mu + se, color=COH_COL[coh], alpha=0.2, lw=0)
        ax.axvline(0, color="k", lw=0.8); ax.axhline(0, color="0.6", lw=0.8)
        ax.set_title(("NREM" if typ == "N2W" else "REM") + " → wake", fontsize=13, weight="bold")
        ax.set_xlabel("time from pupil onset (s)")
        ax.legend(frameon=False, fontsize=9, loc="lower left")
    axs[0].set_ylabel("vessel size vs its cycle median (%)")
    for ax in axs:
        y0, y1 = ax.get_ylim()
        ax.text(-60, y1 - 0.05 * (y1 - y0), "asleep", ha="center", fontsize=10, color="0.4")
        ax.text(60, y1 - 0.05 * (y1 - y0), "awake", ha="center", fontsize=10, color="0.4")
    fig.savefig(FIG / "A_levels.png", dpi=160)
    plt.close(fig)


def fig_B_cycles(T):
    d = T[T.n_n2w > 0].dropna(subset=["nrem_minus_awake", "n2w_resp"])
    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    for coh in ("RSC2023", "GCaMP8m"):
        v = d[d.cohort == coh]
        ax.scatter(v.nrem_minus_awake, v.n2w_resp, s=25 + 12 * v.n_n2w, color=COH_COL[coh], alpha=0.8,
                   edgecolor="white", label=f"{COH_LAB[coh]} ({len(v)} cycles)")
    lim = np.nanmax(np.abs(d[["nrem_minus_awake", "n2w_resp"]].to_numpy())) * 1.1
    xx = np.array([-lim, lim])
    ax.plot(xx, -xx, color="0.5", ls="--", lw=1, label="response = −(NREM − Awake)")
    ax.axhline(0, color="0.8", lw=0.8); ax.axvline(0, color="0.8", lw=0.8)
    rho, p = stats.spearmanr(d.nrem_minus_awake, d.n2w_resp)
    w = np.sqrt(d.n_n2w)
    b = np.polyfit(d.nrem_minus_awake, d.n2w_resp, 1, w=w)
    ax.plot(xx, np.polyval(b, xx), color="#C62828", lw=1.5, label=f"fit, slope {b[0]:+.2f}")
    ax.set_xlabel("vessel in NREM minus Awake, this cycle (%)")
    ax.set_ylabel("NREM → wake response, cycle mean (%)")
    ax.set_title(f"Each cycle (dot size = wake-ups). ρ = {rho:+.2f}, "
                 f"{'p < 0.001' if p < 0.001 else f'p = {p:.3f}'} (cycles as units)", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.savefig(FIG / "B_cycles.png", dpi=160)
    plt.close(fig)
    return dict(rho=rho, p=p, n_cycles=len(d), slope=b[0], intercept=b[1])


def fig_C_events(ev):
    d = ev[(ev.type == "N2W")].dropna(subset=["pre_level", "art_resp"])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    res = {}
    for ax, coh in zip(axs, ("RSC2023", "GCaMP8m")):
        v = d[d.cohort == coh]
        days = sorted(v.day.unique())
        cm = plt.get_cmap("Greys" if coh == "RSC2023" else "Greens")
        for i, day in enumerate(days):
            w = v[v.day == day]
            ax.scatter(w.pre_level, w.art_resp, s=26, color=cm(0.4 + 0.55 * i / max(1, len(days) - 1)),
                       edgecolor="white", lw=0.4, label=day)
        rho, p, n = within_day_rho(v, "pre_level", "art_resp")
        res[coh] = dict(rho=rho, p=p, n=n)
        ax.axhline(0, color="0.8", lw=0.8); ax.axvline(0, color="0.8", lw=0.8)
        ax.set_title(f"{COH_LAB[coh]} NREM → wake\nwithin-day ρ = {rho:+.2f} ({'p < 0.001' if p < 0.001 else f'p = {p:.3f}'}, "
                     f"{n} events)", fontsize=11, color=COH_COL[coh], weight="bold")
        ax.set_xlabel("vessel before waking vs cycle median (%)")
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    axs[0].set_ylabel("wake-up response ΔD/D0, 0–15 s (%)")
    fig.savefig(FIG / "C_events_prelevel.png", dpi=160)
    plt.close(fig)
    return res


def fig_D_whisk(ev):
    d = ev[ev.type == "N2W"].dropna(subset=["whisk_rise", "art_resp"])
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    days = d.groupby(["cohort", "day"]).agg(whisk=("whisk_rise", "mean"), art=("art_resp", "mean")).reset_index()
    for x, coh in enumerate(("RSC2023", "GCaMP8m")):
        v = days[days.cohort == coh]
        axs[0].scatter(np.full(len(v), x) + np.linspace(-0.1, 0.1, len(v)), v.whisk, s=70, color=COH_COL[coh])
        axs[0].plot([x - 0.25, x + 0.25], [v.whisk.mean()] * 2, color="k", lw=2)
    r = C.group_test(days[days.cohort == "RSC2023"].whisk, days[days.cohort == "GCaMP8m"].whisk)
    axs[0].set_xticks([0, 1], [COH_LAB["RSC2023"], COH_LAB["GCaMP8m"]], fontsize=9)
    axs[0].set_ylabel("whisking rise after waking (z, 0–15 s minus −20…0 s)")
    axs[0].set_title(f"Day means (Welch p = {r['welch_p']:.2f})", fontsize=11)
    out = {"whisk_cohort_welch_p": r["welch_p"]}
    for ax, coh in zip(axs[1:], ("RSC2023", "GCaMP8m")):
        v = d[d.cohort == coh]
        ax.scatter(v.whisk_rise, v.art_resp, s=22, color=COH_COL[coh], alpha=0.75)
        rho, p, n = within_day_rho(v, "whisk_rise", "art_resp")
        out[f"{coh}_rho"], out[f"{coh}_p"] = rho, p
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_title(f"{COH_LAB[coh]}: within-day ρ = {rho:+.2f} (p = {p:.3f})", fontsize=10.5,
                     color=COH_COL[coh], weight="bold")
        ax.set_xlabel("whisking rise after waking (z)")
        ax.set_ylabel("wake-up response ΔD/D0 (%)")
    fig.savefig(FIG / "D_whisking.png", dpi=160)
    plt.close(fig)
    return out


QUAL = [("equiv_diam_px", "vessel size (equiv. diameter, px)"),
        ("awake_robust_cv_pct", "awake variability (robust CV, %)"),
        ("step_noise_pct", "sample-to-sample noise (%)"),
        ("jumps_gt10pct_per_min", "jumps > 10% per min"),
        ("drift_pct_over_cycle", "drift over the cycle (%)"),
        ("nrem_slow_power", "NREM slow vasomotion power (0.01–0.1 Hz)"),
        ("nrem_minus_awake", "NREM − Awake level (%)")]


QUAL_PLOT = ("equiv_diam_px", "step_noise_pct", "jumps_gt10pct_per_min", "drift_pct_over_cycle", "nrem_minus_awake")


def fig_E_quality(T):
    fig, axs = plt.subplots(1, len(QUAL_PLOT), figsize=(2.6 * len(QUAL_PLOT), 4.2), constrained_layout=True)
    axs = iter(axs)
    rows = []
    dm = T.groupby(["cohort", "day"])[[q for q, _ in QUAL]].mean().reset_index()
    for q, lab in QUAL:
        a, b = dm[dm.cohort == "RSC2023"][q].dropna(), dm[dm.cohort == "GCaMP8m"][q].dropna()
        r = C.group_test(a, b)
        rows.append(dict(measure=q, label=lab, rsc_day_mean=a.mean(), gcamp_day_mean=b.mean(),
                         welch_p=r["welch_p"], mw_p=r["mw_p"], rsc_days=len(a), gcamp_days=len(b)))
        if q not in QUAL_PLOT:
            continue
        ax = next(axs)
        for x, coh in enumerate(("RSC2023", "GCaMP8m")):
            v = T[T.cohort == coh][q].dropna()
            ax.scatter(np.full(len(v), x) + np.random.default_rng(1).uniform(-0.15, 0.15, len(v)), v, s=14,
                       color=COH_COL[coh], alpha=0.5)
            dv = dm[dm.cohort == coh][q].dropna()
            ax.scatter(np.full(len(dv), x + 0.3), dv, s=55, color=COH_COL[coh], edgecolor="white")
        ax.set_xticks([0, 1], ["RSC\n2023", "GCaMP8m"], fontsize=8.5)
        ax.set_title(f"{lab}\nWelch p = {r['welch_p']:.3f}, MW p = {r['mw_p']:.3f}", fontsize=8.5,
                     color="#C62828" if np.isfinite(r["welch_p"]) and r["welch_p"] < 0.05 else "k")
    fig.suptitle("Per-cycle measurement properties (small dots = cycles, large = day means; p on day means)",
                 fontsize=10, x=0.01, ha="left")
    fig.savefig(FIG / "E_quality.png", dpi=160)
    plt.close(fig)
    Q = pd.DataFrame(rows)
    Q.to_csv(OUT / "quality_tests.csv", index=False)
    return Q


def lme_tests(ev, T):
    """response ~ cohort (+ pre_level | + cycle NREM-Awake) + (1 | day), N2W events."""
    d = ev[ev.type == "N2W"].merge(T[["key", "nrem_minus_awake"]], on="key", how="left")
    d = d.dropna(subset=["art_resp", "pre_level", "nrem_minus_awake"]).copy()
    d["gcamp"] = (d.cohort == "GCaMP8m").astype(int)
    out = []
    for fx in ("gcamp", "gcamp + pre_level", "gcamp * pre_level", "nrem_minus_awake",
               "gcamp + nrem_minus_awake", "gcamp * nrem_minus_awake"):
        L = C.lmm(d, "art_resp", fx, group="day")
        if L is None:
            continue
        L["model"] = f"art_resp ~ {fx} + (1|day)"
        out.append(L)
    L = pd.concat(out, ignore_index=True)
    L.to_csv(OUT / "lme_cohort_prelevel.csv", index=False)
    return L


def main():
    C.set_label_mode("manual")
    FIG.mkdir(parents=True, exist_ok=True)
    keys = kept_keys()
    ev, lev = load_events(keys)
    ev.to_csv(OUT / "events.csv", index=False)
    T = cycle_table(keys, ev)
    fig_A_levels(ev, lev)
    B = fig_B_cycles(T)
    Cc = fig_C_events(ev)
    Dw = fig_D_whisk(ev)
    Q = fig_E_quality(T)
    L = lme_tests(ev, T)
    pd.set_option("display.width", 220)
    print("events:", ev.groupby(["cohort", "type"]).size().to_dict())
    print("pre/post level by cohort (N2W, day means):")
    print(ev[ev.type == "N2W"].groupby(["cohort", "day"])[["pre_level", "post_level", "art_resp"]].mean().round(2))
    print("B cycles:", {k: round(v, 3) for k, v in B.items()})
    for coh in ("RSC2023", "GCaMP8m"):
        v = T[(T.cohort == coh) & (T.n_n2w > 0)].dropna(subset=["nrem_minus_awake", "n2w_resp"])
        rho, p = stats.spearmanr(v.nrem_minus_awake, v.n2w_resp)
        print(f"  {coh}: cycles {len(v)}, rho {rho:+.2f}, p {p:.3f}")
    print("C events pre-level:", Cc)
    print("D whisking:", {k: round(v, 3) for k, v in Dw.items()})
    print(Q.round(3).to_string(index=False))
    print(L[["model", "term", "Coef.", "Std.Err.", "P>|z|", "n_obs", "n_mice", "mouse_var_over_resid",
             "degenerate_fit"]].to_string(index=False))
    print(T[["cohort", "day", "cycle", "equiv_diam_px", "awake_level", "nrem_level", "rem_level", "nrem_minus_awake",
             "step_noise_pct", "drift_pct_over_cycle", "n_n2w", "n2w_resp"]].round(2).to_string(index=False))
    print("->", OUT)


if __name__ == "__main__":
    main()
