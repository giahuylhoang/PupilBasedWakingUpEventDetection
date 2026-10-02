"""Figure 8 — IP3R2 KD vs scramble arousal-event astrocyte-vascular responses.

Panels
------
8A  Representative N2W event, SCR vs KD (pupil, whisker, arteriole, Ca).
8B  N2W event-triggered averages, SCR vs KD, mouse mean +/- SEM
    (pupil, whisker, arteriole, Ca); arousal (pupil/whisker) shown first.
8C  N2W heat maps (arteriole, Ca), SCR and KD separately, rows sorted by
    each event's peak pupil response.
8D  N2W per-mouse quantification (peak Ca, peak art, AUC, response
    probability, time-to-peak) + group_test (mouse-level) + LMM on
    event-level peaks, with and without pupil-peak covariate.
8E  Same event-triggered averages as 8B but for MA.
8F  Same quantification as 8D but for MA.
8G  Forest plot of Hedges g (KD vs SCR, mouse-level) with 2000-resample
    bootstrap 95% CI, for N2W art, N2W Ca, MA art, MA Ca, in both label
    modes (manual+ml primary, manual sensitivity).

Primary label mode is 'manual+ml' for KD (ML fills unlabelled bins);
'manual' is run as a sensitivity check and shown alongside, with
ML-derived events flagged. SCR is manual-only in both runs.

Run: ./venv/bin/python -B -m scripts.figs_prelim.fig8   (from repo root)
"""
from __future__ import annotations

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, ".")
from scripts.figs_prelim import common as C  # noqa: E402

RNG = np.random.default_rng(20261001)
N_BOOT = 2000
SIGS = ("pupil", "whisk", "art", "ca")
GROUPS = ("SCR", "KD")


# --------------------------------------------------------------------------- #
# data assembly
# --------------------------------------------------------------------------- #
def build_events(ev_type: str, label_mode: str) -> pd.DataFrame:
    """Event table for SCR + KD under a given KD label mode (SCR always
    manual, since that is all that exists for SCR)."""
    C.set_label_mode("manual")
    scr_keys = C.keys_for("SCR")
    ev_scr = C.response_events(scr_keys)   # pupil-anchored, non-overlapping (v2)
    ev_scr = ev_scr[ev_scr["type"] == ev_type].copy()
    ev_scr["label_mode"] = "manual"
    ev_scr["ml_derived"] = False

    C.set_label_mode(label_mode)
    kd_keys = C.keys_for("KD")
    ev_kd = C.response_events(kd_keys)
    ev_kd = ev_kd[ev_kd["type"] == ev_type].copy()
    ev_kd["label_mode"] = label_mode
    # mark events on days whose bins are partly/fully ML-derived
    src = {k: C.label_source(k) for k in kd_keys}
    ev_kd["ml_derived"] = ev_kd["key"].map(src).isin(("manual+ml", "ml"))

    out = pd.concat([ev_scr, ev_kd], ignore_index=True)
    out["event_id"] = np.arange(len(out))
    C.set_label_mode("manual+ml")  # restore default
    return out


def epochs_dict(ev: pd.DataFrame) -> dict:
    """signal -> (rel_grid, matrix) epochs for the full event table `ev`."""
    out = {}
    for s in SIGS:
        rg, M = C.epochs_for(ev, s)
        out[s] = (rg, M)
    return out


def metrics_table(ev: pd.DataFrame, eps: dict) -> pd.DataFrame:
    """Per-event peak/AUC/tpk/responder for art, ca, pupil; attach to ev."""
    d = ev.reset_index(drop=True).copy()
    for s in ("art", "ca", "pupil"):
        rg, M = eps[s]
        peak, auc, tpk = C.response_metrics(rg, M)
        resp = C.responder(rg, M)
        d[f"{s}_peak"] = peak
        d[f"{s}_auc"] = auc
        d[f"{s}_tpk"] = tpk
        d[f"{s}_responder"] = resp
    return d


# --------------------------------------------------------------------------- #
# 8A representative events
# --------------------------------------------------------------------------- #
def _window_has_signal(cyc, name: str, t0: float, min_std: float = 1e-6) -> bool:
    """True if signal `name` has non-trivial variance (not a flat dropout)
    across the full -30..+60 s event window."""
    if name not in cyc.sig:
        return False
    t, y = cyc.sig[name]
    m = (t >= t0 - C.EV_PRE) & (t <= t0 + C.EV_POST)
    seg = y[m]
    seg = seg[np.isfinite(seg)]
    return seg.size > 10 and np.nanstd(seg) > min_std


def pick_representative(ev: pd.DataFrame, group: str, prefer_manual: bool = False):
    """Pick the event with a clear pupil rise, requiring every trace shown in
    8A to have real (non-flat) variance across the full event window so a
    data dropout is never silently displayed as a flat line."""
    sub = ev[ev["group"] == group]

    def _search(pool):
        best, best_score = None, -np.inf
        for r in pool.itertuples():
            cyc = C.load_cycle(r.key)
            if not all(_window_has_signal(cyc, s, r.t0) for s in SIGS):
                continue
            t, y = cyc.sig["pupil"]
            v = C.epoch(t, y, r.t0, C.EPOCH_FS["pupil"])
            rg = C.rel_grid(C.EPOCH_FS["pupil"])
            m = (rg >= C.RESP_WIN[0]) & (rg <= C.RESP_WIN[1])
            if not np.isfinite(v[m]).any():
                continue
            score = np.nanmax(v[m])
            if score > best_score:
                best_score, best = score, r
        return best

    if sub.empty:
        return None
    if prefer_manual:
        manual_only = sub[~sub["ml_derived"]]
        best = _search(manual_only) if len(manual_only) else None
        if best is not None:
            return best
        # fall back to the full pool (manual-only days had no clean window)
    return _search(sub)


def panel_8A(ax_block, scr_row, kd_row):
    for col, (row, label) in enumerate([(scr_row, "SCR"), (kd_row, "IP3R2 KD")]):
        if row is None:
            for ax in ax_block[:, col]:
                ax.axis("off")
            ax_block[0, col].set_title(f"{label}: no suitable event", fontsize=9)
            continue
        cyc = C.load_cycle(row.key)
        for i, s in enumerate(SIGS):
            ax = ax_block[i, col]
            t, y = cyc.sig[s]
            m = (t >= row.t0 - C.EV_PRE) & (t <= row.t0 + C.EV_POST)
            ax.plot(t[m] - row.t0, y[m], color=C.SIG_COL[s], lw=1.0)
            ax.axvline(0, color="k", lw=0.6, ls="--")
            ax.set_ylabel(C.SIG_LABEL[s], fontsize=7)
            if i == 0:
                tag = " (ML-filled day)" if getattr(row, "ml_derived", False) else " (manual)"
                ax.set_title(f"{label}{tag}\n{row.key}  t0={row.t0:.0f}s", fontsize=7.5)
            if i == len(SIGS) - 1:
                ax.set_xlabel("time from N2W onset (s)", fontsize=7)
            ax.tick_params(labelsize=6)


# --------------------------------------------------------------------------- #
# 8B / 8E event-triggered averages
# --------------------------------------------------------------------------- #
def mouse_level_curve(ev: pd.DataFrame, rg: np.ndarray, M: np.ndarray):
    """Average within mouse first, then across mice: returns (mean, sem, n_mice)."""
    mice = ev["mouse"].to_numpy()
    uniq = pd.unique(mice)
    per_mouse = np.full((len(uniq), M.shape[1]), np.nan)
    for i, mo in enumerate(uniq):
        sel = mice == mo
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            per_mouse[i] = np.nanmean(M[sel], axis=0)
    n = np.sum(np.isfinite(per_mouse), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(per_mouse, axis=0)
        sem = np.nanstd(per_mouse, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mean, sem, len(uniq), per_mouse


ETA_YLABEL = {"pupil": "pupil (z)", "whisk": "whisker (z)",
             "art": "arteriole\nΔD/D₀", "ca": "astro Ca²⁺\nΔF/F₀"}


def panel_eta(axes, ev: pd.DataFrame, eps: dict, label_mode: str, event_label: str):
    """Event-triggered averages across SIGS for SCR vs KD; axes is 1D array len 4."""
    for i, s in enumerate(SIGS):
        ax = axes[i]
        rg, M = eps[s]
        for g in GROUPS:
            sel = ev["group"].to_numpy() == g
            if sel.sum() == 0:
                continue
            mean, sem, n_mice, _ = mouse_level_curve(ev[sel], rg, M[sel])
            ax.plot(rg, mean, color=C.GROUP_COL[g], lw=1.4,
                    label=f"{C.GROUP_LABEL[g]} (n={n_mice} days, {sel.sum()} ev)")
            ax.fill_between(rg, mean - sem, mean + sem, color=C.GROUP_COL[g], alpha=0.25, lw=0)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.axvspan(C.RESP_WIN[0], C.RESP_WIN[1], color="gray", alpha=0.08, lw=0)
        ax.set_ylabel(ETA_YLABEL[s], fontsize=7.5, labelpad=4)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right", frameon=False)
        if i == 0:
            ax.set_title(f"{event_label} event-triggered average ({label_mode})", fontsize=9)
        if i < len(SIGS) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("time from event onset (s)", fontsize=8)


# --------------------------------------------------------------------------- #
# 8C heat maps
# --------------------------------------------------------------------------- #
def panel_8C(fig, ev: pd.DataFrame, eps: dict):
    rg_pupil, M_pupil = eps["pupil"]
    peak_pupil, _, _ = C.response_metrics(rg_pupil, M_pupil)
    axes = fig.subplots(2, 2, gridspec_kw=dict(hspace=0.65, wspace=0.3))
    for col, g in enumerate(GROUPS):
        sel = np.flatnonzero(ev["group"].to_numpy() == g)
        if sel.size == 0:
            for row in range(2):
                axes[row, col].axis("off")
            continue
        order = sel[np.argsort(-np.nan_to_num(peak_pupil[sel], nan=-np.inf))]
        for row, s in enumerate(("art", "ca")):
            ax = axes[row, col]
            rg, M = eps[s]
            sub = M[order]
            vmax = np.nanpercentile(np.abs(sub), 95) if np.isfinite(sub).any() else 1.0
            vmax = vmax if vmax > 0 else 1.0
            im = ax.imshow(sub, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           extent=[rg[0], rg[-1], sub.shape[0], 0])
            ax.axvline(0, color="k", lw=0.6, ls="--")
            ax.set_title(f"{C.GROUP_LABEL[g]} — {C.SIG_LABEL[s]} (n={sub.shape[0]} ev)",
                        fontsize=7.5, pad=4)
            ax.set_xlabel("time (s)", fontsize=7)
            ax.set_ylabel("event (sorted by pupil peak)", fontsize=6.5)
            ax.tick_params(labelsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)


# --------------------------------------------------------------------------- #
# 8D / 8F quantification + stats
# --------------------------------------------------------------------------- #
QUANT_SIGNALS = {"ca_peak": "peak Ca2+ (dF/F0)", "art_peak": "peak arteriole (dD/D0)",
                 "ca_auc": "Ca2+ AUC", "art_auc": "arteriole AUC",
                 "ca_responder": "Ca2+ response prob.", "art_responder": "arteriole response prob.",
                 "ca_tpk": "Ca2+ time-to-peak (s)", "art_tpk": "arteriole time-to-peak (s)"}


def quant_panel(axes, d: pd.DataFrame, event_label: str):
    """d has one row per event, columns from metrics_table. axes: 1D array of 8."""
    mouse_means = d.groupby(["group", "mouse"], as_index=False)[list(QUANT_SIGNALS)].mean()
    for i, (col, title) in enumerate(QUANT_SIGNALS.items()):
        ax = axes[i]
        for gi, g in enumerate(GROUPS):
            ev_vals = d.loc[d["group"] == g, col].dropna().to_numpy()
            mo_vals = mouse_means.loc[mouse_means["group"] == g, col].dropna().to_numpy()
            jitter_e = gi + 0.15 * (RNG.random(len(ev_vals)) - 0.5) * 2
            ax.scatter(jitter_e, ev_vals, s=6, color=C.GROUP_COL[g], alpha=0.18, zorder=1)
            jitter_m = np.full(len(mo_vals), gi)
            jitter_m = gi + 0.08 * (RNG.random(len(mo_vals)) - 0.5) * 2
            ax.scatter(jitter_m, mo_vals, s=38, color=C.GROUP_COL[g], edgecolor="k",
                      linewidth=0.6, zorder=3)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([C.GROUP_LABEL[g] for g in GROUPS], fontsize=7)
        ax.set_ylabel(title, fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[len(QUANT_SIGNALS):]:
        ax.axis("off")


def run_mouse_stats(d: pd.DataFrame) -> pd.DataFrame:
    """group_test on mouse-level means for each quantity in QUANT_SIGNALS."""
    mouse_means = d.groupby(["group", "mouse"], as_index=False)[list(QUANT_SIGNALS)].mean()
    rows = []
    for col in QUANT_SIGNALS:
        a = mouse_means.loc[mouse_means["group"] == "SCR", col].to_numpy()
        b = mouse_means.loc[mouse_means["group"] == "KD", col].to_numpy()
        res = C.group_test(a, b)
        res["metric"] = col
        res["level"] = "mouse"
        rows.append(res)
    out = pd.DataFrame(rows)
    out["welch_p_bh"] = C.bh(out["welch_p"].to_numpy())
    out["mw_p_bh"] = C.bh(out["mw_p"].to_numpy())
    return out


def run_lmm(d: pd.DataFrame) -> pd.DataFrame:
    """Event-level LMM for peak Ca and peak art: value ~ C(group), random
    intercept mouse; and value ~ C(group) + pupil_peak."""
    frames = []
    for val in ("ca_peak", "art_peak"):
        dd = d.dropna(subset=[val, "group", "mouse"]).copy()
        t1 = C.lmm(dd, val, "C(group)")
        if t1 is not None:
            t1 = t1.copy()
            t1["response"] = val
            t1["model"] = "group_only"
            frames.append(t1)
        dd2 = dd.dropna(subset=["pupil_peak"])
        t2 = C.lmm(dd2, val, "C(group) + pupil_peak")
        if t2 is not None:
            t2 = t2.copy()
            t2["response"] = val
            t2["model"] = "group_plus_pupil"
            frames.append(t2)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 8G forest plot
# --------------------------------------------------------------------------- #
def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / (len(a) + len(b) - 2))
    if sp <= 0:
        return np.nan
    g = (b.mean() - a.mean()) / sp
    g *= 1 - 3 / (4 * (len(a) + len(b)) - 9)
    return g


def bootstrap_g_ci(a: np.ndarray, b: np.ndarray, n_boot=N_BOOT):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    g0 = hedges_g(a, b)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ra = RNG.choice(a, size=len(a), replace=True)
        rb = RNG.choice(b, size=len(b), replace=True)
        boots[i] = hedges_g(ra, rb)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return g0, lo, hi


def forest_entries(datasets: dict) -> pd.DataFrame:
    """datasets: label -> event-level df with group/mouse/<metric>_peak."""
    rows = []
    for label, (d, metric) in datasets.items():
        mm = d.groupby(["group", "mouse"], as_index=False)[metric].mean()
        a = mm.loc[mm["group"] == "SCR", metric].to_numpy()
        b = mm.loc[mm["group"] == "KD", metric].to_numpy()
        g, lo, hi = bootstrap_g_ci(a, b)
        rows.append(dict(label=label, g=g, lo=lo, hi=hi, n_scr=len(a[np.isfinite(a)]),
                         n_kd=len(b[np.isfinite(b)])))
    return pd.DataFrame(rows)


def panel_8G(ax, forest: pd.DataFrame):
    y = np.arange(len(forest))[::-1]
    for yi, r in zip(y, forest.itertuples()):
        ax.plot([r.lo, r.hi], [yi, yi], color="black", lw=1.3, zorder=1)
        ax.scatter([r.g], [yi], color="#C62828" if "ml" in r.label else "#1565C0",
                  s=40, zorder=2)
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.label}  (SCR n={r.n_scr}, KD n={r.n_kd})" for r in forest.itertuples()],
                       fontsize=7)
    ax.set_xlabel("Hedges g (KD vs SCR, mouse-level; bootstrap 95% CI, 2000 resamples)", fontsize=8)
    ax.set_title("Fig 8G — summary effect sizes", fontsize=10)
    ax.tick_params(labelsize=7)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    C.ensure_dirs()
    notes = []

    # ---- build datasets for both event types x both label modes -------- #
    data = {}  # (event_type, label_mode) -> dict(ev=, eps=, met=)
    for ev_type in ("N2W", "MA"):
        for label_mode in ("manual+ml", "manual"):
            ev = build_events(ev_type, label_mode)
            eps = epochs_dict(ev)
            met = metrics_table(ev, eps)
            data[(ev_type, label_mode)] = dict(ev=ev, eps=eps, met=met)
            n_days = ev.groupby("group")["mouse"].nunique().to_dict()
            n_ev = ev.groupby("group").size().to_dict()
            notes.append(f"- {ev_type} / {label_mode}: n days = {n_days}, n events = {n_ev}")

    primary = {k: v for k, v in data.items() if k[1] == "manual+ml"}

    # ===================== Fig 8A-D (N2W, primary label mode) =========== #
    n2w_primary = data[("N2W", "manual+ml")]
    ev_n2w, eps_n2w, met_n2w = n2w_primary["ev"], n2w_primary["eps"], n2w_primary["met"]

    fig = plt.figure(figsize=(14, 26))   # 8B event averages made taller (Grant: ETAs too flat)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 2.6, 1.4, 1.1], hspace=0.45)

    # 8A
    gsA = gs[0].subgridspec(4, 2, hspace=0.5, wspace=0.25)
    axA = np.array([[fig.add_subplot(gsA[r, c]) for c in range(2)] for r in range(4)])
    scr_row = pick_representative(ev_n2w, "SCR")
    kd_row = pick_representative(ev_n2w, "KD", prefer_manual=True)
    panel_8A(axA, scr_row, kd_row)
    axA[0, 0].annotate("8A", xy=(-0.25, 1.25), xycoords="axes fraction", fontsize=13, fontweight="bold")

    # 8B
    gsB = gs[1].subgridspec(4, 1, hspace=0.5)
    axB = np.array([fig.add_subplot(gsB[r]) for r in range(4)])
    panel_eta(axB, ev_n2w, eps_n2w, "manual+ml", "N2W")
    axB[0].annotate("8B", xy=(-0.08, 1.3), xycoords="axes fraction", fontsize=13, fontweight="bold")

    # 8C
    sub8C = fig.add_subfigure(gs[2])
    panel_8C(sub8C, ev_n2w, eps_n2w)
    sub8C.text(0.0, 1.02, "8C", fontsize=13, fontweight="bold", transform=sub8C.transSubfigure)

    # 8D
    gsD = gs[3].subgridspec(1, 8, wspace=0.6)
    axD = np.array([fig.add_subplot(gsD[0, i]) for i in range(8)])
    quant_panel(axD, met_n2w, "N2W")
    axD[0].annotate("8D", xy=(-0.35, 1.15), xycoords="axes fraction", fontsize=13, fontweight="bold")

    fig.suptitle("Figure 8A-D — N2W arousal events: SCR vs IP3R2 KD (manual+ml labels)\n" + C.PRELIM_STAMP,
                 fontsize=10, y=0.995)
    p1 = C.save_fig(fig, "fig8_A-D_N2W")
    plt.close(fig)

    stats_n2w_mouse = run_mouse_stats(met_n2w)
    stats_n2w_lmm = run_lmm(met_n2w)
    C.save_table(stats_n2w_mouse, "fig8_D_N2W_group_test")
    C.save_table(stats_n2w_lmm, "fig8_D_N2W_lmm")

    # also run the sensitivity (manual-only KD) quantification table for N2W
    met_n2w_manual = data[("N2W", "manual")]["met"]
    stats_n2w_manual_mouse = run_mouse_stats(met_n2w_manual)
    C.save_table(stats_n2w_manual_mouse, "fig8_D_N2W_group_test_manual_sensitivity")

    # ===================== Fig 8E-F (MA, primary label mode) ============= #
    ma_primary = data[("MA", "manual+ml")]
    ev_ma, eps_ma, met_ma = ma_primary["ev"], ma_primary["eps"], ma_primary["met"]

    fig2 = plt.figure(figsize=(14, 16))   # 8E event averages made taller
    gs2 = fig2.add_gridspec(2, 1, height_ratios=[2.6, 1.1], hspace=0.35)

    gsE = gs2[0].subgridspec(4, 1, hspace=0.5)
    axE = np.array([fig2.add_subplot(gsE[r]) for r in range(4)])
    panel_eta(axE, ev_ma, eps_ma, "manual+ml", "MA")
    axE[0].annotate("8E", xy=(-0.08, 1.3), xycoords="axes fraction", fontsize=13, fontweight="bold")

    gsF = gs2[1].subgridspec(1, 8, wspace=0.6)
    axF = np.array([fig2.add_subplot(gsF[0, i]) for i in range(8)])
    quant_panel(axF, met_ma, "MA")
    axF[0].annotate("8F", xy=(-0.35, 1.15), xycoords="axes fraction", fontsize=13, fontweight="bold")

    fig2.suptitle("Figure 8E-F — micro-arousal (MA) events: SCR vs IP3R2 KD (manual+ml labels)\n" + C.PRELIM_STAMP,
                  fontsize=10, y=0.995)
    p2 = C.save_fig(fig2, "fig8_E-F_MA")
    plt.close(fig2)

    stats_ma_mouse = run_mouse_stats(met_ma)
    stats_ma_lmm = run_lmm(met_ma)
    C.save_table(stats_ma_mouse, "fig8_F_MA_group_test")
    C.save_table(stats_ma_lmm, "fig8_F_MA_lmm")

    met_ma_manual = data[("MA", "manual")]["met"]
    stats_ma_manual_mouse = run_mouse_stats(met_ma_manual)
    C.save_table(stats_ma_manual_mouse, "fig8_F_MA_group_test_manual_sensitivity")

    # BH across 8D + 8F primary (manual+ml) welch p-values together
    combo = pd.concat([
        stats_n2w_mouse.assign(panel="8D_N2W"),
        stats_ma_mouse.assign(panel="8F_MA"),
    ], ignore_index=True)
    combo["welch_p_bh_combined"] = C.bh(combo["welch_p"].to_numpy())
    combo["mw_p_bh_combined"] = C.bh(combo["mw_p"].to_numpy())
    C.save_table(combo, "fig8_D_F_combined_BH")

    # ===================== Fig 8G summary forest plot ===================== #
    forest_sets = {
        "N2W arteriole (manual+ml)": (data[("N2W", "manual+ml")]["met"], "art_peak"),
        "N2W Ca2+ (manual+ml)": (data[("N2W", "manual+ml")]["met"], "ca_peak"),
        "MA arteriole (manual+ml)": (data[("MA", "manual+ml")]["met"], "art_peak"),
        "MA Ca2+ (manual+ml)": (data[("MA", "manual+ml")]["met"], "ca_peak"),
        "N2W arteriole (manual, sensitivity)": (data[("N2W", "manual")]["met"], "art_peak"),
        "N2W Ca2+ (manual, sensitivity)": (data[("N2W", "manual")]["met"], "ca_peak"),
        "MA arteriole (manual, sensitivity)": (data[("MA", "manual")]["met"], "art_peak"),
        "MA Ca2+ (manual, sensitivity)": (data[("MA", "manual")]["met"], "ca_peak"),
    }
    forest = forest_entries(forest_sets)
    C.save_table(forest, "fig8_G_forest")

    figG, axG = plt.subplots(figsize=(9, 5.5))
    panel_8G(axG, forest)
    figG.suptitle(C.PRELIM_STAMP, fontsize=7, y=0.02)
    p3 = C.save_fig(figG, "fig8_G_summary")
    plt.close(figG)

    # ---- write notes ----------------------------------------------------- #
    headline = []
    for r in forest.itertuples():
        if np.isfinite(r.g):
            headline.append(f"{r.label}: g={r.g:.2f} [{r.lo:.2f}, {r.hi:.2f}] (SCR n={r.n_scr}, KD n={r.n_kd})")

    note_lines = [
        "# Figure 8 — IP3R2 KD vs scramble arousal-event responses",
        "",
        "## Panel summaries",
        "- 8A: representative N2W events, one per group (pupil/whisker/arteriole/Ca). "
        "KD example drawn preferentially from manual-only days when available.",
        "- 8B: N2W event-triggered averages, mouse mean +/- SEM (averaged within mouse "
        "first), SCR vs KD, manual+ml labels. Pupil/whisker shown first to confirm "
        "arousal magnitude is comparable before interpreting art/Ca.",
        "- 8C: heat maps of arteriole and Ca responses to N2W events, SCR and KD panels "
        "separate, rows sorted by each event's pupil-peak response.",
        "- 8D: per-mouse quantification (peak Ca, peak arteriole, AUC, response "
        "probability, time-to-peak) for N2W, with mouse-level group_test and "
        "event-level LMM (plain and pupil-peak-adjusted).",
        "- 8E/8F: same as 8B/8D but for micro-arousal (MA) events.",
        "- 8G: forest plot of Hedges g (KD vs SCR, mouse-level, mean peak per mouse) "
        "with bootstrap 95% CI (2000 resamples, resampling mice within group) for "
        "N2W/MA x art/Ca, in both manual+ml (primary) and manual (sensitivity) label modes.",
        "",
        "## Event counts (n days / n events)",
    ]
    for line in notes:
        note_lines.append(line)
    note_lines += [
        "",
        "## Headline effect sizes (Hedges g, mouse-level, bootstrap 95% CI)",
    ]
    note_lines += [f"- {h}" for h in headline]
    note_lines += [
        "",
        "## BH-corrected mouse-level group tests (primary, manual+ml)",
        "See tables fig8_D_N2W_group_test.csv and fig8_F_MA_group_test.csv for "
        "per-metric Welch p / Mann-Whitney p and their BH-adjusted values; "
        "fig8_D_F_combined_BH.csv applies BH jointly across both event types' "
        "8 metrics each (16 tests total).",
        "",
        "## Caveats",
        "- KD sensitivity (manual-only) N2W has only 2 recording days with any N2W "
        "events (n=5 events total) and MA has that plus a 3rd day for 55 events; "
        "mouse-level stats on manual-only N2W are therefore descriptive at best "
        "(n=2 KD days) — see fig8_D_N2W_group_test_manual_sensitivity.csv.",
        "- SCR has no ML-derived labels; all SCR events are manual by construction.",
        "- ML-derived KD events are flagged in the underlying event table "
        "(column 'ml_derived') but not visually distinguished in 8B/8C pooled "
        "curves/heatmaps because under manual+ml, manual labels always win and "
        "only unlabelled bins are filled; entire-day ML inclusion is noted in 8A "
        "titles instead.",
        "- LMM fixed-effect tables may show 'FIT_FAILED' or be omitted if too few "
        "mice/observations (common.lmm requires >=3 mice and >=8 observations); "
        "check fig8_D_N2W_lmm.csv / fig8_F_MA_lmm.csv for the actual outcome.",
        "- Unit of inference is recording day ('mouse' in common.py); event- and "
        "cycle-level points are shown faintly behind mouse-level means in 8D/8F.",
        "- Response probability (C.responder) uses event's own pre-event SD "
        "(-30..-5s) and may be unstable for mice with very few events.",
        "",
        "## Reproduce",
        "`./venv/bin/python -B -m scripts.figs_prelim.fig8` from the repo root.",
        "",
        f"Figures: {p1.name}, {p2.name}, {p3.name}",
    ]
    notes_path = C.OUT / "notes" / "fig8.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text("\n".join(note_lines) + "\n")

    print("Saved:", p1, p2, p3)
    print("Notes:", notes_path)
    print("\nHeadline effect sizes:")
    for h in headline:
        print(" ", h)


if __name__ == "__main__":
    main()
