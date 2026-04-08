"""
Event-aligned multimodal analysis (-5s baseline, +15s event) from clean event CSVs.

Reads clean event files, loads pupil_standardized, arteriole-diameter, and calcium from
data/pupil_whisker_arteriole, aligns traces to Start_Time, computes metrics and PSTHs,
aggregates cycle -> day -> group, saves figures to data/figures and tables to data/analysis.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Default styling aligned with plotting.ipynb run_full_pipeline
DEFAULT_DPI = 120
COLORS = {
    "pupil": "#6A1B9A",
    "arteriole": "#C62828",
    "calcium": "#1565C0",
    "scramble": "#2E7D32",
    "KD": "#C62828",
}


@dataclass
class ParsedPrefix:
    group: str  # "scramble" | "KD" | "unknown"
    date: str  # YYYY.MM.DD
    cycle: int
    day_key: str  # e.g. IP3R2_KD_2023.08.25 — all cycles on same calendar day
    prefix: str  # full basename without _clean_events


def parse_event_prefix(prefix: str) -> ParsedPrefix:
    """Parse filename prefix like IP3R2_KD_2023.08.25_cycle5."""
    m = re.match(
        r"^(?P<line>IP3R2_scramble|IP3R2_KD)_(?P<date>\d{4}\.\d{2}\.\d{2})_cycle(?P<cy>\d+)$",
        prefix,
    )
    if not m:
        return ParsedPrefix(
            group="unknown",
            date="",
            cycle=-1,
            day_key=prefix,
            prefix=prefix,
        )
    line = m.group("line")
    date = m.group("date")
    cycle = int(m.group("cy"))
    group = "scramble" if line == "IP3R2_scramble" else "KD"
    day_key = f"{line}_{date}"
    return ParsedPrefix(
        group=group,
        date=date,
        cycle=cycle,
        day_key=day_key,
        prefix=prefix,
    )


def event_basename_to_prefix(base: str) -> Optional[str]:
    """IP3R2_KD_2023.08.25_cycle5_clean_events.csv -> IP3R2_KD_2023.08.25_cycle5."""
    for suf in ("_clean_events.csv", "_final_events_clean_events.csv"):
        if base.endswith(suf):
            return base[: -len(suf)]
    return None


def load_timeseries_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load (time, value) from CSV; uses first two columns."""
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    if df.shape[1] < 2:
        raise ValueError(f"Need >=2 columns: {path}")
    t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 2:
        raise ValueError(f"Not enough finite samples: {path}")
    order = np.argsort(t)
    return t[order], y[order]


def make_rel_grid(t_pre: float, t_post: float, dt: float) -> np.ndarray:
    n = int(round((t_pre + t_post) / dt)) + 1
    return np.linspace(-t_pre, t_post, n, endpoint=True)


def interp_aligned(
    t: np.ndarray,
    y: np.ndarray,
    t_onset: float,
    t_pre: float,
    t_post: float,
    rel_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate y onto relative time grid (t_onset + rel_grid)."""
    targ = t_onset + rel_grid
    # require coverage
    if targ[0] < t[0] - 1e-6 or targ[-1] > t[-1] + 1e-6:
        return np.full_like(rel_grid, np.nan, dtype=float)
    return np.interp(targ, t, y, left=np.nan, right=np.nan)


def baseline_window_mask(rel_grid: np.ndarray, t_pre: float) -> np.ndarray:
    return (rel_grid >= -t_pre) & (rel_grid <= 0)


def event_window_mask(rel_grid: np.ndarray) -> np.ndarray:
    return (rel_grid >= 0) & (rel_grid <= rel_grid[-1])  # 0 .. t_post


def event_segment_mask(rel_grid: np.ndarray, t_post: float) -> np.ndarray:
    return (rel_grid >= 0) & (rel_grid <= t_post)


def arteriole_dd_over_d0(y_aligned: np.ndarray, rel_grid: np.ndarray, t_pre: float) -> np.ndarray:
    m = baseline_window_mask(rel_grid, t_pre)
    d0 = np.nanmean(y_aligned[m])
    if not np.isfinite(d0) or abs(d0) < 1e-12:
        return np.full_like(y_aligned, np.nan)
    return (y_aligned - d0) / d0


def calcium_baseline_correct(y_aligned: np.ndarray, rel_grid: np.ndarray, t_pre: float) -> np.ndarray:
    m = baseline_window_mask(rel_grid, t_pre)
    b = np.nanmean(y_aligned[m])
    return y_aligned - b if np.isfinite(b) else y_aligned


def _trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def trapz_event(y: np.ndarray, rel_grid: np.ndarray, t_pre: float, t_post: float) -> float:
    m = event_segment_mask(rel_grid, t_post)
    if not np.any(m) or np.sum(m) < 2:
        return np.nan
    return _trapz_compat(y[m], rel_grid[m])


def max_in_event(y: np.ndarray, rel_grid: np.ndarray, t_post: float) -> float:
    m = event_segment_mask(rel_grid, t_post)
    if not np.any(m):
        return np.nan
    return float(np.nanmax(y[m]))


def arteriole_tmax_sec(dd: np.ndarray, rel_grid: np.ndarray, t_post: float) -> float:
    m = event_segment_mask(rel_grid, t_post)
    if not np.any(m):
        return np.nan
    idx = np.nanargmax(dd[m])
    return float(rel_grid[m][idx])


def arteriole_rise_slope_max(
    y_raw: np.ndarray, rel_grid: np.ndarray, t_pre: float, window_post: float = 5.0
) -> float:
    """Max dy/dt in first `window_post` s after onset (on raw diameter before %)."""
    m = (rel_grid >= 0) & (rel_grid <= window_post)
    if np.sum(m) < 2:
        return np.nan
    rg = rel_grid[m]
    yy = y_raw[m]
    dt = np.diff(rg)
    dy = np.diff(yy)
    valid = dt > 0
    if not np.any(valid):
        return np.nan
    slopes = dy[valid] / dt[valid]
    return float(np.nanmax(slopes))


def calcium_onset_latency(
    y_bc: np.ndarray,
    rel_grid: np.ndarray,
    t_pre: float,
    t_post: float,
    n_std: float = 2.0,
    min_sustain: int = 3,
) -> float:
    """Seconds from t=0 to first sustained rise above baseline_mean + n_std * baseline_std."""
    mb = baseline_window_mask(rel_grid, t_pre)
    bmean = np.nanmean(y_bc[mb])
    bstd = np.nanstd(y_bc[mb])
    if not np.isfinite(bmean) or not np.isfinite(bstd):
        return np.nan
    thresh = bmean + n_std * bstd
    me = event_segment_mask(rel_grid, t_post)
    idx0 = np.where(me)[0]
    if len(idx0) == 0:
        return np.nan
    above = y_bc[me] > thresh
    # first run of length >= min_sustain
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if j - i >= min_sustain:
                return float(rel_grid[me][i])
            i = j
        else:
            i += 1
    return np.nan


def sem(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.asarray(x, float)
    n = np.sum(np.isfinite(x), axis=axis)
    s = np.nanstd(x, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / np.sqrt(np.maximum(n, 1))


def plot_group_psth(
    rel_grid: np.ndarray,
    mean_kd: np.ndarray,
    sem_kd: np.ndarray,
    mean_sc: np.ndarray,
    sem_sc: np.ndarray,
    title: str,
    ylabel: str,
    out_path: str,
    dpi: int = DEFAULT_DPI,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.axvspan(-5, 0, color="#ECEFF1", alpha=0.6, lw=0)
    if np.any(np.isfinite(mean_kd)):
        ax.plot(rel_grid, mean_kd, color=COLORS["KD"], label="KD", lw=2)
        ax.fill_between(
            rel_grid,
            mean_kd - sem_kd,
            mean_kd + sem_kd,
            color=COLORS["KD"],
            alpha=0.2,
        )
    if np.any(np.isfinite(mean_sc)):
        ax.plot(rel_grid, mean_sc, color=COLORS["scramble"], label="Scramble", lw=2)
        ax.fill_between(
            rel_grid,
            mean_sc - sem_sc,
            mean_sc + sem_sc,
            color=COLORS["scramble"],
            alpha=0.2,
        )
    ax.set_xlabel("Time rel. onset (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    out_path: str,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Strip plot + mean ± SEM by group (scramble vs KD)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    groups = ["scramble", "KD"]
    positions = [0, 1]
    xs_all: List[float] = []
    ys_all: List[float] = []
    for g, pos in zip(groups, positions):
        sub = df[df["group"] == g][value_col].dropna()
        if len(sub) == 0:
            continue
        x = np.random.normal(pos, 0.05, size=len(sub))
        color = COLORS.get(g, "#333")
        ax.scatter(x, sub, alpha=0.5, s=36, color=color, edgecolors="none")
        xs_all.extend(x)
        ys_all.extend(sub.tolist())
        m = float(np.nanmean(sub))
        s = float(np.nanstd(sub) / np.sqrt(len(sub)))
        ax.errorbar(pos, m, yerr=s, fmt="o", color="black", capsize=6, ms=8, zorder=5)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Scramble", "KD"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_event_aligned_group_analysis(
    events_glob: str = "data/events/*_clean_events.csv",
    signal_dir: str = "data/pupil_whisker_arteriole",
    out_fig_dir: str = "data/figures",
    out_data_dir: str = "data/analysis",
    t_pre: float = 5.0,
    t_post: float = 15.0,
    dt: float = 0.02,
    dpi: int = DEFAULT_DPI,
    compute_optional: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Main entry: load events, align signals, compute metrics, aggregate, save CSVs and figures.

    Returns dict with paths to saved artifacts and summary counts.
    """
    os.makedirs(out_fig_dir, exist_ok=True)
    os.makedirs(out_data_dir, exist_ok=True)

    rel_grid = make_rel_grid(t_pre, t_post, dt)
    event_paths = sorted(glob.glob(events_glob))
    if not event_paths:
        raise FileNotFoundError(f"No event files matched: {events_glob}")

    rows: List[Dict[str, Any]] = []
    # Per-cycle stacks for PSTH (list of aligned arrays per signal type)
    cycle_psth: Dict[str, Dict[str, List[np.ndarray]]] = {}

    for ev_path in event_paths:
        base = os.path.basename(ev_path)
        prefix = event_basename_to_prefix(base)
        if prefix is None:
            if verbose:
                print(f"[skip] unexpected event filename: {base}")
            continue

        meta = parse_event_prefix(prefix)
        pupil_p = os.path.join(signal_dir, f"{prefix}_pupil_standardized.csv")
        art_p = os.path.join(signal_dir, f"{prefix}_arteriole-diameter.csv")
        ca_p = os.path.join(signal_dir, f"{prefix}_calcium.csv")

        if not all(os.path.exists(p) for p in (pupil_p, art_p, ca_p)):
            if verbose:
                print(f"[skip] missing signal files for {prefix}")
            continue

        try:
            t_p, y_p = load_timeseries_csv(pupil_p)
            t_a, y_a = load_timeseries_csv(art_p)
            t_c, y_c = load_timeseries_csv(ca_p)
        except Exception as e:
            if verbose:
                print(f"[skip] load error {prefix}: {e}")
            continue

        ev_df = pd.read_csv(ev_path)
        if "Start_Time" in ev_df.columns:
            start_col = "Start_Time"
        elif "Start" in ev_df.columns:
            start_col = "Start"
        else:
            if verbose:
                print(f"[skip] no Start column: {base}")
            continue

        if prefix not in cycle_psth:
            cycle_psth[prefix] = {
                "group": meta.group,
                "day_key": meta.day_key,
                "pupil": [],
                "arteriole_dd": [],
                "calcium_bc": [],
            }

        for ev_i, row in ev_df.iterrows():
            t0 = float(row[start_col])
            if not np.isfinite(t0):
                continue

            p_w = interp_aligned(t_p, y_p, t0, t_pre, t_post, rel_grid)
            a_w = interp_aligned(t_a, y_a, t0, t_pre, t_post, rel_grid)
            c_w = interp_aligned(t_c, y_c, t0, t_pre, t_post, rel_grid)

            if np.all(~np.isfinite(p_w)) or np.all(~np.isfinite(a_w)) or np.all(~np.isfinite(c_w)):
                continue

            dd = arteriole_dd_over_d0(a_w, rel_grid, t_pre)
            ca_bc = calcium_baseline_correct(c_w, rel_grid, t_pre)

            # Metrics on event window 0..t_post
            art_peak = max_in_event(dd, rel_grid, t_post)
            art_auc = trapz_event(dd, rel_grid, t_pre, t_post)
            ca_peak = max_in_event(ca_bc, rel_grid, t_post)
            ca_auc = trapz_event(ca_bc, rel_grid, t_pre, t_post)

            row_dict: Dict[str, Any] = {
                "prefix": prefix,
                "group": meta.group,
                "date": meta.date,
                "cycle": meta.cycle,
                "day_key": meta.day_key,
                "event_idx": int(ev_i),
                "start_time": t0,
                "pupil_peak_event": max_in_event(p_w, rel_grid, t_post),
                "pupil_auc_event": trapz_event(p_w, rel_grid, t_pre, t_post),
                "arteriole_peak_dd": art_peak,
                "arteriole_auc_dd": art_auc,
                "calcium_peak_bc": ca_peak,
                "calcium_auc_bc": ca_auc,
            }
            if compute_optional:
                row_dict["arteriole_tmax_sec"] = arteriole_tmax_sec(dd, rel_grid, t_post)
                row_dict["arteriole_rise_slope_max"] = arteriole_rise_slope_max(
                    a_w, rel_grid, t_pre, window_post=5.0
                )
                row_dict["calcium_onset_latency_sec"] = calcium_onset_latency(
                    ca_bc, rel_grid, t_pre, t_post
                )
            rows.append(row_dict)

            cycle_psth[prefix]["pupil"].append(np.asarray(p_w, float))
            cycle_psth[prefix]["arteriole_dd"].append(np.asarray(dd, float))
            cycle_psth[prefix]["calcium_bc"].append(np.asarray(ca_bc, float))

    events_df = pd.DataFrame(rows)
    event_csv = os.path.join(out_data_dir, "event_metrics.csv")
    events_df.to_csv(event_csv, index=False)
    if verbose:
        print(f"[ok] wrote {len(events_df)} event rows -> {event_csv}")

    # --- Cycle-level mean PSTH ---
    cycle_keys = list(cycle_psth.keys())
    day_to_cycles: Dict[str, List[str]] = {}
    for ck in cycle_keys:
        dk = cycle_psth[ck]["day_key"]
        day_to_cycles.setdefault(dk, []).append(ck)

    def mean_stack(lst: List[np.ndarray]) -> np.ndarray:
        if not lst:
            return np.full_like(rel_grid, np.nan)
        a = np.stack(lst, axis=0)
        return np.nanmean(a, axis=0)

    cycle_mean_psth: Dict[str, Dict[str, np.ndarray]] = {}
    for ck, d in cycle_psth.items():
        cycle_mean_psth[ck] = {
            "pupil": mean_stack(d["pupil"]),
            "arteriole_dd": mean_stack(d["arteriole_dd"]),
            "calcium_bc": mean_stack(d["calcium_bc"]),
            "group": d["group"],
        }

    # --- Day-level: mean of cycle PSTHs sharing day_key ---
    day_psth: Dict[str, Dict[str, Any]] = {}
    for day_key, cks in day_to_cycles.items():
        pups = [cycle_mean_psth[c]["pupil"] for c in cks if c in cycle_mean_psth]
        arts = [cycle_mean_psth[c]["arteriole_dd"] for c in cks if c in cycle_mean_psth]
        cals = [cycle_mean_psth[c]["calcium_bc"] for c in cks if c in cycle_mean_psth]
        g0 = cycle_psth[cks[0]]["group"]
        day_psth[day_key] = {
            "group": g0,
            "pupil": mean_stack(pups),
            "arteriole_dd": mean_stack(arts),
            "calcium_bc": mean_stack(cals),
        }

    # --- Group PSTH: mean across day_keys (each day one replicate) ---
    def group_psth_arrays(kind: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        kd_days = [day_psth[k][kind] for k in day_psth if day_psth[k]["group"] == "KD"]
        sc_days = [day_psth[k][kind] for k in day_psth if day_psth[k]["group"] == "scramble"]
        kd_m = np.nanmean(np.stack(kd_days, axis=0), axis=0) if kd_days else np.full_like(rel_grid, np.nan)
        kd_s = sem(np.stack(kd_days, axis=0), axis=0) if kd_days else np.full_like(rel_grid, np.nan)
        sc_m = np.nanmean(np.stack(sc_days, axis=0), axis=0) if sc_days else np.full_like(rel_grid, np.nan)
        sc_s = sem(np.stack(sc_days, axis=0), axis=0) if sc_days else np.full_like(rel_grid, np.nan)
        return kd_m, kd_s, sc_m, sc_s

    kd_p, kd_ps, sc_p, sc_ps = group_psth_arrays("pupil")
    kd_a, kd_as, sc_a, sc_as = group_psth_arrays("arteriole_dd")
    kd_c, kd_cs, sc_c, sc_cs = group_psth_arrays("calcium_bc")

    plot_group_psth(
        rel_grid,
        kd_p,
        kd_ps,
        sc_p,
        sc_ps,
        "Group PSTH: standardized pupil (mean ± SEM across days)",
        "Pupil (standardized)",
        os.path.join(out_fig_dir, "group_psth_pupil.png"),
        dpi=dpi,
    )
    plot_group_psth(
        rel_grid,
        kd_a,
        kd_as,
        sc_a,
        sc_as,
        "Group PSTH: arteriole ΔD/D₀ (mean ± SEM across days)",
        "ΔD / D₀",
        os.path.join(out_fig_dir, "group_psth_arteriole_dd_over_d0.png"),
        dpi=dpi,
    )
    plot_group_psth(
        rel_grid,
        kd_c,
        kd_cs,
        sc_c,
        sc_cs,
        "Group PSTH: calcium (baseline-subtracted, mean ± SEM across days)",
        "Calcium (a.u., baseline-subtracted)",
        os.path.join(out_fig_dir, "group_psth_calcium.png"),
        dpi=dpi,
    )

    # --- Metric plots (event-level) ---
    if len(events_df) > 0:
        for col, title, ylab, fname in [
            ("arteriole_peak_dd", "Peak arteriole ΔD/D₀ (0–15 s)", "ΔD/D₀", "metric_arteriole_peak.png"),
            ("arteriole_auc_dd", "AUC arteriole ΔD/D₀ (0–15 s)", "AUC (a.u.·s)", "metric_arteriole_auc.png"),
            ("calcium_peak_bc", "Peak calcium (baseline-subtracted, 0–15 s)", "ΔF (a.u.)", "metric_calcium_peak.png"),
            ("calcium_auc_bc", "AUC calcium (baseline-subtracted, 0–15 s)", "AUC (a.u.·s)", "metric_calcium_auc.png"),
        ]:
            if col in events_df.columns:
                plot_metric_comparison(
                    events_df,
                    col,
                    title,
                    ylab,
                    os.path.join(out_fig_dir, fname),
                    dpi=dpi,
                )
        if compute_optional:
            for col, title, ylab, fname in [
                (
                    "arteriole_tmax_sec",
                    "Time to peak arteriole dilation (0–15 s)",
                    "Time (s)",
                    "metric_arteriole_tmax.png",
                ),
                (
                    "arteriole_rise_slope_max",
                    "Max rise slope arteriole (0–5 s, raw diameter)",
                    "Slope (a.u./s)",
                    "metric_arteriole_rise_slope.png",
                ),
                (
                    "calcium_onset_latency_sec",
                    "Calcium onset latency (threshold: baseline + 2 SD)",
                    "Latency (s)",
                    "metric_calcium_onset_latency.png",
                ),
            ]:
                if col in events_df.columns:
                    plot_metric_comparison(
                        events_df,
                        col,
                        title,
                        ylab,
                        os.path.join(out_fig_dir, fname),
                        dpi=dpi,
                    )

    # Save aggregated tables
    day_rows = []
    for dk, d in day_psth.items():
        day_rows.append({"day_key": dk, "group": d["group"], "n_cycles": len(day_to_cycles.get(dk, []))})
    pd.DataFrame(day_rows).to_csv(os.path.join(out_data_dir, "day_summary.csv"), index=False)

    if len(events_df) > 0:
        mouse_metrics = events_df.groupby("group", dropna=False).agg(
            n_events=("start_time", "count"),
            arteriole_peak_dd_mean=("arteriole_peak_dd", "mean"),
            arteriole_auc_dd_mean=("arteriole_auc_dd", "mean"),
            calcium_peak_bc_mean=("calcium_peak_bc", "mean"),
            calcium_auc_bc_mean=("calcium_auc_bc", "mean"),
        )
        mouse_metrics.to_csv(os.path.join(out_data_dir, "group_metrics.csv"))
    else:
        pd.DataFrame().to_csv(os.path.join(out_data_dir, "group_metrics.csv"))

    result = {
        "event_metrics_csv": event_csv,
        "n_events": len(events_df),
        "figures": [
            "group_psth_pupil.png",
            "group_psth_arteriole_dd_over_d0.png",
            "group_psth_calcium.png",
        ],
        "day_summary_csv": os.path.join(out_data_dir, "day_summary.csv"),
        "group_metrics_csv": os.path.join(out_data_dir, "group_metrics.csv"),
    }
    if verbose:
        print("[ok] done:", result)
    return result


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Event-aligned multimodal group analysis")
    p.add_argument("--events-glob", default="data/events/*_clean_events.csv")
    p.add_argument("--signal-dir", default="data/pupil_whisker_arteriole")
    p.add_argument("--out-fig", default="data/figures")
    p.add_argument("--out-data", default="data/analysis")
    p.add_argument("--t-pre", type=float, default=5.0)
    p.add_argument("--t-post", type=float, default=15.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--no-optional", action="store_true", help="Skip tmax/slope/latency")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()
    run_event_aligned_group_analysis(
        events_glob=args.events_glob,
        signal_dir=args.signal_dir,
        out_fig_dir=args.out_fig,
        out_data_dir=args.out_data,
        t_pre=args.t_pre,
        t_post=args.t_post,
        dt=args.dt,
        dpi=args.dpi,
        compute_optional=not args.no_optional,
        verbose=not args.quiet,
    )
