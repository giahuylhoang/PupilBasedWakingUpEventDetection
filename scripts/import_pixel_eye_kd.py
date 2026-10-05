"""Turn Florence's pixel-based pupil trace for IP3R2 KD into an eye-speed track
for the labeling site.

Input: "pupil moving speed for ip3R2 kd.xlsx" (repo root). One sheet per date,
no headers, one value per video frame (50 fps), one column per cycle with empty
spacer columns in between.

What the values are (inspected 2026-10-05): NOT a speed. They are smooth,
positive, ~40-110 px with SD 1-10 px, and track pupil size (r up to 0.85 at 5 s).
That matches ONE coordinate of the pupil centre in pixels: in our DLC data the
centre has SD ~4-5 px and also correlates 0.4-0.9 with diameter, whereas the
diameter itself has SD 20-37 px. So we treat each column as a 1-D pupil-centre
position and differentiate it.

Processing per cycle (mirrors scripts/pupil_eye_motion.py where possible):
  1. position outliers |robust z| > 5 (blinks / tracking loss) -> NaN
  2. interpolate gaps <= 1 s; longer gaps stay NaN
  3. 10 Hz 4th-order Butterworth low-pass (filtfilt), per finite segment.
     This also removes the duplicated-frame staircase in the 2023 videos.
  4. speed = |d position| in px/frame (x50 for px/s)
  5. eye_speed = robust z of speed within the cycle (median / 1.4826 MAD), so
     cycles from different cameras/crops display on the same scale.
Caveat: 1-D speed misses movement along the other axis, so it underestimates
true 2-D eye speed. Use for labelling, not for comparing with DLC speed.

Cycle mapping: columns are assumed to be in ascending cycle order. The script
checks it: each column is correlated (5-s medians) with every candidate cycle's
pupil trace and the best match is reported.

Outputs: data/pupil_eye_motion_pixel_2026-10-05/
    <key>_eyemotion_perframe.csv   time, pos_px_raw, pos_px_clean,
                                   speed_px_per_frame, eye_speed (robust z)
    mapping.csv, qc/<key>.png
Kept apart from data/pupil_eye_motion/ so analyses never mix the two methods.

Usage: ./venv/bin/python -B scripts/import_pixel_eye_kd.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parent / "figs_prelim"))
import common as C  # noqa: E402

XLSX = C.ROOT / "pupil moving speed for ip3R2 kd.xlsx"
OUT = C.ROOT / "data" / "pupil_eye_motion_pixel_2026-10-05"
FPS = 50.0
OUTLIER_Z = 5.0
MAX_GAP_S = 1.0
LP_CUT = 10.0
# sheet name -> (site date, cycles in column order). Sheet "2025.11.25" has no
# match on the site; its first two columns match 2025.10.25 cycles 9 and 10.
# Its third column has no cycle on the site and is skipped.
SHEETS = {
    "2023.08.25": ("2023.08.25", [5, 6, 7]),
    "2023.08.26": ("2023.08.26", [4, 5, 6]),
    "2024.02.18": ("2024.02.18", [5, 6, 7]),
    "2024.06.01": ("2024.06.01", [3, 4, 5, 6, 7]),
    "2025.11.25": ("2025.10.25", [9, 10, None]),
    "2025.11.01": ("2025.11.01", [9, 10, 11]),
}


def robust_z(y):
    med = np.nanmedian(y)
    mad = 1.4826 * np.nanmedian(np.abs(y - med))
    return (y - med) / mad if mad > 0 else y * np.nan


def clean_position(x):
    x = x.copy()
    # outliers relative to a 5-s running median, so a camera/ROI step (e.g.
    # 2023.08.25 cycle 6 at ~140 s) is not mistaken for 140 s of bad frames
    local = pd.Series(x).rolling(int(5 * FPS), center=True, min_periods=25).median().to_numpy()
    x[np.abs(robust_z(x - local)) > OUTLIER_Z * 2] = np.nan
    s = pd.Series(x).interpolate(limit=int(MAX_GAP_S * FPS), limit_area="inside")
    x = s.to_numpy()
    b, a = butter(4, LP_CUT / (FPS / 2), "low")
    out = np.full_like(x, np.nan)
    fin = np.isfinite(x)
    edges = np.flatnonzero(np.diff(np.r_[0, fin.astype(int), 0]))
    for i0, i1 in zip(edges[::2], edges[1::2]):
        if i1 - i0 > 3 * 13:                       # filtfilt needs > 3*(order*2+1) samples
            out[i0:i1] = filtfilt(b, a, x[i0:i1])
    return out


def match_scores(x, date, cycles):
    """corr(5-s medians of column, 5-s medians of each candidate cycle's pupil)."""
    t = np.arange(len(x)) / FPS
    xs = pd.Series(x).rolling(int(5 * FPS), center=True, min_periods=50).median().to_numpy()
    sc = {}
    for c in cycles:
        tp, p = C.read_trace(f"IP3R2_KD_{date}_cycle{c}", "pupil")
        if tp is None:
            continue
        pi = np.interp(t, tp, p, left=np.nan, right=np.nan)
        ps = pd.Series(pi).rolling(int(5 * FPS), center=True, min_periods=50).median().to_numpy()
        m = np.isfinite(xs) & np.isfinite(ps)
        sc[c] = np.corrcoef(xs[m], ps[m])[0, 1] if m.sum() > 1000 else np.nan
    return sc


def qc_plot(key, t, raw, pos, spd, z, path):
    fig, axs = plt.subplots(3, 1, figsize=(15, 7), sharex=True, constrained_layout=True)
    axs[0].plot(t, raw, color="0.7", lw=0.5, label="raw")
    axs[0].plot(t, pos, color="#5D4037", lw=0.7, label="cleaned")
    axs[0].set_ylabel("pupil centre (px)"); axs[0].legend(fontsize=7, frameon=False)
    tp, p = C.read_trace(key, "pupil")
    if tp is not None:
        axs[1].plot(tp, p, color="#1565C0", lw=0.5)
    axs[1].set_ylabel("pupil size (site trace)")
    axs[2].plot(t, z, color="#00897B", lw=0.5)
    axs[2].set_ylabel("eye speed (robust z)"); axs[2].set_xlabel("time (s)")
    fig.suptitle(f"{key}: pixel-based pupil position -> eye speed  "
                 f"(median {np.nanmedian(spd) * FPS:.1f} px/s)", fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=100); plt.close(fig)


def main():
    (OUT / "qc").mkdir(parents=True, exist_ok=True)
    xl = pd.ExcelFile(XLSX)
    rows = []
    for sheet in xl.sheet_names:
        date, cycles = SHEETS[sheet]
        d = pd.read_excel(xl, sheet, header=None)
        cols = [c for c in d.columns if d[c].notna().sum() > 100]
        if len(cols) != len(cycles):
            sys.exit(f"{sheet}: {len(cols)} data columns but {len(cycles)} cycles mapped")
        cands = [c for c in cycles if c is not None]
        for col, cyc in zip(cols, cycles):
            raw = d[col].to_numpy(float)
            sc = match_scores(raw, date, cands)
            best = max(sc, key=lambda k: -np.inf if np.isnan(sc[k]) else sc[k]) if sc else None
            row = dict(sheet=sheet, column=int(col), cycle=cyc, n_frames=int(np.isfinite(raw).sum()),
                       r_assigned=sc.get(cyc, np.nan), best_match=best,
                       r_best=sc.get(best, np.nan), **{f"r_c{k}": v for k, v in sc.items()})
            if cyc is None:
                row["status"] = "skipped: no cycle on site"
                rows.append(row); continue
            key = f"IP3R2_KD_{date}_cycle{cyc}"
            pos = clean_position(raw)
            spd = np.abs(np.diff(pos, prepend=np.nan))
            z = robust_z(spd)
            t = np.arange(len(raw)) / FPS
            pd.DataFrame(dict(time=np.round(t, 3), pos_px_raw=raw, pos_px_clean=pos,
                              speed_px_per_frame=spd, eye_speed=z)).to_csv(
                OUT / f"{key}_eyemotion_perframe.csv", index=False, float_format="%.5g")
            qc_plot(key, t, raw, pos, spd, z, OUT / "qc" / f"{key}.png")
            row.update(key=key, status="ok", frac_valid=float(np.isfinite(z).mean()),
                       median_speed_px_s=float(np.nanmedian(spd) * FPS))
            rows.append(row)
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "mapping.csv", index=False)
    pd.set_option("display.width", 220)
    print(m[["sheet", "column", "cycle", "n_frames", "r_assigned", "best_match", "r_best",
             "frac_valid", "median_speed_px_s", "status"]].round(2).to_string(index=False))
    print("->", OUT)


if __name__ == "__main__":
    main()
