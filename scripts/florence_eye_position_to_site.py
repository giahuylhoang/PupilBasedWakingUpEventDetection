"""Turn Florence's "pupil moving speed" sheets (2026-10-23) into site eye-motion files.

Best guess (checked on NE 2026.03.10 c14-16 against our DLC tracking): each column is
the VERTICAL POSITION of the pupil centre in pixels, 50 Hz, not a speed (r = 0.62-0.80
with DLC centre-y; never reaches 0). So, mirroring scripts/pupil_eye_motion.py:
  position -> interpolate interior gaps -> 10 Hz 4th-order Butterworth low-pass (= cy)
  eye_speed = |diff(cy)| in px / frame   (1-D, vertical only; DLC speed is 2-D)
  diameter  = our existing pupil trace for that cycle (pupil_whisker_arteriole), z-scored
  cx, min_lh = NaN (not provided); include = position present
Writes <eye key>_eyemotion_perframe.csv and _5s_features.csv, same columns as the DLC files.

Sheet quirks handled:
  * "pupil moving speed scramble": "2024.06.01" is taken as 2025.06.01 (our scramble day
    with cycles 7, 10, 11, 12; there is no 2024.06.01 recording).
  * "pupil moving speed for Gcmap8m" is RSC 2023.07.12; header labels sit in columns B-D
    but data are in E-G with no header -> E, F, G taken as cycles 6, 7, 8.
  * Scramble 2024.05.05: her cycles 11-14 are our cycles 10-13 (see CYCLE_SHIFT).
  * RSC 2023.07.12 column order E, F, G = cycles 6, 7, 8 confirmed the same way.
  * NE 2026.03.10 is skipped (DLC eye motion already on the site).
  * Positions shorter than the cycle (2026.05.10 c12, 643 s) are NaN after they end.

Usage:  venv/bin/python -B scripts/florence_eye_position_to_site.py
Output: data/upload_staging_2026-10-23/pupil_eye_motion/  (+ _summary.csv)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pupil_eye_motion as PEM  # noqa: E402

ROOT = HERE.parent
XLSX = ROOT / "data" / "florence_data" / "NE_2026-10-23" / "NE pupil moving speed 2.xlsx"
TRACE_DIR = ROOT / "data" / "pupil_whisker_arteriole"
STAGE_TRACES = ROOT / "data" / "upload_staging_2026-10-23" / "pupil_whisker_arteriole"
OUT = ROOT / "data" / "upload_staging_2026-10-23" / "pupil_eye_motion"
SKIP_DAYS = {"2026.03.10"}
# Florence (2026-10-23): the "2024.06.01" header is the 2025.06.01 scramble day (2024.06.01 is a
# KD day). Blink timing agrees: her c7 / c10 match scramble 2025.06.01 c7 / c10 (0.42 / 0.36).
DATE_FIX = {"2024.06.01": "2025.06.01"}
# Her scramble 2024.05.05 columns say cycles 11-14; we have cycles 10-13. Matching blink /
# artefact timing (5-s max |d position| vs max |d pupil diameter|, Spearman) puts her
# c12, c13, c14 on our c11 (0.25), c12 (0.35), c13 (0.31): her cycle N = our cycle N-1.
CYCLE_SHIFT = {"2024.05.05": -1}


def columns(sheet: str, df: pd.DataFrame):
    """Yield (trace key, eye key, position array) for one sheet."""
    if sheet.startswith("pupil moving speed for Gcmap8m"):
        for col, cyc in ((4, 6), (5, 7), (6, 8)):
            k = f"Extra_RestrosplenialCortex_2023.07.12_cycle{cyc}"
            yield k, k.removeprefix("Extra_"), pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        return
    scr = "scramble" in sheet
    date = None
    for col in df.columns:
        h = str(df.iloc[0, col]).strip()
        if re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", h):
            date = DATE_FIX.get(h, h)
            continue
        m = re.search(r"c\w*le\s*(\d+)", h)
        if not m or date is None or date in SKIP_DAYS:
            continue
        cyc = int(m.group(1)) + CYCLE_SHIFT.get(date, 0)
        k = f"IP3R2_scramble_{date}_cycle{cyc}" if scr else f"Extra_NE_{date}_cycle{cyc}"
        yield k, k.removeprefix("Extra_"), pd.to_numeric(df[col].iloc[1:], errors="coerce").to_numpy(float)


def trace_path(key):
    for d in (TRACE_DIR, STAGE_TRACES):
        p = d / f"{key}_pupil.csv"
        if p.exists():
            return p
    return None


def build(key, pos):
    p = trace_path(key)
    if p is None:
        return None, "no pupil trace for this key"
    diam = pd.read_csv(p, encoding_errors="replace").iloc[:, 1].pipe(pd.to_numeric, errors="coerce").to_numpy(float)
    diam[diam <= 0] = np.nan
    n = len(diam)
    pos = pos[:n]
    last = np.flatnonzero(np.isfinite(pos))
    if len(last) < 100:
        return None, "position column empty"
    end = last[-1] + 1                                   # keep NaN after the column ends
    y = np.full(n, np.nan)
    seg = pos[:end]
    good = np.isfinite(seg)
    y[:end] = PEM._lowpass(PEM._interp_masked(seg, good))
    speed = np.full(n, np.nan)
    speed[1:end] = np.abs(np.diff(y[:end]))              # px / frame, vertical only
    speed[0] = 0.0
    d = PEM._interp_masked(diam, np.isfinite(diam))
    dz = (d - np.nanmean(d)) / (np.nanstd(d) + 1e-9)
    c = {"diameter": d, "diameter_z": dz, "cx": np.full(n, np.nan), "cy": y, "eye_speed": speed}
    pf = pd.DataFrame({"time": np.arange(n) / PEM.FS, **c, "min_lh": np.nan,
                       "include": np.r_[good, np.zeros(n - end, bool)]})
    with np.errstate(all="ignore"):
        bins = PEM.bin_features(c)
    nb = int(PEM.FS * PEM.BIN_SEC)
    empty = [b for b in range(len(bins)) if not np.isfinite(speed[b * nb:(b + 1) * nb]).any()]
    bins.loc[empty, ["motion_var", "cy_var", "cy_maxdisp"]] = np.nan
    # motion_sum must stay finite: the labeling server passes it straight into JSON, and a
    # NaN there makes the browser's JSON.parse fail, so the cycle never loads (seen on
    # 2026.05.10 c12 and 2025.09.11 c12, 2026-10-29). Bins with no position get 0.
    bins.loc[empty, "motion_sum"] = 0.0
    return (pf, bins, end / PEM.FS, float(good.mean())), None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sheets = pd.read_excel(XLSX, sheet_name=None, header=None)
    rows = []
    for name, df in sheets.items():
        if "moving speed" not in name:
            continue
        for key, ekey, pos in columns(name, df):
            res, err = build(key, pos)
            if err:
                rows.append(dict(sheet=name.strip(), key=key, status=err))
                continue
            pf, bins, cover_s, frac = res
            pf.to_csv(OUT / f"{ekey}_eyemotion_perframe.csv", index=False)
            bins.to_csv(OUT / f"{ekey}_eyemotion_5s_features.csv", index=False)
            rows.append(dict(sheet=name.strip(), key=key, eye_key=ekey, status="ok", covered_s=round(cover_s, 1),
                             position_present=round(frac, 4), speed_median=float(np.nanmedian(pf.eye_speed)),
                             bins=len(bins)))
    S = pd.DataFrame(rows)
    S.to_csv(OUT.parent / "eye_motion_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
