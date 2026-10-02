"""Shared definitions for the preliminary figure set (Grant's 8-figure plan).

Every figure script imports from here so that states, events, normalisation,
spectra and statistics mean the same thing on every panel. If a definition
changes, it changes once, here.

Data
----
* Traces: data/pupil_whisker_arteriole/<key>_<suffix>.csv  (time, value)
    pupil / whisker-gradient   ~50 Hz
    arteriole-diameter (AREA)  ~1.955 Hz  -> diameter = sqrt(area)
    calcium / ne               ~1.955 Hz
* Manual state labels: data/labels_snapshot_2026-10-01/<key>_labels.csv,
  5-s bins, labels Awake / NREM / REM / Unclear (snapshot of the labeling
  bucket taken 2026-10-01; never edited in place).
* Eye motion (where DLC exists): data/pupil_eye_motion/<eyekey>_eyemotion_perframe.csv

Cohorts (groups)
----------------
  WT  = Extra_GCaMP8m + Extra_RestrosplenialCortex   (retrosplenial cortex)
  KD  = IP3R2_KD
  SCR = IP3R2_scramble
  NE  = Extra_NE   (GRAB-NE; no calcium channel)

Biological unit
---------------
The data carry no animal ID, so the unit is the RECORDING DAY ("mouse" below
= cohort + date). Consecutive-date sessions (e.g. KD 2023.08.25/26) may be one
animal, which would shrink n further. Every figure states n as days.

Events (from manual labels; 5-s resolution, Unclear breaks a bout)
-------------------------------------------------------------------
  MA   micro-arousal: Awake bout of 1-3 bins (5-15 s) with NREM >= 2 bins
       (>=10 s) immediately before AND after  (Luthi & Nedergaard 2025).
  N2W  NREM-to-wake: NREM bout >= 6 bins (30 s) followed by Awake >= 4 bins
       (>= 20 s, i.e. longer than an MA).
  W2N  wake-to-NREM (semi-control): Awake >= 6 bins then NREM >= 6 bins.
  REM  onset of any REM bout (descriptive only).
  t0 is the label boundary. Label resolution is 5 s, so latencies finer than
  that are not interpretable.

Signals after normalisation
---------------------------
  art   arteriole dD/D0, D = sqrt(area), D0 = per-cycle median
  ca    astrocyte Ca2+ dF/F0, F0 = per-cycle median
  ne    GRAB-NE dF/F0, same convention
  pupil within-cycle robust z (median / 1.4826*MAD), despiked with a 0.5-s
        running median (removes 1-2 frame blink/tracking drops), at 10 Hz.
        Pupil is NOT on a common scale across sessions (session medians span
        24 to 11,772), so only within-cycle z is used.
  whisk within-cycle robust z of the gradient-sum, at 10 Hz.

Spectra (same convention as scripts/pub_arteriole_psd.py and the vasomotion tool)
-----------------------------------------------------------------------------------
  whole record: 0.005 Hz 4th-order Butterworth high-pass -> linear detrend ->
                multitaper Welch, nperseg 256 (~131 s), 50% overlap, NW 3, K 5.
  state windows: 60-s windows stepped 10 s; a window counts for a state when
                >= 90% of its 5-s bins carry that label. Single-window
                multitaper periodogram (NW 2.5, K 4) on a common padded grid.
                60 s resolves the vasomotion band but NOT the ultra-slow band.
  Bands: ULTRA_SLOW 0.005-0.04 Hz, VASOMOTION 0.04-0.15 Hz (Grant's plan).
"""
from __future__ import annotations

import io
import math
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats
from scipy.ndimage import median_filter
from scipy.signal import detrend as sp_detrend
from scipy.signal import periodogram
from scipy.signal.windows import dpss

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vasomotion_app.preprocess import highpass_filter  # noqa: E402
from vasomotion_app.psd_methods import PSDParams, compute_psd  # noqa: E402

TRACE_DIR = ROOT / "data" / "pupil_whisker_arteriole"
# Which manual label set every analysis reads. Chosen once per run via the
# PRELIM_LABEL_SET environment variable so a version can never mix label sets:
#   original   Florence's labels as saved (snapshot 2026-10-01)   -> v2 (locked)
#   corrected  + pattern-2 corrections (scripts/propose_label_corrections.py) -> v3
LABEL_SET = os.environ.get("PRELIM_LABEL_SET", "original")
_LABEL_SETS = {
    "original": dict(labels="labels_snapshot_2026-10-01", pred="labels_predicted_2026-10-01",
                     ml="RESULTS_state_ml_2026-10-01", out="RESULTS_figures_prelim_2026-10-01_v2",
                     stamp="v2"),
    "corrected": dict(labels="labels_corrected_2026-10-01", pred="labels_predicted_2026-10-01_v3",
                      ml="RESULTS_state_ml_2026-10-01_v3", out="RESULTS_figures_prelim_2026-10-01_v3",
                      stamp="v3, pattern-2 label corrections NOT YET REVIEWED"),
}
if LABEL_SET not in _LABEL_SETS:
    raise ValueError(f"PRELIM_LABEL_SET must be one of {sorted(_LABEL_SETS)}, got {LABEL_SET!r}")
_LS = _LABEL_SETS[LABEL_SET]
LABEL_DIR = ROOT / "data" / _LS["labels"]
LABEL_PRED_DIR = ROOT / "data" / _LS["pred"]
ML_DIR = ROOT / "data" / _LS["ml"]
EYE_DIR = ROOT / "data" / "pupil_eye_motion"
# v2 (2026-10-01): event-triggered responses use refined events (response_events).
# v1 output, built on raw label-boundary events, is kept in RESULTS_figures_prelim_2026-10-01/.
# PRELIM_OUT_SUFFIX writes a layout-only refresh of a locked version to a new folder
# (e.g. "_tallETA" -> RESULTS_figures_prelim_2026-10-01_v2.1_tallETA); numbers must not change.
OUT = ROOT / "data" / (_LS["out"] + os.environ.get("PRELIM_OUT_SUFFIX", ""))
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"

# Which labels the figures use. "manual+ml": manual everywhere, ML fills the
# unlabelled IP3R2 KD bins. Every KD panel is also re-run with "manual" as a
# sensitivity check (see set_label_mode).
LABEL_MODE = "manual+ml"
# Days where the ML model failed leave-one-day-out validation (kappa 0.07 on
# IP3R2_KD_2023.08.26 -- its pupil file is the flat, pre-averaged 2023 export).
# Predictions are never used on these days; only manual labels.
ML_EXCLUDED_DAYS = frozenset({"IP3R2_KD_2023.08.26"})

BIN = 5.0
FS_BEH = 10.0                     # pupil / whisker / eye resampled to this
STATES = ("Awake", "NREM", "REM")
MAIN_STATES = ("Awake", "NREM")

COHORT_GROUP = {
    "Extra_GCaMP8m": "WT",
    "Extra_RestrosplenialCortex": "WT",
    "IP3R2_KD": "KD",
    "IP3R2_scramble": "SCR",
    "Extra_NE": "NE",
}
GROUP_LABEL = {"WT": "WT (RSC)", "KD": "IP3R2 KD", "SCR": "Scramble", "NE": "GRAB-NE"}

STATE_COL = {"Awake": "#1565C0", "NREM": "#C62828", "REM": "#F9A825", "Unclear": "#BDBDBD"}
GROUP_COL = {"WT": "#424242", "SCR": "#1E88E5", "KD": "#E53935", "NE": "#8E24AA"}
SIG_COL = {"pupil": "#5D4037", "whisk": "#2E7D32", "art": "#C62828",
           "ca": "#00897B", "ne": "#8E24AA", "eye": "#F57C00"}
SIG_LABEL = {"pupil": "pupil (z)", "whisk": "whisker (z)", "eye": "eye speed (z)",
             "art": "arteriole ΔD/D₀", "ca": "astro Ca²⁺ ΔF/F₀", "ne": "GRAB-NE ΔF/F₀"}

ULTRA_SLOW = (0.005, 0.04)
VASOMOTION = (0.04, 0.15)
BANDS = {"ultra_slow": ULTRA_SLOW, "vasomotion": VASOMOTION}
F_MAX = 0.5

WT_PSD = PSDParams(method="multitaper_welch", nperseg=256, noverlap=128,
                   nw=3.0, k=5, zero_pad=1, scaling="density")
HP_CUTOFF, HP_ORDER = 0.005, 4

STATE_WIN_S, STATE_STEP_S, STATE_PURITY = 60.0, 10.0, 0.90
STATE_NW, STATE_K = 2.5, 4

EVENT_TYPES = ("N2W", "MA", "W2N", "REM")
EVENT_LABEL = {"N2W": "NREM→wake", "MA": "micro-arousal", "W2N": "wake→NREM",
               "REM": "REM-like onset"}
EV_PRE, EV_POST = 30.0, 60.0         # epoch window (s)
BASE_WIN = (-5.0, 0.0)               # Grant: baseline -5 s
RESP_WIN = (0.0, 15.0)               # Grant: response 0..+15 s

KEY_RE = re.compile(r"^(?P<cohort>.*?)_(?P<date>\d{4}\.\d{2}\.\d{2})_cycle_?(?P<cycle>\d+)$")


# --------------------------------------------------------------------------- #
# keys, cohorts, mice
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KeyInfo:
    key: str
    cohort: str
    date: str
    cycle: int
    group: str
    mouse: str          # recording day: the biological unit we can defend


def parse_key(key: str) -> KeyInfo:
    m = KEY_RE.match(key)
    if not m:
        raise ValueError(f"unparseable key {key!r}")
    cohort = m["cohort"]
    group = COHORT_GROUP.get(cohort, cohort)
    return KeyInfo(key, cohort, m["date"], int(m["cycle"]), group,
                   f"{cohort}_{m['date']}")


def all_trace_keys() -> list[str]:
    return sorted(p.name[: -len("_pupil.csv")] for p in TRACE_DIR.glob("*_pupil.csv"))


def labeled_keys(mode: str | None = None) -> list[str]:
    mode = mode or LABEL_MODE
    keys = {p.name[: -len("_labels.csv")] for p in LABEL_DIR.glob("*_labels.csv")}
    if mode != "manual":
        for p in LABEL_PRED_DIR.glob("*_labels.csv"):
            k = p.name[: -len("_labels.csv")]
            try:
                if parse_key(k).mouse not in ML_EXCLUDED_DAYS:
                    keys.add(k)
            except ValueError:
                pass
    return sorted(keys)


def set_label_mode(mode: str) -> None:
    """Switch every subsequent load between 'manual' and 'manual+ml'."""
    global LABEL_MODE
    if mode not in ("manual", "manual+ml"):
        raise ValueError(mode)
    LABEL_MODE = mode
    load_labels.cache_clear()
    load_cycle.cache_clear()


def keys_for(groups, labeled_only: bool = True) -> list[str]:
    groups = {groups} if isinstance(groups, str) else set(groups)
    pool = labeled_keys() if labeled_only else all_trace_keys()
    out = []
    for k in pool:
        try:
            info = parse_key(k)
        except ValueError:
            continue
        if info.group in groups and (TRACE_DIR / f"{k}_pupil.csv").exists():
            out.append(k)
    return out


# --------------------------------------------------------------------------- #
# reading + normalisation
# --------------------------------------------------------------------------- #
def read_trace(key: str, suffix: str):
    """(t, y) as float arrays, or (None, None) if missing.

    Encoding fallback mirrors the labeling server: some exports carry a
    MacRoman non-breaking space in the header. Single-column files (no time)
    get an evenly spaced 0..900 s axis, as the labeling site does.
    """
    p = TRACE_DIR / f"{key}_{suffix}.csv"
    if not p.exists():
        return None, None
    raw = p.read_bytes()
    try:
        d = pd.read_csv(io.BytesIO(raw), encoding="utf-8")
    except (UnicodeDecodeError, pd.errors.ParserError):
        d = pd.read_csv(io.BytesIO(raw), encoding="latin-1", engine="python")
    if d.shape[1] == 1:
        y = pd.to_numeric(d.iloc[:, 0], errors="coerce").to_numpy(float)
        t = np.linspace(0.0, 900.0, len(y))
    else:
        t = pd.to_numeric(d.iloc[:, 0], errors="coerce").to_numpy(float)
        y = pd.to_numeric(d.iloc[:, -1], errors="coerce").to_numpy(float)
    ok = np.isfinite(t)
    return t[ok], y[ok]


def _robust_z(y: np.ndarray) -> np.ndarray:
    med = np.nanmedian(y)
    mad = 1.4826 * np.nanmedian(np.abs(y - med))
    if not np.isfinite(mad) or mad <= 0:
        sd = np.nanstd(y)
        mad = sd if sd > 0 else 1.0
    return (y - med) / mad


def _resample(t, y, fs, t_end=None):
    """Linear resample onto a regular grid; gaps (NaN) stay NaN-free by interp
    over finite samples only, then re-masked where the source gap exceeds 1 s."""
    fin = np.isfinite(y)
    if fin.sum() < 2:
        return None, None
    t_end = float(t[-1]) if t_end is None else t_end
    grid = np.arange(0.0, t_end + 1e-9, 1.0 / fs)
    out = np.interp(grid, t[fin], y[fin])
    # mask long source gaps
    tf = t[fin]
    gaps = np.flatnonzero(np.diff(tf) > 1.0)
    for g in gaps:
        out[(grid > tf[g]) & (grid < tf[g + 1])] = np.nan
    return grid, out


def _ratio_norm(y):
    x0 = np.nanmedian(y)
    if not np.isfinite(x0) or abs(x0) < 1e-12:
        return np.full_like(y, np.nan)
    return (y - x0) / x0


@dataclass
class Cycle:
    info: KeyInfo
    sig: dict           # name -> (t, y) normalised
    fs: dict            # name -> nominal fs
    labels: np.ndarray  # per 5-s bin, object (str or None)

    @property
    def key(self):
        return self.info.key

    @property
    def duration(self):
        return len(self.labels) * BIN


@lru_cache(maxsize=None)
def load_cycle(key: str) -> Cycle:
    info = parse_key(key)
    sig, fs = {}, {}

    t, y = read_trace(key, "pupil")
    if t is not None and np.isfinite(y).sum() > 100:
        fs_raw = 1.0 / np.nanmedian(np.diff(t))
        k = max(1, int(round(0.5 * fs_raw)) | 1)            # 0.5-s odd window
        yd = median_filter(np.where(np.isfinite(y), y, np.nanmedian(y)), size=k, mode="nearest")
        yd[~np.isfinite(y)] = np.nan
        g, r = _resample(t, _robust_z(yd), FS_BEH)
        if g is not None:
            sig["pupil"], fs["pupil"] = (g, r), FS_BEH

    t, y = read_trace(key, "whisker-gradient")
    if t is not None and np.isfinite(y).sum() > 100:
        g, r = _resample(t, _robust_z(y), FS_BEH)
        if g is not None:
            sig["whisk"], fs["whisk"] = (g, r), FS_BEH

    t, y = read_trace(key, "arteriole-diameter")
    if t is not None and np.isfinite(y).sum() > 50:
        d = np.sqrt(np.clip(y, 0, None))
        d[y <= 0] = np.nan
        sig["art"], fs["art"] = (t, _ratio_norm(d)), 1.0 / np.nanmedian(np.diff(t))

    for suf, name in (("calcium", "ca"), ("ne", "ne")):
        t, y = read_trace(key, suf)
        if t is not None and np.isfinite(y).sum() > 50:
            sig[name], fs[name] = (t, _ratio_norm(y)), 1.0 / np.nanmedian(np.diff(t))

    eye = _eye_motion(key)
    if eye is not None:
        sig["eye"], fs["eye"] = eye, FS_BEH

    return Cycle(info, sig, fs, load_labels(key))


def _eye_motion(key):
    for cand in (key, re.sub(r"^Extra_", "", key)):
        p = EYE_DIR / f"{cand}_eyemotion_perframe.csv"
        if p.exists():
            d = pd.read_csv(p)
            if "include" in d:
                d.loc[~d["include"].astype(bool), "eye_speed"] = np.nan
            t = d["time"].to_numpy(float)
            y = d["eye_speed"].to_numpy(float)
            g, r = _resample(t, _robust_z(y), FS_BEH)
            return (g, r) if g is not None else None
    return None


@lru_cache(maxsize=None)
def load_labels(key: str, mode: str | None = None) -> np.ndarray:
    """Per-bin labels; bins with no label are None. Unclear stays 'Unclear'.

    mode "manual"     Florence's labels only.
    mode "manual+ml"  manual where it exists; ML predictions (pupil+whisker
                      model, scripts/predict_states_ml.py) fill unlabelled KD
                      bins, except on days where the model failed validation
                      (ML_EXCLUDED_DAYS). Manual labels always win.
    Default comes from LABEL_MODE so a whole figure run can be switched once.
    """
    mode = mode or LABEL_MODE
    manual = _read_label_file(LABEL_DIR / f"{key}_labels.csv")
    if mode == "manual":
        return manual
    pred_path = LABEL_PRED_DIR / f"{key}_labels.csv"
    try:
        day = parse_key(key).mouse
    except ValueError:
        return manual
    if not pred_path.exists() or day in ML_EXCLUDED_DAYS:
        return manual
    pred = _read_label_file(pred_path)
    if manual.size:
        n = max(len(pred), len(manual))
        pred = np.concatenate([pred, np.full(n - len(pred), None, dtype=object)])
        for i, v in enumerate(manual):
            if v is not None:
                pred[i] = v
    return pred


def label_source(key: str, mode: str | None = None) -> str:
    """'manual', 'manual+ml' (partly predicted), 'ml', or 'none'."""
    mode = mode or LABEL_MODE
    man = (LABEL_DIR / f"{key}_labels.csv").exists()
    if mode == "manual":
        return "manual" if man else "none"
    pred = (LABEL_PRED_DIR / f"{key}_labels.csv").exists() and \
        parse_key(key).mouse not in ML_EXCLUDED_DAYS
    if pred and man:
        return "manual+ml"
    return "ml" if pred else ("manual" if man else "none")


def _read_label_file(p: Path) -> np.ndarray:
    if not p.exists():
        return np.array([], dtype=object)
    d = pd.read_csv(p)
    n = int(d["bin"].max()) + 1
    out = np.full(n, None, dtype=object)
    for b, lab in zip(d["bin"].astype(int), d["label"].astype(str)):
        out[b] = lab
    return out


def bin_series(t, y, n_bins, how="mean"):
    """Reduce a trace to one value per 5-s bin."""
    out = np.full(n_bins, np.nan)
    idx = np.floor(t / BIN).astype(int)
    for i in range(n_bins):
        v = y[idx == i]
        v = v[np.isfinite(v)]
        if v.size:
            out[i] = v.mean() if how == "mean" else v.std()
    return out


# --------------------------------------------------------------------------- #
# bouts and events
# --------------------------------------------------------------------------- #
def bouts(labels) -> pd.DataFrame:
    """Contiguous runs. Columns: state, start_bin, n_bins, t_start, dur_s."""
    rows, i, lab = [], 0, list(labels)
    while i < len(lab):
        j = i
        while j + 1 < len(lab) and lab[j + 1] == lab[i]:
            j += 1
        rows.append(dict(state=lab[i], start_bin=i, n_bins=j - i + 1,
                         t_start=i * BIN, dur_s=(j - i + 1) * BIN))
        i = j + 1
    return pd.DataFrame(rows, columns=["state", "start_bin", "n_bins", "t_start", "dur_s"])


def events(labels) -> pd.DataFrame:
    """Event table from labels. Columns: type, t0, pre_dur_s, post_dur_s."""
    b = bouts(labels).to_dict("records")
    rows = []
    for i in range(1, len(b) - 1):
        p, c, n = b[i - 1], b[i], b[i + 1]
        if (c["state"] == "Awake" and 1 <= c["n_bins"] <= 3
                and p["state"] == "NREM" and p["n_bins"] >= 2
                and n["state"] == "NREM" and n["n_bins"] >= 2):
            rows.append(dict(type="MA", t0=c["t_start"], pre_dur_s=p["dur_s"],
                             post_dur_s=c["dur_s"]))
    for i in range(len(b) - 1):
        c, n = b[i], b[i + 1]
        if c["state"] == "NREM" and c["n_bins"] >= 6 and n["state"] == "Awake" and n["n_bins"] >= 4:
            rows.append(dict(type="N2W", t0=n["t_start"], pre_dur_s=c["dur_s"],
                             post_dur_s=n["dur_s"]))
        if c["state"] == "Awake" and c["n_bins"] >= 6 and n["state"] == "NREM" and n["n_bins"] >= 6:
            rows.append(dict(type="W2N", t0=n["t_start"], pre_dur_s=c["dur_s"],
                             post_dur_s=n["dur_s"]))
        if n["state"] == "REM":
            rows.append(dict(type="REM", t0=n["t_start"], pre_dur_s=c["dur_s"],
                             post_dur_s=n["dur_s"]))
    return pd.DataFrame(rows, columns=["type", "t0", "pre_dur_s", "post_dur_s"])


def event_table(keys) -> pd.DataFrame:
    """Events across cycles, with key / mouse / group attached."""
    frames = []
    for k in keys:
        cyc = load_cycle(k)
        e = events(cyc.labels)
        if e.empty:
            continue
        e = e.assign(key=k, mouse=cyc.info.mouse, group=cyc.info.group)
        frames.append(e)
    if not frames:
        return pd.DataFrame(columns=["type", "t0", "pre_dur_s", "post_dur_s", "key", "mouse", "group"])
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# event refinements (default for response analyses since v2, via response_events)
# --------------------------------------------------------------------------- #
ANCHOR_SEARCH = (-10.0, 10.0)    # search window around the label boundary (s)
ANCHOR_MIN_RISE = 0.5            # pupil must rise >= 0.5 z, else keep the label t0


def pupil_onset(t, y, t_label):
    """Pupil dilation onset near a label boundary, or None if there is no clear rise.

    The label boundary is only known to the nearest 5-s bin. This refines it from
    pupil alone (never from arteriole or Ca, so it cannot select for the outcome):
      baseline b = median pupil z over [t_label-15, t_label-5]
      peak     p = max of 1-s-smoothed pupil over [t_label-5, t_label+10]
      half-max crossing = first time pupil >= b + 0.5 (p - b), searched from
                          t_label-10
      onset    = last time before that crossing when pupil <= b + 0.1 (p - b),
                 not earlier than t_label-10
    """
    fin = np.isfinite(y)
    if fin.sum() < 20:
        return None
    s = pd.Series(np.where(fin, y, np.nan)).rolling(int(round(FS_BEH)), center=True,
                                                     min_periods=1).mean().to_numpy()
    lo, hi = t_label + ANCHOR_SEARCH[0], t_label + ANCHOR_SEARCH[1]
    bm = (t >= t_label - 15) & (t < t_label - 5)
    pm = (t >= t_label - 5) & (t <= t_label + 10)
    if not np.isfinite(s[bm]).any() or not np.isfinite(s[pm]).any():
        return None
    b = np.nanmedian(s[bm])
    p = np.nanmax(s[pm])
    rise = p - b
    if not np.isfinite(rise) or rise < ANCHOR_MIN_RISE:
        return None
    w = np.flatnonzero((t >= lo) & (t <= hi))
    half = [i for i in w if np.isfinite(s[i]) and s[i] >= b + 0.5 * rise]
    if not half:
        return None
    ih = half[0]
    below = [i for i in w if i <= ih and np.isfinite(s[i]) and s[i] <= b + 0.1 * rise]
    return float(t[below[-1]]) if below else float(t[w[0]])


def anchor_to_pupil(ev: pd.DataFrame) -> pd.DataFrame:
    """Copy of `ev` with t0 moved to pupil dilation onset for N2W and MA.

    Adds t0_label (original), shift_s, anchored (bool). Events with no clear
    pupil rise keep the label t0 (anchored=False). W2N and REM are not moved:
    their defining pupil change is a constriction or none.
    """
    out = ev.copy()
    out["t0_label"] = out["t0"]
    out["anchored"] = False
    for i, r in out.iterrows():
        if r["type"] not in ("N2W", "MA"):
            continue
        cyc = load_cycle(r["key"])
        if "pupil" not in cyc.sig:
            continue
        t, y = cyc.sig["pupil"]
        on = pupil_onset(t, y, r["t0"])
        if on is not None:
            out.at[i, "t0"] = on
            out.at[i, "anchored"] = True
    out["shift_s"] = out["t0"] - out["t0_label"]
    return out


OVERLAP_WIN = (-15.0, 20.0)      # another event onset inside this window -> drop


def response_events(keys) -> pd.DataFrame:
    """Events for EVENT-TRIGGERED RESPONSE analyses (ETAs, peaks, heat maps, KD vs
    scramble responses): t0 moved to pupil dilation onset, then events with a
    neighbour within OVERLAP_WIN dropped. Adopted after the 2026-10-01 sensitivity
    analysis (scripts/event_refinement_sensitivity.py) showed WT responses unchanged
    and the label boundary lagging pupil onset by ~1-2 s.

    Do NOT use for event COUNTS or RATES (bouts/h, MA/h): dropping overlapping
    events would undercount them. Use event_table() there.
    """
    return drop_overlapping(anchor_to_pupil(event_table(keys))).reset_index(drop=True)


def drop_overlapping(ev: pd.DataFrame, win=OVERLAP_WIN) -> pd.DataFrame:
    """Keep only events with no OTHER event (any type) starting within `win` of
    their t0 in the same cycle. The window covers the -5..0 s baseline and the
    0..15 s response plus margin, so a kept event's metrics cannot include a
    neighbour's response. Uses the label t0 when present, so the result does
    not depend on whether pupil anchoring was applied first."""
    tcol = "t0_label" if "t0_label" in ev else "t0"
    keep = np.ones(len(ev), bool)
    for _, idx in ev.groupby("key").indices.items():
        tt = ev[tcol].to_numpy()[idx]
        for j, i in enumerate(idx):
            d = np.delete(tt, j) - tt[j]
            if np.any((d >= win[0]) & (d <= win[1])):
                keep[i] = False
    return ev[keep].copy()


REL_GRID = {}


def rel_grid(fs):
    fs = round(fs, 3)
    if fs not in REL_GRID:
        REL_GRID[fs] = np.arange(-EV_PRE, EV_POST + 1e-9, 1.0 / fs)
    return REL_GRID[fs]


def epoch(t, y, t0, fs_out, baseline=True):
    """One event epoch on a common relative grid; NaN where out of range."""
    rg = rel_grid(fs_out)
    tt = t0 + rg
    fin = np.isfinite(y)
    if fin.sum() < 2:
        return np.full_like(rg, np.nan)
    tf = t[fin]
    v = np.interp(tt, tf, y[fin], left=np.nan, right=np.nan)
    # mask samples whose nearest finite source sample is > 1.5 s away (gaps)
    j = np.clip(np.searchsorted(tf, tt), 1, len(tf) - 1)
    nearest = np.minimum(np.abs(tf[j] - tt), np.abs(tf[j - 1] - tt))
    v[nearest > 1.5] = np.nan
    if baseline:
        m = (rg >= BASE_WIN[0]) & (rg < BASE_WIN[1])
        v = v - np.nanmean(v[m]) if np.isfinite(v[m]).any() else v * np.nan
    return v


EPOCH_FS = {"pupil": 2.0, "whisk": 2.0, "eye": 2.0, "art": 1.955, "ca": 1.955, "ne": 1.955}


def epochs_for(ev: pd.DataFrame, signal: str, baseline=True):
    """Matrix (n_events, n_time) for `signal`; rows with no data are NaN."""
    fs = EPOCH_FS[signal]
    rg = rel_grid(fs)
    M = np.full((len(ev), len(rg)), np.nan)
    for i, r in enumerate(ev.itertuples()):
        cyc = load_cycle(r.key)
        if signal in cyc.sig:
            t, y = cyc.sig[signal]
            M[i] = epoch(t, y, r.t0, fs, baseline)
    return rg, M


def response_metrics(rg, M):
    """Per-event peak (max in RESP_WIN), AUC and time-to-peak."""
    m = (rg >= RESP_WIN[0]) & (rg <= RESP_WIN[1])
    seg = M[:, m]
    peak = np.nanmax(seg, axis=1)
    tpk = np.array([rg[m][np.nanargmax(r)] if np.isfinite(r).any() else np.nan for r in seg])
    dt = np.median(np.diff(rg))
    auc = np.nansum(seg, axis=1) * dt
    auc[~np.isfinite(seg).any(axis=1)] = np.nan
    peak[~np.isfinite(seg).any(axis=1)] = np.nan
    return peak, auc, tpk


def responder(rg, M, k_sd=2.0):
    """Event 'responds' if its response-window peak exceeds k x SD of its own
    pre-event window (-30..-5 s). Used for response probability."""
    pre = (rg >= -EV_PRE) & (rg < BASE_WIN[0])
    sd = np.nanstd(M[:, pre], axis=1)
    peak, _, _ = response_metrics(rg, M)
    out = peak > k_sd * sd
    out = out.astype(float)
    out[~np.isfinite(peak) | ~np.isfinite(sd)] = np.nan
    return out


# --------------------------------------------------------------------------- #
# spectra
# --------------------------------------------------------------------------- #
def _fill(y):
    y = np.asarray(y, float).copy()
    fin = np.isfinite(y)
    if fin.sum() < 2:
        return None
    idx = np.arange(len(y))
    y[~fin] = np.interp(idx[~fin], idx[fin], y[fin])
    return y


def psd_whole(y, fs):
    """Whole-record spectrum, pub_arteriole_psd convention."""
    y = _fill(y)
    if y is None or len(y) < WT_PSD.nperseg:
        return None, None
    y = highpass_filter(y, fs, HP_CUTOFF, HP_ORDER)
    y = sp_detrend(y, type="linear")
    f, P = compute_psd(y, fs, WT_PSD)
    keep = f <= F_MAX
    return f[keep], P[keep]


STATE_F = np.linspace(0, F_MAX, 257)


def _mt_periodogram(seg, fs):
    seg = sp_detrend(seg, type="linear")
    n = len(seg)
    tapers = dpss(n, NW=STATE_NW, Kmax=STATE_K)
    nfft = int(2 ** math.ceil(math.log2(max(n, 2)) + 2))
    acc = None
    for tp in tapers:
        f, P = periodogram(seg, fs=fs, window=tp, nfft=nfft, scaling="density",
                           detrend=False, return_onesided=True)
        acc = P if acc is None else acc + P
    P = acc / len(tapers)
    return STATE_F, np.interp(STATE_F, f, P)


def state_windows(labels, state, win_s=STATE_WIN_S, step_s=STATE_STEP_S, purity=STATE_PURITY):
    """Start times of windows dominated (>= purity) by `state`."""
    nb = int(round(win_s / BIN))
    step = int(round(step_s / BIN))
    lab = np.asarray(labels, dtype=object)
    out = []
    for s in range(0, len(lab) - nb + 1, step):
        w = lab[s:s + nb]
        if np.mean(w == state) >= purity:
            out.append(s * BIN)
    return out


def psd_state(t, y, fs, labels, state):
    """Mean multitaper spectrum over state-dominant windows; (f, P, n_windows)."""
    starts = state_windows(labels, state)
    Ps = []
    nwin = int(round(STATE_WIN_S * fs))
    for s in starts:
        m = (t >= s) & (t < s + STATE_WIN_S)
        seg = y[m]
        if len(seg) < 0.9 * nwin or np.isfinite(seg).mean() < 0.9:
            continue
        seg = _fill(seg)
        Ps.append(_mt_periodogram(seg, fs)[1])
    if not Ps:
        return STATE_F, None, 0
    return STATE_F, np.mean(Ps, axis=0), len(Ps)


def band_power(f, P, band):
    if f is None or P is None:
        return np.nan
    m = (f >= band[0]) & (f < band[1])
    if m.sum() < 2:
        return np.nan
    return float(np.trapezoid(P[m], f[m]))


# --------------------------------------------------------------------------- #
# aggregation and statistics
# --------------------------------------------------------------------------- #
def mouse_mean(df, value, by=("group", "mouse"), extra=()):
    """Collapse to one value per mouse (and any extra keys)."""
    keys = list(by) + list(extra)
    return df.groupby(keys, as_index=False)[value].mean()


def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    q = p[ok]
    order = np.argsort(q)
    ranked = q[order] * ok.sum() / (np.arange(ok.sum()) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty_like(q)
    res[order] = np.minimum(ranked, 1.0)
    out[ok] = res
    return out


def paired_test(a, b):
    """Paired Wilcoxon + paired t on matched arrays (e.g. Awake vs NREM per mouse)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = b - a
    res = dict(n=int(ok.sum()), mean_a=np.nan, mean_b=np.nan, mean_diff=np.nan,
               dz=np.nan, t_p=np.nan, wilcoxon_p=np.nan, wilcoxon_min_p=np.nan)
    if len(d) == 0:
        return res
    res.update(mean_a=a.mean(), mean_b=b.mean(), mean_diff=d.mean())
    scale = max(np.abs(d).max(), 1e-12)
    if len(d) >= 2 and d.std(ddof=1) > 1e-9 * scale:
        res["dz"] = d.mean() / d.std(ddof=1)
        res["t_p"] = sstats.ttest_rel(b, a).pvalue
    if len(d) >= 1 and np.any(d != 0):
        try:
            res["wilcoxon_p"] = sstats.wilcoxon(d).pvalue
        except ValueError:
            pass
    res["wilcoxon_min_p"] = 2.0 / 2 ** int((d != 0).sum()) if (d != 0).any() else np.nan
    return res


def group_test(a, b):
    """Unpaired: Welch t + Mann-Whitney + Hedges g. a = control, b = treatment."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    res = dict(n_a=len(a), n_b=len(b), mean_a=np.nan, mean_b=np.nan,
               hedges_g=np.nan, welch_p=np.nan, mw_p=np.nan)
    if len(a):
        res["mean_a"] = a.mean()
    if len(b):
        res["mean_b"] = b.mean()
    if len(a) >= 2 and len(b) >= 2:
        sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                       / (len(a) + len(b) - 2))
        if sp > 0:
            g = (b.mean() - a.mean()) / sp
            g *= 1 - 3 / (4 * (len(a) + len(b)) - 9)
            res["hedges_g"] = g
            res["welch_p"] = sstats.ttest_ind(b, a, equal_var=False).pvalue
    if len(a) >= 1 and len(b) >= 1:
        res["mw_p"] = sstats.mannwhitneyu(b, a, alternative="two-sided").pvalue
    return res


LMM_DEGENERATE_RATIO = 0.01


def lmm(df, value, fixed, group="mouse"):
    """Linear mixed model value ~ fixed + (1|mouse). Returns the fixed-effect
    table, or None when it cannot be fitted (too few mice)."""
    import warnings

    import statsmodels.formula.api as smf
    d = df.dropna(subset=[value]).copy()
    if d[group].nunique() < 3 or len(d) < 8:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm(f"{value} ~ {fixed}", d, groups=d[group]).fit(reml=True)
        t = m.summary().tables[1]
        t = t.reset_index().rename(columns={"index": "term"})
        t["n_obs"], t["n_mice"] = len(d), d[group].nunique()
        # If the between-mouse variance collapses (<1% of residual), the model has
        # silently become event-level OLS -- events treated as independent, the
        # pseudoreplication in the May ANCOVA -- and its p-values are too small.
        ratio = float(m.cov_re.iloc[0, 0]) / float(m.scale) if m.scale else float("nan")
        t["mouse_var_over_resid"] = ratio
        t["degenerate_fit"] = bool(np.isfinite(ratio) and ratio < LMM_DEGENERATE_RATIO)
        return t
    except Exception as exc:  # noqa: BLE001 - report, don't crash a figure
        return pd.DataFrame([dict(term="FIT_FAILED", note=str(exc)[:120])])


# --------------------------------------------------------------------------- #
# output helpers
# --------------------------------------------------------------------------- #
def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    ensure_dirs()
    p = FIG_DIR / f"{name}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    return p


def save_table(df, name):
    ensure_dirs()
    p = TAB_DIR / f"{name}.csv"
    df.to_csv(p, index=False)
    return p


def shade_states(ax, labels, t_offset=0.0, alpha=0.18, states=STATES):
    """Shade a time axis by manual state."""
    for b in bouts(labels).itertuples():
        if b.state in states:
            ax.axvspan(b.t_start - t_offset, b.t_start + b.dur_s - t_offset,
                       color=STATE_COL[b.state], alpha=alpha, lw=0)


def n_note(df, group_col="group", mouse_col="mouse", event_col=None):
    """'WT: 8 days / 30 cyc' style caption fragment."""
    parts = []
    for g, s in df.groupby(group_col):
        frag = f"{GROUP_LABEL.get(g, g)}: {s[mouse_col].nunique()} days"
        if "key" in s:
            frag += f", {s['key'].nunique()} cyc"
        if event_col:
            frag += f", {len(s)} ev"
        parts.append(frag)
    return " · ".join(parts)


PRELIM_STAMP = (f"PRELIMINARY {_LS['stamp']} — labels 2026-10-01; event responses use pupil-anchored, "
                "non-overlapping events; unit = recording day (no animal IDs); not for manuscript")
