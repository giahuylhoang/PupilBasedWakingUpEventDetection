#!/usr/bin/env python3
"""Are the IP3R2 scramble 2025.06.01 pupil traces real pupil traces?

Florence flagged cycles 7/10/11/12 of 2025.06.01 as looking "a little bit
weird". Two things make them look unlike the cycles she has been labelling,
and this figure shows both:

  1. They are RAW pupil area (integer pixel counts, unsmoothed). Frame-to-frame
     noise is ~1.5% of the median, against ~0.02-0.05% for the 2024 sessions,
     which were smoothed before upload. Same signal, different processing.

  2. They contain long stretches (up to 44 s) where pupil area collapses to
     near zero. Those are not tracking failures: whisker activity falls 3-7x
     inside them while arteriole variability holds or rises, which is the
     signature of sleep with the eye closed.

Writes to a dated folder; never overwrites a previous run.

    ./venv/bin/python scripts/check_pupil_2025_06_01.py
"""

from __future__ import annotations

import io
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRACES = ROOT / "data" / "pupil_whisker_arteriole"
OUT = ROOT / "data" / "figures_pupil_check_2025.06.01_2026-09-25"

IN_QUESTION = [
    "IP3R2_scramble_2025.06.01_cycle7",
    "IP3R2_scramble_2025.06.01_cycle10",
    "IP3R2_scramble_2025.06.01_cycle11",
    "IP3R2_scramble_2025.06.01_cycle12",
]
REFERENCE = "IP3R2_scramble_2024.05.05_cycle13"   # a cycle she already labelled
FS = 50.0
LOW_FRAC = 0.2      # "collapsed" = below 20% of the cycle median


def load(path: pathlib.Path) -> pd.DataFrame:
    """Read like the server does: utf-8, falling back to latin-1 for the
    Mac-exported files whose headers carry a MacRoman non-breaking space."""
    raw = path.read_bytes()
    try:
        return pd.read_csv(io.BytesIO(raw), encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(raw), encoding="latin-1", engine="python")


def series(key: str, suffix: str):
    path = TRACES / f"{key}_{suffix}.csv"
    if not path.exists():
        return None, None
    d = load(path)
    t = pd.to_numeric(d[d.columns[0]], errors="coerce").to_numpy(float)
    y = pd.to_numeric(d[d.columns[-1]], errors="coerce").to_numpy(float)
    return t, y


def low_spans(t: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """Contiguous True runs as (t_start, t_end), for shading."""
    spans, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((t[start], t[i - 1]))
            start = None
    if start is not None:
        spans.append((t[start], t[-1]))
    return spans


def panel(ax_p, ax_w, key: str, title_extra: str = "") -> dict:
    tp, yp = series(key, "pupil")
    tw, yw = series(key, "whisker-gradient")
    med = np.nanmedian(yp)
    low = yp < LOW_FRAC * med
    spans = [s for s in low_spans(tp, low) if s[1] - s[0] >= 1.0]

    noise = 100 * np.nanmedian(np.abs(np.diff(yp))) / med

    ax_p.plot(tp, yp, lw=0.4, color="#1f77b4")
    for a, b in spans:
        for ax in (ax_p, ax_w):
            ax.axvspan(a, b, color="#d62728", alpha=0.16, lw=0)
    ax_p.axhline(med, color="0.4", lw=0.7, ls="--")
    ax_p.set_ylabel("pupil area\n(px²)", fontsize=8)
    ax_p.set_title(
        f"{key}   —   frame-to-frame noise {noise:.2f}% of median, "
        f"{len(spans)} collapse period(s) ≥1 s{title_extra}",
        fontsize=9, loc="left",
    )
    ax_p.tick_params(labelsize=7)

    if tw is not None:
        ax_w.plot(tw, yw, lw=0.3, color="#2ca02c")
        ax_w.set_ylabel("whisker\n(gradient)", fontsize=8)
    ax_w.set_xlabel("time (s)", fontsize=8)
    ax_w.tick_params(labelsize=7)

    # whisker activity inside vs outside the collapse periods
    ratio = np.nan
    if tw is not None and spans:
        idx = np.clip(np.searchsorted(tp, tw), 0, len(low) - 1)
        m = low[idx]
        f = lambda a: np.median(np.abs(a - np.median(a))) if a.size > 10 else np.nan
        inside, outside = yw[m & np.isfinite(yw)], yw[(~m) & np.isfinite(yw)]
        if inside.size and outside.size and f(outside):
            ratio = f(inside) / f(outside)
    return {
        "key": key, "median_px2": med, "noise_pct": noise,
        "n_spans": len(spans),
        "longest_span_s": max((b - a for a, b in spans), default=0.0),
        "collapsed_pct": 100 * float(low.mean()),
        "whisker_in_over_out": ratio,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = IN_QUESTION + [REFERENCE]
    fig, axes = plt.subplots(
        2 * len(keys), 1, figsize=(13, 2.5 * len(keys)),
        gridspec_kw={"height_ratios": [3, 1] * len(keys), "hspace": 0.55},
    )
    rows = []
    for i, key in enumerate(keys):
        extra = "   [REFERENCE: already labelled, smoothed]" if key == REFERENCE else ""
        rows.append(panel(axes[2 * i], axes[2 * i + 1], key, extra))
    fig.suptitle(
        "IP3R2 scramble 2025.06.01 pupil traces — red shading = pupil area below "
        "20% of median for ≥1 s\n"
        "Whisker quiet inside the shaded periods = asleep with the eye closed, "
        "not a tracking failure",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    png = OUT / "pupil_2025.06.01_vs_reference.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)

    summary = pd.DataFrame(rows)
    csv = OUT / "pupil_2025.06.01_summary.csv"
    summary.to_csv(csv, index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {png}\nwrote {csv}")


if __name__ == "__main__":
    main()
