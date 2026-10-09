"""Findings part 1 deck (2026-10-09): why WT wake-up arteriole responses differ.

Figure-led: one figure per slide, title = the finding, one takeaway line.
Figures from scripts/deck_part1_figures.py (run that first).
Style helpers shared with scripts/deck_2026_10_09.py.

    <python with python-pptx> scripts/deck_part1.py
"""
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_2026_10_09 import BLUE, GREEN, GREY, header, picture, takeaway, text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "data" / "RESULTS_deck_part1_2026-10-09" / "figures"
OUT = ROOT / "data" / "presentation_materials" / "2026-10-09_meeting" / "Findings_part1_2026_10_09.pptx"
METHOD = ("ΔD/D0 = (D − D0)/D0, D = √(vessel area), D0 = median of the 20 s before pupil onset; "
          "ΔF/F0 the same for Ca²⁺. WT, final labels 2026-10-09; p values use recording day as the unit.")


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    new = lambda: prs.slides.add_slide(prs.slide_layouts[6])  # noqa: E731

    # title
    s = new()
    text(s, 0.8, 1.2, 11.7, 1.3, "Why do WT wake-up arteriole responses differ?", size=36, color=GREEN, bold=True)
    text(s, 0.8, 2.3, 12.0, 0.5, "Part 1   |   Gia-Huy Hoang   |   2026-10-09   |   Gordon Lab", size=17, color=GREY)
    text(s, 0.8, 3.2, 11.8, 3.2, [
        "At wake-up, astrocyte Ca²⁺ rises on every WT day, but the arteriole does not respond the same way. "
        "The 2023 RSC vessels dilate when the mouse wakes from NREM (+2.3%, 5 of 5 days); the 2025–26 GCaMP8m "
        "vessels do not, and drift slightly down (−1.1%; p = 0.03).", "",
        "This is not explained by how strongly the mouse wakes up, and not by REM: waking from REM gives a large "
        "apparent constriction (8–20%) as the vessel leaves REM's dilation, but the GCaMP8m NREM → wake events "
        "contain no REM and still do not dilate.",
    ], size=18)
    text(s, 0.8, 6.5, 11.8, 0.6, METHOD, size=12, color=GREY, italic=True)

    # 1 heatmap
    s = new()
    header(s, "Ca²⁺ rises on every WT day; dilation only in 2023",
           "All 104 WT NREM → wake events. Top block RSC 2023, bottom block GCaMP8m 2025–26; each sorted by pupil rise.",
           METHOD)
    picture(s, FIG / "1_heatmap_wt_n2w.png", 0.3, 1.45, 9.9, 5.45)
    text(s, 10.3, 1.7, 2.8, 4.8, [
        ("Read across one row", dict(bold=True)), "= one wake-up", "",
        ("Pupil, Ca²⁺", dict(bold=True)), "rise in both blocks", "",
        ("Arteriole", dict(bold=True)), ("red (dilates) on top", dict(color="C62828")),
        ("blue (narrows) below", dict(color=BLUE)), "",
        ("Eye movement", dict(bold=True)), "top: pixel speed (2023 days)", "bottom: DLC (03.03 / 03.06)",
        ("different measures; display only, not used in stats", dict(color=GREY, size=11)),
    ], size=14)

    # 2 per day
    s = new()
    header(s, "REM → wake constricts; NREM → wake dilates only in 2023",
           "Each WT day: average after waking from NREM (orange) and from REM (purple). Grey titles = RSC 2023, green = GCaMP8m.",
           METHOD)
    picture(s, FIG / "2_per_day_n2w_r2w.png", 0.3, 1.5, 12.7, 3.6)
    text(s, 0.6, 5.2, 12.1, 1.0, [
        "REM → wake exists only on 2025.12.18 and 2026.03.03 (14 events). It falls 8–15%: the vessel was dilated in REM "
        "and returns to its awake level.",
        "NREM → wake: the five 2023 days go up after onset; the three GCaMP8m days stay flat or go down.",
    ], size=15)
    takeaway(s, "REM explains the large constrictions, but not why the GCaMP8m NREM → wake events do not dilate.", y=6.3)

    # 3 cohorts
    s = new()
    header(s, "Without REM, GCaMP8m vessels still don't dilate; Ca²⁺ is equal",
           "NREM → wake only. Mean ± SEM across recording days (RSC 2023: 5 days, 69 events; GCaMP8m: 3 days, 35 events).",
           METHOD)
    picture(s, FIG / "3_cohorts_n2w.png", 0.3, 1.5, 12.7, 4.5)
    takeaway(s, "Same Ca²⁺ signal, opposite vessel response: the difference is in the vessel or its measurement, "
                "not in how the mouse wakes.", y=6.2)

    # 4 arousal
    s = new()
    header(s, "Stronger wake-ups give more Ca²⁺; more dilation only in 2023",
           "NREM → wake events split into thirds of pupil rise within each day. Top: RSC 2023 (5 days). Bottom: GCaMP8m (3 days).",
           METHOD)
    picture(s, FIG / "4_arousal.png", 0.3, 1.45, 9.6, 4.8)
    text(s, 10.1, 1.7, 3.0, 4.8, [
        ("RSC 2023 (69 events)", dict(bold=True)), "Ca²⁺: ρ = +0.37, p = 0.002", "Arteriole: ρ = +0.26, p = 0.03", "",
        ("GCaMP8m (35 events)", dict(bold=True)), "Ca²⁺: ρ = +0.54, p = 0.001", "Arteriole: ρ = +0.02, p = 0.90", "",
        ("ρ = rank correlation of pupil rise vs 0–15 s mean, within day; events as units", dict(color=GREY, size=12)),
    ], size=14)
    takeaway(s, "Arousal scales Ca²⁺ in both cohorts, but dilation only in 2023: GCaMP8m stays flat even at high arousal.", y=6.35)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT, len(prs.slides), "slides")


if __name__ == "__main__":
    main()
