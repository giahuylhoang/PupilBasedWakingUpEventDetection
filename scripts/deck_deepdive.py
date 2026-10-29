"""Deep-dive deck (2026-10-16): why GCaMP8m vessels dip at wake-up.

Figure-led, same style as scripts/deck_part1.py. Figures from
scripts/dilation_deepdive.py and scripts/dilation_split.py (run those first).

    venv/bin/python scripts/deck_deepdive.py
"""
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_2026_10_09 import BLUE, GREEN, GREY, header, picture, takeaway, text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DD = ROOT / "data" / "RESULTS_dilation_deepdive_2026-10-16" / "figures"
DS = ROOT / "data" / "RESULTS_dilation_split_2026-10-14" / "figures"
CY = ROOT / "data" / "RESULTS_cycle_deepdive_2026-10-16" / "figures"
OUT = ROOT / "data" / "presentation_materials" / "2026-10-16_meeting" / "Deepdive_dilation_2026_10_16_v3.pptx"
METHOD = ("D = √(vessel area). WT, final labels 2026-10-09; noisy cycles excluded (2026.03.03 c13, 2026.03.06 c13–16). "
          "Wake-up time = pupil onset. p values use recording day as the unit unless stated.")
LEVEL = "Vessel size relative to the median of its own 15-min cycle (no pre-wake baseline). "


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    new = lambda: prs.slides.add_slide(prs.slide_layouts[6])  # noqa: E731

    # title
    s = new()
    text(s, 0.8, 1.2, 11.7, 1.3, "Why GCaMP8m vessels dip at wake-up", size=36, color=GREEN, bold=True)
    text(s, 0.8, 2.3, 12.0, 0.5, "Deep dive   |   Gia-Huy Hoang   |   2026-10-16   |   Gordon Lab", size=17, color=GREY)
    text(s, 0.8, 3.2, 11.8, 3.2, [
        "Splitting wake-ups at +1% dilation: RSC 2023 has 42 dilating / 27 non-dilating NREM → wake events; "
        "GCaMP8m has 2 / 21 (NREM → wake) and 0 / 14 (REM → wake). Recording cycle number does not explain it.", "",
        "The main difference is where the vessel sits during sleep. GCaMP8m vessels are larger in NREM (and much "
        "larger in REM) than when awake, so waking returns them to a smaller awake size. RSC 2023 vessels are "
        "smaller in NREM, so the same return looks like a dilation.", "",
        "This explains part of the cohort difference, not all of it.",
    ], size=18)
    text(s, 0.8, 6.5, 11.8, 0.6, METHOD, size=12, color=GREY, italic=True)

    # 1 vessel by state
    s = new()
    header(s, "In sleep, GCaMP8m vessels are larger; 2023 vessels smaller",
           "Median vessel size in each state, per recording day. W = Awake, N = NREM, R = REM.", METHOD)
    picture(s, DS / "5_vessel_by_state.png", 0.6, 1.45, 8.6, 4.8)
    text(s, 9.5, 1.8, 3.5, 4.4, [
        ("NREM − Awake", dict(bold=True)), "RSC 2023: −1.9% (4 of 5 days below)", "GCaMP8m: +2.3% (3 of 3 days above)",
        "p = 0.032 (Welch, days)", "",
        ("REM", dict(bold=True)), "+14% and +21% on the two GCaMP8m days with REM", "",
        ("RSC 2023 days have no REM", dict(color=GREY, size=13)),
    ], size=15)
    takeaway(s, "The two cohorts start their wake-ups from opposite sides of the awake size.", y=6.35)

    # 2 levels around wake-up
    s = new()
    header(s, "At wake-up the vessel returns to its awake size",
           "Mean ± SEM across days, −120…+120 s around pupil onset. " + LEVEL, METHOD)
    picture(s, DD / "A_levels.png", 0.3, 1.5, 12.7, 4.5)
    text(s, 0.6, 5.95, 12.1, 0.5,
         "NREM → wake: RSC 2023 goes from about −1% to +1.5%; GCaMP8m goes from about 0 to −2%. "
         "REM → wake: GCaMP8m falls from +14% to 0.", size=14)
    takeaway(s, "The GCaMP8m 'dip' is the vessel leaving a sleep-dilated state, not a constriction below awake.", y=6.45)

    # 3 per cycle
    s = new()
    header(s, "Cycles with more sleep dilation have a bigger wake-up dip",
           "Each dot = one cycle with NREM → wake events. x: NREM − Awake vessel level in that cycle, "
           "excluding −30…+60 s around every wake-up. y: mean wake-up response (ΔD/D0, 0–15 s).", METHOD)
    picture(s, DD / "B_cycles.png", 0.9, 1.65, h=4.6)
    text(s, 7.6, 1.8, 5.4, 4.4, [
        ("All 22 cycles", dict(bold=True)), "ρ = −0.68, slope −0.43", "", 
        ("Within each cohort, same direction", dict(bold=True)),
        "RSC 2023: ρ = −0.51 (14 cycles, p = 0.06)", "GCaMP8m: ρ = −0.60 (8 cycles, p = 0.12)", "",
        "Both cohorts fall on one line: RSC cycles on the left, GCaMP8m on the right.", "",
        ("p values here treat cycles as independent", dict(color=GREY, size=13)),
    ], size=15)
    takeaway(s, "One relationship, two positions on it: the cohorts differ mainly in how dilated the vessel is during sleep.",
             y=6.35)

    # 4 per event
    s = new()
    header(s, "In GCaMP8m, the more dilated before waking, the bigger the drop",
           "Each dot = one NREM → wake event. x: vessel level in the 20 s before pupil onset, vs the cycle median. "
           "y: wake-up response (ΔD/D0, 0–15 s).", METHOD)
    picture(s, DD / "C_events_prelevel.png", 0.3, 1.5, w=9.5)
    text(s, 9.9, 1.7, 3.2, 4.4, [
        ("Mixed model", dict(bold=True)), "response ~ cohort + (1 | day), 92 events, 8 days", "",
        "Cohort gap: −4.0%, p < 0.001", "Adding pre-wake level: −3.6%", "",
        "Using the cycle's NREM − Awake level instead brings it to about −2.8%, but that fit is degenerate (no p)",
    ], size=14)
    takeaway(s, "Starting level predicts the drop event by event in GCaMP8m, but explains only part of the cohort gap.", y=6.35)

    # 5 measurement
    s = new()
    header(s, "Signal quality is similar; GCaMP8m vessels are about twice as large",
           "Per-cycle properties of the vessel trace. Small dots = cycles, large dots = day means; p on day means "
           "(RSC 5 days, GCaMP8m 3 days).", METHOD)
    picture(s, DD / "E_quality.png", 0.3, 1.5, 12.7, 4.1)
    text(s, 0.6, 5.65, 12.1, 0.7,
         "Equivalent diameter 108 vs 58 px. Noise, jumps and drift do not differ. "
         "Whisking after waking also does not differ (p = 0.71) and does not track the response.", size=14)
    takeaway(s, "Not a noisier signal. The size difference fits a different vessel type, but pixels also depend on zoom.",
             y=6.35)

    # 6 cycles 6-10 overlap
    s = new()
    header(s, "Later cycles don't explain it: GCaMP8m dips from cycle 6",
           "NREM → wake events in cycles 6–10 only, the range both cohorts were recorded in. Dots = events, bar = day mean.",
           METHOD)
    picture(s, CY / "2_overlap_cycles_6_10.png", 0.5, 1.5, h=4.7)
    text(s, 9.0, 1.8, 4.1, 4.4, [
        ("Same cycles, opposite sign", dict(bold=True)), "RSC 2023: +0.6 to +5.8% (4 of 4 days)",
        "GCaMP8m 2025.12.18: −1.3% (13 events)", "",
        "12.18 already dips at cycle 6 (−2.8%), its first cycle.", "",
        ("Only one GCaMP8m day has cycles below 11", dict(color=GREY, size=13)),
    ], size=15)
    takeaway(s, "The dip is there from the start of the GCaMP8m recording, not something that builds up in later cycles.",
             y=6.35)

    # 7 per-cycle trends
    s = new()
    header(s, "Across cycles, nothing trends in either cohort",
           "Per cycle, one line per day. ρ = within-day rank correlation with cycle number, cycles as units.", METHOD)
    picture(s, CY / "1_by_cycle.png", 0.4, 1.4, h=5.0)
    text(s, 7.1, 1.6, 6.0, 4.8, [
        ("Wake-up response vs cycle", dict(bold=True)), "RSC ρ = +0.12 (p = 0.70); GCaMP8m ρ = −0.09 (p = 0.87)", "",
        ("NREM − Awake level", dict(bold=True)), "GCaMP8m above 0 from cycle 6 on; no rise with cycle", "",
        ("Vessel size", dict(bold=True)), "Differs by day (80–107 vs 133–146 px), not by cycle", "",
        ("Sleep amount", dict(bold=True)), "% NREM and % Awake similar; REM only on GCaMP8m days", "",
        ("Late noisy cycles are excluded, so cycles > 14 can't be tested", dict(color=GREY, size=13)),
    ], size=14)
    takeaway(s, "What separates the cohorts is fixed per day (vessel, sleep dilation), not the cycle number.", y=6.5)

    # 8 summary / questions
    s = new()
    header(s, "Summary and open questions", None, METHOD)
    text(s, 0.8, 1.4, 11.8, 5.2, [
        ("What we know", dict(bold=True, color=GREEN)),
        "• At wake-up, the vessel returns to its awake size from wherever it was during sleep.",
        "• Not a later-cycle effect: GCaMP8m dips from cycle 6, and the response doesn't trend with cycle in either cohort.",
        "• GCaMP8m vessels are dilated in sleep, so they shrink; RSC 2023 vessels are not, so they grow.",
        "• This accounts for part of the cohort gap; the rest is unexplained. Cohort and year are confounded (5 vs 3 days).",
        "• In mouse somatosensory cortex, arterioles are larger in NREM (+9%) and REM (+33%) than awake rest, and blood "
        "volume falls on waking (Turner et al. 2020 eLife). On waking from REM or NREM microarousals they constrict back "
        "to awake size (Bojarskaite et al. 2023 Nat Commun). The GCaMP8m pattern matches; the 2023 RSC one does not.", "",
        ("What would settle it", dict(bold=True, color=GREEN)),
        "• Vessel type (pial artery vs penetrating arteriole), imaging depth and zoom for each day.",
        "• Raw red-channel images to check the area measure (dye changed 15 → 20 mg/mL).",
        "• Dates of any protocol changes between 2023 and 2025–26.",
        "• Add GCaMP8m 2026.03.07 once labelled.",
    ], size=17)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT, len(prs.slides), "slides")


if __name__ == "__main__":
    main()
