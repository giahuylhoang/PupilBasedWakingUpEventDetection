"""Build the 2026-10-09 meeting deck (same style as Update_2026_09_15.pptx).

One question per slide: title = the answer, subtitle = what was measured,
figure or table, one-line takeaway. Numbers are copied from:
  data/RESULTS_wake_arousal_explore_2026-10-02/WT/...   (old labels, 2026-10-01)
  data/RESULTS_wake_arousal_explore_2026-10-02/WT/rem_check_2026-10-08/
  data/RESULTS_groups_relabel_2026-10-08/               (new labels, 2026-10-08)
  data/RESULTS_figures_prelim_2026-10-08_relabel/tables/
Only figures without a name in their footer are used.

Needs python-pptx only:  <python with python-pptx==1.0.2> scripts/deck_2026_10_09.py
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
OUT = D / "presentation_materials" / "2026-10-09_meeting" / "Update_2026_10_09.pptx"
WT = D / "RESULTS_wake_arousal_explore_2026-10-02" / "WT"
GR = D / "RESULTS_groups_relabel_2026-10-09"   # final labels

GREEN, GREY, BLUE, RED, DARK = "2E7D32", "757575", "0D47A1", "C62828", "212121"


def rgb(h):
    return RGBColor.from_string(h)


def text(slide, x, y, w, h, lines, size=16, color=DARK, bold=False, italic=False, align=None):
    """lines: str or list of str / (str, dict(size,color,bold)). '- ' prefix = bullet indent."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, ln in enumerate([lines] if isinstance(lines, str) else lines):
        opt = {}
        if isinstance(ln, tuple):
            ln, opt = ln
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if ln.startswith("- "):
            ln = "•  " + ln[2:]
            p.level = 1
        r = p.add_run()
        r.text = ln
        f = r.font
        f.size = Pt(opt.get("size", size))
        f.bold = opt.get("bold", bold)
        f.italic = italic
        f.color.rgb = rgb(opt.get("color", color))
        p.space_after = Pt(opt.get("space", 6))
        if align:
            p.alignment = align
    return tb


def header(slide, title, sub=None, source=None):
    text(slide, 0.5, 0.3, 12.4, 0.8, title, size=30, color=GREEN, bold=True)
    if sub:
        text(slide, 0.5, 1.0, 12.4, 0.5, sub, size=15, color=GREY)
    if source:
        text(slide, 0.5, 7.0, 12.4, 0.3, source, size=12, color=GREY, italic=True)


def picture(slide, path, x, y, w=None, h=None):
    """Fit inside w x h (inches), keeping aspect ratio, centred in the box."""
    iw, ih = Image.open(path).size
    ar = iw / ih
    if w and h:
        if w / h > ar:
            pw, ph = h * ar, h
        else:
            pw, ph = w, w / ar
        x, y = x + (w - pw) / 2, y + (h - ph) / 2
    elif w:
        pw, ph = w, w / ar
    else:
        pw, ph = h * ar, h
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(pw), Inches(ph))


def table(slide, x, y, w, rows, col_w=None, size=13, row_h=0.36, highlight=None):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                               Inches(row_h * len(rows))).table
    if col_w:
        for j, cw in enumerate(col_w):
            t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            p = c.text_frame.paragraphs[0]
            p.text = ""
            f = p.add_run().font
            p.runs[0].text = str(v)
            f.size = Pt(size)
            if i == 0:
                c.fill.solid(); c.fill.fore_color.rgb = rgb(BLUE)
                f.color.rgb = rgb("FFFFFF"); f.bold = True
            else:
                c.fill.solid()
                c.fill.fore_color.rgb = rgb("FFF3E0" if highlight and i in highlight else
                                            ("F5F5F5" if i % 2 else "FFFFFF"))
                f.color.rgb = rgb(DARK)
            if j > 0:
                p.alignment = PP_ALIGN.CENTER
    return t


def takeaway(slide, s, y=6.35):
    text(slide, 0.6, y, 12.1, 0.55, s, size=16, color=BLUE, bold=True)


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    new = lambda: prs.slides.add_slide(blank)  # noqa: E731

    # 1 title
    s = new()
    text(s, 0.8, 2.0, 11.7, 1.3, "Why are wake-up arteriole responses mixed?", size=38, color=GREEN, bold=True)
    text(s, 0.5, 3.3, 12.4, 0.5, "Gia-Huy Hoang   |   2026-10-09   |   Gordon Lab", size=18, color=GREY)
    text(s, 0.8, 4.2, 11.5, 3.0, [
        ("Three answers", dict(bold=True, size=20)),
        "1.  Not arousal strength: pupil rise does not predict the arteriole response",
        "2.  Some was REM: wake-ups from REM constrict; two newer days still constrict without them",
        "3.  The rest follows recording date: wake-up dilation fades from 2023 to 2026, across all groups",
        ("With full manual labels, KD = scramble at wake-up", dict(bold=True, color=BLUE)),
    ], size=18)

    # 2 the question
    s = new()
    header(s, "The question: Ca²⁺ rises almost every time, the arteriole does not",
           "WT NREM→wake events (130, 8 days, labels of 2026-10-01). Each row = one event, sorted by pupil rise.",
           "data/RESULTS_wake_arousal_explore_2026-10-02/WT/figures/A_heatmap_by_pupil.png")
    picture(s, WT / "figures" / "A_heatmap_by_pupil.png", 0.4, 1.5, 8.6, 4.75)
    text(s, 9.2, 1.7, 3.8, 4.5, [
        ("Ca²⁺ goes up", dict(bold=True)), "119 / 130 events", "",
        ("Arteriole goes up", dict(bold=True)), "83 / 130 events", "47 go down", "",
        ("How we measure it", dict(bold=True)),
        "Time 0 = pupil dilation onset",
        "Response = 0–15 s after, minus −5…0 s before",
    ], size=15)
    takeaway(s, "Grant's question: are these different kinds of wake-up, or something else?")

    # 3 arousal
    s = new()
    header(s, "1. Bigger arousal does not mean bigger dilation",
           "Events split into thirds by pupil rise. Mean ± SEM across events.",
           "data/RESULTS_wake_arousal_explore_2026-10-02/WT/figures/D_tertile_averages.png")
    picture(s, WT / "figures" / "D_tertile_averages.png", 0.4, 1.6, 12.5, 3.5)
    text(s, 0.6, 5.2, 12.1, 1.1, [
        "Ca²⁺ is larger after bigger pupil rises; the arteriole is the same in all three thirds "
        "(within-day correlation with pupil rise ρ = −0.02).",
        "Eye movement: only 26 events on 2 days, so it cannot be separated from day effects yet. Whisking left out (unreliable).",
    ], size=15)
    takeaway(s, "The arousal-strength idea does not explain the mixed responses.")

    # 4 split by day
    s = new()
    header(s, "The split is by recording day",
           "Median arteriole level change at wake-up per day (labels of 2026-10-01): mean +10…+40 s minus −30…−10 s, % of cycle median",
           "data/RESULTS_wake_arousal_explore_2026-10-02/WT/w2n_vs_n2w/per_day.csv")
    table(s, 0.6, 1.7, 7.4, [
        ["Day", "Events", "Wake-up change", "Events going down"],
        ["RSC 2023.05.05", "18", "+1.4%", "33%"],
        ["RSC 2023.05.07", "16", "+3.9%", "31%"],
        ["RSC 2023.05.17", "18", "0.0%", "50%"],
        ["RSC 2023.06.02", "10", "+2.7%", "20%"],
        ["RSC 2023.07.12", "7", "+3.3%", "29%"],
        ["GCaMP8m 2025.12.18", "35", "−0.9%", "54%"],
        ["GCaMP8m 2026.03.03", "12", "−2.7%", "75%"],
        ["GCaMP8m 2026.03.06", "14", "−0.6%", "71%"],
    ], col_w=[2.6, 1.2, 1.9, 1.7], highlight={6, 7, 8})
    text(s, 8.4, 1.8, 4.5, 4.4, [
        ("2023 days dilate, the newer days do not", dict(bold=True)), "",
        "Not a movement artifact: no sudden jumps, Ca²⁺ normal in the same events", "",
        "Vessel size differs (more Texas Red from 2025: 15 → 20 mg/mL), but every trace is % of its own median, "
        "so size itself does not change these numbers",
    ], size=15)
    takeaway(s, "So we checked two explanations: REM, and recording date.")

    # 5 relabel
    s = new()
    header(s, "Labels were updated on 2026-10-08, final pass 2026-10-09",
           "All KD and scramble cycles now have manual labels; GCaMP8m cycles re-checked for REM and micro-arousals",
           "data/labels_bucket_2026-10-08_1456/, labels_bucket_2026-10-09_1022/  (old labels kept unchanged)")
    table(s, 0.6, 1.7, 7.6, [
        ["", "Before (10-01)", "Now (10-08)"],
        ["KD cycles with manual labels", "7 of 19 (rest predicted)", "19 of 19"],
        ["Scramble cycles", "16 of 16", "16 of 16 (no edits)"],
        ["KD bins changed", "—", "346 (2023.08.25/26 mostly)"],
        ["GCaMP8m bins changed", "—", "343, incl. 75 NREM → REM"],
        ["GCaMP8m final pass (10-09)", "—", "+219 bins, mostly NREM → Awake"],
        ["2023 RSC and GRAB-NE", "—", "unchanged"],
    ], col_w=[3.2, 2.2, 2.2], size=13)
    text(s, 8.6, 1.8, 4.3, 4.4, [
        ("Checks", dict(bold=True)),
        "Pupil larger in Awake than NREM in every cycle",
        "Weakest on KD 2023.08.26 (degraded pupil video); c6 only 37% labelled",
        "Our classifier vs the new KD labels: κ = 0.63, so earlier predicted labels were reasonable",
    ], size=15)
    takeaway(s, "Everything from here on uses the final labels of 2026-10-09.")

    # 6 REM
    s = new()
    header(s, "2. REM → wake explains some constrictions, not all",
           "Wake-up change per day with old labels, new labels, and new labels excluding any REM in the 3 min before",
           "data/RESULTS_wake_arousal_explore_2026-10-02/WT/rem_check_2026-10-09_1022/")
    picture(s, WT / "rem_check_2026-10-09_1022" / "per_day_old_vs_new.png", 0.4, 1.5, 8.4, 3.6)
    text(s, 9.0, 1.6, 4.0, 4.6, [
        ("REM → wake events constrict", dict(bold=True)),
        "- Old wake-ups now labelled REM → wake had constricted 4–9%", "",
        ("Final labels, no REM in the 3 min before", dict(bold=True)),
        "- 2025.12.18: −2.2% (15 events)",
        "- 2026.03.03: −2.5% (6 events)",
        "- 2026.03.06: +0.2% (13 events)",
    ], size=14)
    text(s, 0.6, 5.3, 8.2, 0.9, "Caution: REM was re-labelled with the arteriole visible. New REM bins: pupil small (as NREM), "
         "whisker higher (twitches), arteriole +6.8%. Partly independent, partly circular.", size=13, color=GREY)
    takeaway(s, "REM → wake must be kept apart, but 2025.12.18 and 2026.03.03 still constrict without it.")

    # 7 chronology
    s = new()
    header(s, "3. Wake-up dilation fades with recording date, across all groups",
           "Each dot = one recording day, wake-up response minus random NREM moments of the same day (21 days, 4 groups)",
           "data/RESULTS_groups_relabel_2026-10-09/figures/chronology_n2w.png, chronology_tests.csv")
    picture(s, GR / "figures" / "chronology_n2w.png", 0.4, 1.5, 12.5, 4.0)
    text(s, 0.6, 5.55, 12.1, 0.8, [
        "Rank correlation with date (AUC): ρ = −0.52 (p = 0.015); without KD ρ = −0.61 (p = 0.016); level change ρ = −0.46 (p = 0.035).   "
        "Before Mar 2024: 9 / 10 days dilate.  After: 4 / 11 (cut-off chosen after looking).",
    ], size=15)
    takeaway(s, "Matches the protocol changes (habituation, rig, dye), but those all changed together, so the cause is open.")

    # 8 groups
    s = new()
    header(s, "No group shows a clear wake-up dilation once all days are included",
           "Day-level response minus random NREM moments; p = one-sample t vs 0",
           "data/RESULTS_groups_relabel_2026-10-09/response_vs_zero.csv")
    picture(s, GR / "figures" / "groups_summary.png", 0.4, 1.5, 12.5, 3.9)
    table(s, 0.6, 5.45, 12.1, [
        ["Wake-up", "WT (8 days)", "Scramble (4)", "KD (6)", "GRAB-NE (3)"],
        ["Arteriole dilates (AUC), days positive", "6 / 8, p = 0.13", "2 / 4, p = 0.58", "4 / 6, p = 0.64", "1 / 3, p = 0.95"],
        ["Ca²⁺ rises, days positive", "8 / 8, p = 0.004", "4 / 4, p = 0.17", "5 / 6, p = 0.06", "—"],
    ], col_w=[4.1, 2.0, 2.0, 2.0, 2.0], size=12, row_h=0.3)

    # 9 KD vs SCR
    s = new()
    header(s, "KD vs scramble: no difference at wake-up",
           "Day-level values, Welch t-test (unit = recording day). New labels.",
           "data/RESULTS_groups_relabel_2026-10-09/kd_vs_scr.csv;  RESULTS_figures_prelim_2026-10-08_relabel/tables/")
    table(s, 0.6, 1.7, 12.1, [
        ["", "Scramble (4 days)", "KD (6 days)", "p", "Before (10-01 labels)"],
        ["Wake-up arteriole AUC", "+0.09", "+0.04", "0.77", "n.s."],
        ["Wake-up Ca²⁺ AUC", "1.70", "0.95", "0.50", "KD 0.52×, p = 0.058"],
        ["Micro-arousal arteriole AUC", "+0.43 (20 events)", "−0.11 (4 events, 3 days)", "0.019", "p = 0.013"],
        ["% NREM", "45%", "35%", "0.085", "equal"],
        ["Micro-arousals / h", "5.2", "0.8", "0.06", "KD higher (predicted labels)"],
        ["% REM", "1.0% (2 of 4 days)", "0% (all days)", "—", "—"],
        ["Ultra-slow vessel power (no labels)", "reference", "0.40×", "0.015 (LMM)", "same"],
    ], col_w=[3.6, 2.2, 2.6, 1.5, 2.2], size=13, highlight={3})
    text(s, 0.6, 5.0, 12.1, 1.2, [
        "Micro-arousal result rests on 4 KD events, and WT micro-arousals do not dilate either: too thin to report.",
        "The robust KD difference is still the ultra-slow vessel oscillation, which does not depend on labels.",
    ], size=15)
    takeaway(s, "With full manual labels, KD and scramble respond the same at wake-up; KD may sleep a little less.")

    # 10 open questions
    s = new()
    header(s, "Questions for discussion")
    text(s, 0.6, 1.3, 12.1, 5.6, [
        ("Is everything retrosplenial cortex?", dict(bold=True)),
        "- Barrel cortex constricts at waking (Tim; published Fig 2). File names say RSC; needs confirming.",
        ("What changed over time?", dict(bold=True)),
        "- Habituation protocol, rig sound insulation, Texas Red 15 → 20 mg/mL, more REM. Which can we date?",
        ("Exclude or report the constricting days?", dict(bold=True)),
        "- 2025.12.18 and 2026.03.03 still constrict after REM is excluded, but both fit the date trend.",
        ("How is vessel size measured?", dict(bold=True)),
        "- If it counts bright pixels, brightness and red blood cells could mimic constriction. "
        "Raw red-channel images for 03.03 and one 2023 day would let us measure width directly.",
        ("2026.03.07", dict(bold=True)),
        "- Not on the site: no dimer re-run, and cycle numbers differ (7–14 vs 12–16).",
    ], size=16)

    # 11 next steps
    s = new()
    header(s, "Next steps")
    text(s, 0.6, 1.3, 12.1, 5.6, [
        "1.  Measure vessel width from raw images on 03.03 and a 2023 day (needs images)",
        "2.  Add dates to each protocol change, then test the date trend against them",
        "3.  Use GRAB-NE days to test whether noradrenaline goes with constriction at waking",
        "4.  Add 2026.03.07 once its version and cycle numbers are confirmed",
        "5.  Redo the figure set (Figs 3, 6, 7, 8) on the new labels, with REM → wake kept separate",
        "",
        ("Preliminary: unit = recording day, small n, no correction across all questions.", dict(color=GREY, size=15)),
    ], size=18)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT, len(prs.slides), "slides")


if __name__ == "__main__":
    main()
