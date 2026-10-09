"""Findings part 2 deck (2026-10-09): IP3R2 scramble vs KD at NREM -> wake.

Figure-led: one figure per slide, title = the finding, one takeaway line.
Figures and numbers from scripts/deck_part2_figures.py (run that first); every
number on the slides is read from its CSV output.

    <python with python-pptx> scripts/deck_part2.py
"""
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.util import Inches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_2026_10_09 import GREEN, GREY, header, picture, takeaway, text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "RESULTS_deck_part2_2026-10-09"
FIG = RES / "figures"
OUT = ROOT / "data" / "presentation_materials" / "2026-10-09_meeting" / "Findings_part2_2026_10_09.pptx"
METHOD = ("NREM → wake from the final labels (2026-10-09), t0 = pupil onset, every event kept. ΔD/D0 = (D − D0)/D0, "
          "D = √(vessel area), D0 = median of the 20 s before pupil onset; ΔF/F0 the same. p values: recording day = unit.")


METHOD_MAX = ("NREM → wake from the final labels (2026-10-09), every event kept. Ca²⁺ and arteriole t0 = best onset "
              "in −5…+20 s of pupil onset (max integrated (X − X0)/X0 over 10 s, X0 = 5 s mean); then D0 / F0 = median "
              "of the 20 s before that onset. LME: events, recording day as random effect.")


def p_txt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def main():
    T = pd.read_csv(RES / "tests.csv").set_index("metric")
    D = pd.read_csv(RES / "day_values.csv")
    R = pd.read_csv(RES / "coupling_arousal.csv")
    ev = pd.read_csv(RES / "events.csv")
    n = ev.groupby("group").agg(e=("t0", "size"), d=("day", "nunique"))
    rho = lambda g, x, y: R[(R.group == g) & (R.x == x) & (R.y == y)].iloc[0]  # noqa: E731
    scr_wo = D[(D.group == "SCR") & (D.day != "2025.06.01")]
    out_day = D[(D.group == "SCR") & (D.day == "2025.06.01")].iloc[0]

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    new = lambda: prs.slides.add_slide(prs.slide_layouts[6])  # noqa: E731

    # title
    s = new()
    text(s, 0.8, 1.2, 11.7, 1.3, "Does IP3R2 knockdown change wake-up?", size=36, color=GREEN, bold=True)
    text(s, 0.8, 2.3, 12.0, 0.5, "Part 2   |   Gia-Huy Hoang   |   2026-10-09   |   Gordon Lab", size=17, color=GREY)
    text(s, 0.8, 3.2, 11.8, 3.2, [
        f"Scramble ({n.loc['SCR', 'd']} days, {n.loc['SCR', 'e']} wake-ups) and KD ({n.loc['KD', 'd']} days, "
        f"{n.loc['KD', 'e']} wake-ups) wake up with a similar pupil rise, and their astrocyte Ca²⁺ and arteriole "
        f"responses do not differ (Ca²⁺ {p_txt(T.loc['ca_mean', 'welch_p'])}, arteriole "
        f"{p_txt(T.loc['art_mean', 'welch_p'])}).", "",
        "Scramble's higher average Ca²⁺ comes from one recording day. In both groups, a bigger Ca²⁺ response goes "
        "with a bigger dilation, and a stronger wake-up gives more of both.",
    ], size=18)
    text(s, 0.8, 6.5, 11.8, 0.6, METHOD, size=12, color=GREY, italic=True)

    # 1 PSTH
    s = new()
    header(s, "KD and scramble wake up alike, and respond alike",
           "Group mean ± SEM across recording days. Green = scramble, red = KD. Dashed line = pupil onset.", METHOD)
    picture(s, FIG / "1_psth.png", 0.3, 1.6, 12.7, 3.7)
    text(s, 0.6, 5.4, 12.1, 0.8, [
        f"Pupil rise: scramble {T.loc['pupil_rise', 'scr_mean']:.1f} z vs KD {T.loc['pupil_rise', 'kd_mean']:.1f} z "
        f"({p_txt(T.loc['pupil_rise', 'welch_p'])}).   Mean 0–15 s: Ca²⁺ {T.loc['ca_mean', 'scr_mean']:+.1f}% vs "
        f"{T.loc['ca_mean', 'kd_mean']:+.1f}% ({p_txt(T.loc['ca_mean', 'welch_p'])});   arteriole "
        f"{T.loc['art_mean', 'scr_mean']:+.1f}% vs {T.loc['art_mean', 'kd_mean']:+.1f}% "
        f"({p_txt(T.loc['art_mean', 'welch_p'])}).",
    ], size=15)
    takeaway(s, "Same arousal, same Ca²⁺ and arteriole response at wake-up.", y=6.25)

    # 2 heatmaps
    s = new()
    header(s, "Ca²⁺ rises on every day; vessels vary within both groups",
           "Every NREM → wake event, one row each. Grouped by recording day; within a day, sorted by pupil rise.",
           METHOD)
    picture(s, FIG / "2_heatmaps.png", 0.3, 1.45, 9.9, 5.45)
    text(s, 10.3, 1.7, 2.8, 4.8, [
        ("Read across one row", dict(bold=True)), "= one wake-up", "",
        ("Top", dict(bold=True)), "scramble, 4 days", "",
        ("Bottom", dict(bold=True)), "KD, 6 days", "",
        ("White", dict(bold=True)), "no Ca²⁺ / vessel recording in that cycle (KD 2023.08.26, 2025.10.25)",
    ], size=14)

    # 3 summary
    s = new()
    header(s, "No difference in Ca²⁺ or arteriole peak or AUC",
           "0–15 s after pupil onset. Bar = mean ± SEM of day means; large dots = days; open circles = events.",
           METHOD)
    picture(s, FIG / "3_summary.png", 0.3, 1.5, 12.7, 4.2)
    text(s, 0.6, 5.75, 12.1, 0.5,
         f"Ca²⁺ peak {T.loc['ca_peak', 'scr_mean']:.0f}% vs {T.loc['ca_peak', 'kd_mean']:.0f}% "
         f"({p_txt(T.loc['ca_peak', 'welch_p'])}).   Arteriole peak {T.loc['art_peak', 'scr_mean']:.1f}% vs "
         f"{T.loc['art_peak', 'kd_mean']:.1f}% ({p_txt(T.loc['art_peak', 'welch_p'])}).   "
         f"Welch t on day means; Mann–Whitney agrees.", size=14)
    takeaway(s, "With 4 vs 6 days, the measured differences are small compared with day-to-day spread.", y=6.3)

    # 3b max-response onset: PSTH
    T2 = pd.read_csv(RES / "tests_maxonset.csv").set_index("metric")
    e2 = pd.read_csv(RES / "events_maxonset.csv")
    sh = e2.groupby("group")[["ca_shift", "art_shift"]].median()
    s = new()
    header(s, "Aligning each response to its best onset: still no group difference",
           "Ca²⁺ and arteriole each at the onset that maximises its next 10 s; pupil at pupil onset. Day mean ± SEM.",
           METHOD_MAX)
    picture(s, FIG / "1b_psth_maxonset.png", 0.3, 1.6, 12.7, 3.7)
    text(s, 0.6, 5.4, 12.1, 0.8, [
        f"Median onset shift: Ca²⁺ +{sh.loc['SCR', 'ca_shift']:.1f} s (scramble) / +{sh.loc['KD', 'ca_shift']:.1f} s (KD); "
        f"arteriole +{sh.loc['SCR', 'art_shift']:.1f} s / +{sh.loc['KD', 'art_shift']:.1f} s.   Mean 0–15 s: Ca²⁺ "
        f"{T2.loc['ca_mean', 'scr_mean']:+.1f}% vs {T2.loc['ca_mean', 'kd_mean']:+.1f}%; arteriole "
        f"{T2.loc['art_mean', 'scr_mean']:+.1f}% vs {T2.loc['art_mean', 'kd_mean']:+.1f}%.",
        (f"Picking the best onset builds in a rise after t = 0 and a dip just before it; at pupil onset the arteriole "
         f"is {T.loc['art_mean', 'scr_mean']:+.1f}% vs {T.loc['art_mean', 'kd_mean']:+.1f}%.", dict(color=GREY, size=13)),
    ], size=15)
    takeaway(s, "Even when each response is allowed its largest value, KD and scramble overlap.", y=6.35)

    # 3c max-response onset: summary + LME
    s = new()
    header(s, "Mixed model: no significant KD effect on any measure",
           "Best-onset alignment. LME per measure: value ~ group + (1 | recording day), events as observations.",
           METHOD_MAX)
    picture(s, FIG / "3b_summary_maxonset.png", 0.3, 1.5, 12.7, 4.2)
    parts = []
    for c, lab in (("ca_peak", "Ca²⁺ peak"), ("ca_auc", "Ca²⁺ AUC"), ("art_peak", "arteriole peak"),
                   ("art_auc", "arteriole AUC")):
        r = T2.loc[c]
        parts.append(f"{lab} {p_txt(r.welch_p)} (days, LME degenerate)" if r.lme_degenerate
                     else f"{lab} LME {p_txt(r.lme_p)}")
    text(s, 0.6, 5.75, 12.1, 0.5, ";   ".join(parts) + ".", size=14)
    takeaway(s, "No measure differs between KD and scramble, by mixed model or by day means.", y=6.3)

    # 4 per day
    s = new()
    header(s, "Scramble's higher Ca²⁺ comes from one day",
           "Each recording day, mean ± SEM across its events. Top: Ca²⁺. Bottom: arteriole. Green = scramble, red = KD.",
           METHOD)
    picture(s, FIG / "4_per_day.png", 0.3, 1.5, 12.7, 3.3)
    text(s, 0.6, 5.0, 12.1, 1.1, [
        f"Scramble 2025.06.01: Ca²⁺ peak {out_day.ca_peak:.0f}%, about 3× every other day. Without it, scramble Ca²⁺ "
        f"peak is {scr_wo.ca_peak.mean():.0f}% vs KD {T.loc['ca_peak', 'kd_mean']:.0f}%.",
        f"Arteriole: {int((D.art_mean.abs() <= 1).sum())} of {len(D)} days sit within ±1% of baseline over 0–15 s, in both groups.",
    ], size=15)
    takeaway(s, "No day-level pattern separates KD from scramble.", y=6.25)

    # 5 coupling
    s = new()
    c_s, c_k = rho("SCR", "ca_mean", "art_mean"), rho("KD", "ca_mean", "art_mean")
    header(s, "Ca²⁺ and dilation are coupled in both groups",
           "One dot per wake-up, coloured by day. ρ = rank correlation within day (events as units).", METHOD)
    picture(s, FIG / "5_coupling.png", 0.6, 1.5, 9.0, 4.65)
    text(s, 9.9, 1.9, 3.2, 4.0, [
        ("Scramble", dict(bold=True, color="2E7D32")), f"ρ = {c_s.rho:+.2f}, {p_txt(c_s.p)}", "",
        ("KD", dict(bold=True, color="C62828")), f"ρ = {c_k.rho:+.2f}, {p_txt(c_k.p)}", "",
        ("Bigger Ca²⁺ response → bigger dilation, in both groups.", dict(color=GREY, size=13)),
    ], size=15)
    takeaway(s, "The Ca²⁺–vessel coupling at wake-up is not weaker in KD.", y=6.3)

    # 6 arousal
    s = new()
    header(s, "Stronger wake-ups give more Ca²⁺ and dilation in both groups",
           "NREM → wake events split into thirds of pupil rise within each day. Top: scramble. Bottom: KD.", METHOD)
    picture(s, FIG / "6_arousal.png", 0.3, 1.45, 9.6, 4.8)
    lines = []
    for g, lab, col in (("SCR", "Scramble", "2E7D32"), ("KD", "KD", "C62828")):
        a, b = rho(g, "pupil_rise", "ca_mean"), rho(g, "pupil_rise", "art_mean")
        lines += [(f"{lab}", dict(bold=True, color=col)), f"Ca²⁺: ρ = {a.rho:+.2f}, {p_txt(a.p)}",
                  f"Arteriole: ρ = {b.rho:+.2f}, {p_txt(b.p)}", ""]
    lines.append(("ρ = rank correlation of pupil rise vs 0–15 s mean, within day; events as units",
                  dict(color=GREY, size=12)))
    text(s, 10.1, 1.7, 3.0, 4.6, lines, size=14)
    takeaway(s, "Arousal scales the response the same way in KD and scramble.", y=6.35)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT, len(prs.slides), "slides")


if __name__ == "__main__":
    main()
