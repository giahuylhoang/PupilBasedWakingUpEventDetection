# Brief shared by every figure agent — preliminary figure set, 2026-10-01

Repo: /Users/giahuyhoangle/Code/research/GordonLab/PupilBasedWakingUpEventDetection
Python: ./venv/bin/python -B   (bare `python` lacks numpy). Run from the repo root.

## Hard rules
1. You own exactly ONE script: `scripts/figs_prelim/<yourfig>.py`. Create only that,
   plus outputs named with your figure prefix (e.g. `fig3_...`). Do NOT edit
   `scripts/figs_prelim/common.py`, any test, any other script, or any other agent's
   outputs. If you need a helper, define it inside your own script.
2. Outputs go ONLY to `data/RESULTS_figures_prelim_2026-10-01/` via
   `C.save_fig(fig, "figN_...")` / `C.save_table(df, "figN_...")`, and one notes file
   `data/RESULTS_figures_prelim_2026-10-01/notes/figN.md`.
3. No git, no gcloud, no network, no package installs, no deleting files.
4. Read `scripts/figs_prelim/common.py` FIRST, top to bottom. Its docstring defines
   states, events, normalisation, spectra and stats. Use its functions; do not
   redefine an event, normalisation or band differently.
5. Run your script, then OPEN EVERY PNG you produced (read it as an image) and fix
   anything illegible, empty, clipped, or mislabelled. Iterate until it is a figure a
   PI could read. A panel that silently shows nothing is a failure.

## Data facts (verified)
* `import sys; sys.path.insert(0, "."); from scripts.figs_prelim import common as C`
* Groups: WT = retrosplenial cortex (30 cycles, 8 recording days, has calcium),
  SCR = IP3R2 scramble (16 cyc, 4 days), KD = IP3R2 KD (19 cyc, 6 days with ML-filled
  labels; 7 cyc / 3 days manual-only), NE = GRAB-NE (13 cyc, 3 days; NE channel, NO calcium).
* `C.keys_for("WT")` -> cycles with labels. `C.load_cycle(k)` -> `.sig[name] = (t, y)`
  for names in pupil, whisk, art, ca, ne, eye (only those present), `.labels` per 5-s bin,
  `.info.mouse` (= recording day, THE biological unit), `.info.group`.
* Arteriole/calcium/NE ~1.955 Hz; pupil/whisker/eye resampled to 10 Hz. 900-s cycles.
* Eye motion (`sig['eye']`) exists for only 9 of the 30 labelled WT cycles (GCaMP8m
  2026.03.03 / 03.06) and 3 NE cycles. Never for IP3R2. Treat it as optional.
* Events: `C.event_table(keys)` gives type in N2W / MA / W2N / REM with t0, key, mouse,
  group. Counts (labels as used): WT 130 N2W, 30 MA, 96 W2N, 4 REM; SCR 82/20/64/3;
  NE 72/28/51/7; KD (manual+ML) 73 N2W, 67 MA, 63 W2N.
* `C.epochs_for(ev, signal)` -> (rel_time, matrix) baseline-corrected to -5..0 s.
  `C.response_metrics`, `C.responder` for peak / AUC / latency / response probability.
* Spectra: `C.psd_whole(y, fs)`, `C.psd_state(t, y, fs, labels, state)`, `C.band_power`,
  bands `C.BANDS`. State windows are 60 s: they resolve the vasomotion band
  (0.04-0.15 Hz) but NOT the ultra-slow band (0.005-0.04 Hz) — say so on the panel;
  do not report state-specific ultra-slow power as if it were reliable.
* Stats: `C.paired_test`, `C.group_test`, `C.lmm(df, value, fixed)`, `C.bh`.
* Styling: `C.STATE_COL`, `C.GROUP_COL`, `C.SIG_COL`, `C.SIG_LABEL`, `C.shade_states`,
  `C.n_note`, and put `C.PRELIM_STAMP` in every figure (small text, suptitle or footer).

## Statistical discipline (non-negotiable)
* Unit = recording day (`mouse`). Average within mouse first; plot mouse-level dots
  prominently, cycle- or event-level points faintly behind. Never use bins, events
  or cycles as independent n in a test.
* Where you have event- or cycle-level data, ALSO fit `C.lmm(..., fixed=...)` with a
  random intercept per mouse and report it in a table. If it cannot be fitted, say so.
* Report n (days, cycles, events) on every panel.
* With n of 3-8 days, a Wilcoxon cannot reach small p (see `wilcoxon_min_p`); report it.
* Do not write "significant" for p >= 0.05, and do not imply causation.
* Pupil is not on a common scale across sessions; use only the provided within-cycle z.

## Notes file (notes/figN.md), keep it tight
1. One line per panel: what it shows + the key number(s) with n.
2. Panels you could NOT make and exactly why (missing data, too few events).
3. Caveats a reader must know. 4. How to reproduce (one command).

## Finish with
A short final message: panels produced (file names), the 3-5 headline numbers,
anything that failed or looked wrong, and anything in common.py you think is a bug
(describe it; do not edit it).
