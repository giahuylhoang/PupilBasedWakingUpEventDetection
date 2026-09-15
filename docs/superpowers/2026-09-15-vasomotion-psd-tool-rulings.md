# Vasomotion PSD tool: decisions I made on your behalf

Date: 2026-09-15
Spec: `docs/superpowers/specs/2026-09-13-vasomotion-psd-tool-design.md`
Plan: `docs/superpowers/plans/2026-09-14-vasomotion-psd-tool.md`

You asked me to implement and review the plan without stopping to check in.
This is the complete list of what I decided while you were not in the loop.
The rulings that could change a reported number are first, because those are
the ones worth your disagreement.

---

## 1. Rulings that affect the numbers the tool reports

### R3. Preprocessing stage order

**Decided:** trim → area-to-diameter → normalise → high-pass → detrend.

**Why:** matches `scripts/pub_arteriole_psd.py`, the pipeline whose numbers are
already in the manuscript, with **trim moved to the front** so a photobleaching
transient cannot pull the median used for dX/X0 normalisation.

**If you disagree:** band powers shift. It is a single-line reorder in
`preprocess.py`, and `test_preprocess.py` now has order-sensitive fixtures, so
a reorder will show up as red tests rather than silently.

### C1. `scaling="spectrum"` is rejected for both multitaper methods

**Decided:** raise `ValueError` rather than compute it.

**Why:** scipy's `spectrum` scaling divides by `win.sum()**2`. A DPSS taper
above the first is antisymmetric, so `win.sum()` is approximately zero and the
result explodes — I measured values inflated by six orders of magnitude at
K=5. There is no settled convention in the literature for coherent-gain
correction across a taper bank, so implementing one would mean inventing a
normalisation and presenting it as standard.

**If you disagree:** users lose a combination that never produced a meaningful
number. If you want it, it should be added deliberately with a stated
convention, not as a fallback.

**This is the ruling I am least comfortable making without you** — it removes a
control the spec listed as available. Everything else on this page either adds
a guard or fixes something broken.

---

## 2. Rulings that changed behaviour away from what the plan told implementers to write

The plan is an argument from the spec; where the two disagreed, I ruled for the
spec. Each of these deviates from code the plan spelled out verbatim.

| # | Ruling | Why |
| --- | --- | --- |
| T1-F1 | Made `test_real_sample_file_sheet_names` actually open the workbook | It was named for file I/O it did not perform — a false assurance that would have outlived this plan |
| T1-F2 | A blank Excel cell is now reported, not silently turned into NaN | `isinstance(float('nan'), float)` is `True`, so the plan's check passed blanks through. Latent today (the sample file has none), corrupting on any future edit to the spreadsheet |
| T2-F1 | Added order-sensitive preprocessing fixtures | The reviewer swapped trim and normalise in a scratch copy and **all 24 tests stayed green**. Nothing guarded R3 |
| T2-F2 | Changed the normalise fixture from `[1,2,3,4,5]` | mean == median == 3 on that array, so the median and mean branches were byte-identical and could be swapped undetected |
| T2-promoted | Added a per-animal normalisation leakage test | Scoped Minor by the reviewer; I raised it to Important because a global rather than per-animal median corrupts every animal at once |
| T4 | A band may not be named `animal`, `sex`, `region` or `condition` | **The worst defect found.** Those are valid Python identifiers, so they passed validation, then produced a duplicate column where the band power silently overwrote the group identity. No error, wrong statistics |
| T5 | Added log-axis assertions to all three figure functions | All three could be flipped from log to linear with the full 23-test suite green. On data spanning decades that is a real defect |
| T8-a | Selecting a preset now updates the sidebar widgets | The Method box still read "welch" after selecting a multitaper preset while multitaper was demonstrably running. `parameters.json` was correct, but a bench scientist writes their methods section from the sidebar |
| T8-b | Added a project-wide no-Streamlit-in-core test | The existing check covered only `bundle.py`; adding `import streamlit` to `figures.py` left 131/131 green |
| T8-c | Added a band-edge-validation test for periodogram and Thomson | Those two methods silently skipped `validate_band_edges` |
| T8-d | `parameters.json` no longer records fields the run ignored | It recorded `window: "hann"` for the Raw periodogram preset, which uses boxcar. Structural, not a typo: the variable only exists when the widget is drawn. Inert today, but the spec makes this file both the export *and* the reload format |

---

## 3. Process rulings

- **R1.** Executed on branch `spec/vasomotion-psd-tool`, not a fresh worktree.
  `venv/` is gitignored, so a worktree would have had no Python and every test
  command in the plan would have failed. The branch is not `main`, so isolation
  holds.
- **R2.** Duplicated code *between task bodies in the plan document* is
  intentional — each implementer reads only its own task. Duplicated logic in
  the shipped code is still a defect.
- **R4.** Two Cloud Run services (the root `Dockerfile` for the labeling tool,
  `vasomotion_app/Dockerfile` for this) is the intended topology, not an
  accident. Different tools, different dependencies.
- **R5.** If the Docker daemon were down, I would accept static review of the
  Dockerfile and mark the smoke test unverified rather than block. (It ran.)

### Model-tier rulings

- Task 4's implementer (Haiku) transcribed the plan's 8 reference tests verbatim,
  added none of the 5 cases the dispatch named, and missed 2 entirely — including
  the reserved-band-name defect above. I escalated that fix to Sonnet instead of
  resuming, and ruled Haiku unsuitable for Tasks 5–9.
- You asked me to avoid Opus for everything. I did, with one exception: the
  final whole-branch review. That review is the only thing standing between this
  code and your colleagues' data, and it returned "do not merge" with 2 Critical
  findings that the per-task Sonnet reviews had passed.

---

## 4. What I deferred to you rather than deciding

The skill allows one fix wave after the final review. I spent it on numbers that
are silently wrong and paths that crash. These are known, ship-able, and yours
to call:

| # | Finding | My read |
| --- | --- | --- |
| I1 | The compare-methods figure and its table can show different methods | Presentation gap, visible to the user, not a wrong number |
| I8 | Zero-pad help text doesn't say it adds grid, not resolution | The spec asks for this wording; worth doing before the lab sees it |
| I9 | No per-figure download; figures come only via Download Everything | Spec asks for per-figure. Small, but the spec asked |
| I10 | The log/linear axis control governs one figure of three | Mildly confusing rather than wrong |
| I11 | Some wiring mutations still survive | The three on the spec's flagship inputs are in the current wave; the rest are lower-stakes |
| Minors | Various | Listed in the final review report |

**My recommendation:** I8 and I9 before this goes to Tim, because both are things
the spec promised and a user will notice. I1 and I10 can wait for real feedback.

---

## 5. The one thing you should know about the test suite

The recurring defect across six of nine tasks was **tests that cannot detect the
bug they are named for**. Every one was caught the same way: break the code in a
scratch copy and confirm the tests go red. Nine of these were found and fixed;
the final review still found 12 of 41 mutations surviving, 10 of them in
`app.py` wiring.

That is the honest state of the suite: the analysis core is well covered, the
Streamlit wiring less so.
