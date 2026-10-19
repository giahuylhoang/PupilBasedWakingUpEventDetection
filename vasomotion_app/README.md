# Vasomotion PSD tool

Upload vessel traces, get power spectra, band powers and paired statistics. Every
number is exported as CSV, so you can redraw any figure or rerun any test in
Prism, SPSS or R.

## Two things you must set, every time

The tool cannot infer either of these, and getting either wrong gives you a
plausible-looking wrong answer rather than an error. That is why neither has a
default.

**1. Sampling rate (Hz).** Your spreadsheet has a sample index in column A, not
a time column, so there is no frequency axis until you supply this. A wrong
value rescales every frequency and every band power. Under the field the app
shows you the resulting duration and lowest resolvable frequency — check that
the duration matches how long the recording actually was. That is the fastest
way to catch a typo.

**2. Area or Diameter.** Area is converted to diameter by square root. Applying
that to data which is already diameter is silently wrong. Pick the one your
values actually are.

## Spreadsheet format

One sheet per region/condition, named so the tool can read them, e.g.
`SpVB pre`, `SpVB TMX`, `SmFP pre`, `SmFP TMX`.

| row | content |
| --- | --- |
| 1 | `Animal ID`, then one animal identifier per column |
| 2 | `Sex`, then M or F per column |
| 3 onward | sample index in column A, then one value per animal |

**Delete any trailing `Mean` or `SD` row or column before uploading.** The tool
will refuse the file and tell you where they are, rather than quietly treating
a `Mean` column as an extra animal.

If your sheet names don't follow the convention, use the manual override.

## What you get

1. **Group-average spectra** — mean and SEM across animals. The presentable figure.
2. **Band summary table** — one row per animal per condition, one column per band.
3. **Statistics** — paired within-animal tests per band per region, plus a between-region comparison. See below.
4. **Full PSD table** — frequency against animal. Collapsed by default, it's large.
5. **Per-trace amplitude statistics** — one row per animal per condition, with the standard deviation of the trace and its absolute scale. Collapsed by default.
6. **Per-animal spectra** — small multiples, so one odd animal is visible.

Everything is shown on screen and downloadable. **Download Everything** gives a
zip of all six plus `parameters.json`.

## Plotting the spectra yourself

`full_psd.csv` is the trace data behind every figure, tidy long, one row per
frequency point:

| animal | sex | region | condition | frequency | power |
| --- | --- | --- | --- | --- | --- |

Filter to a region and condition, plot `power` against `frequency`, and you have
the same curve the app drew. Two things to know:

- It is **per animal**, not the group average. The group-average figure is the
  mean across animals at each frequency, with SEM; compute that yourself from
  these rows if you want to redraw it.
- It is truncated at **Max frequency**. If you want to plot beyond 0.5 Hz, raise
  that control before downloading — the rows above it are not in the file.

## Bandpower differences

`band_summary.csv` is one row per animal per condition, one column per band. The
app now runs the paired comparison for you (below), but if you want the raw
per-animal differences:

```r
# R
library(tidyr); library(dplyr)
read.csv("band_summary.csv") |>
  select(animal, region, condition, ultra_low) |>
  pivot_wider(names_from = condition, values_from = ultra_low) |>
  mutate(diff = TMX - pre, pct = 100 * (TMX - pre) / pre)
```

## Statistics

Two tables, because they answer different questions and have different designs.

**`stats_paired.csv` — within-animal change, per band per region.** This is the
primary result. Your workbook is a genuinely paired design (the same animal ID
appears in both condition sheets), so this is the test that matches the data.

**`stats_region.csv` — is the change different between regions?** Each animal's
own fold change, compared between the two regions. The regions contain
*different* animals, so this comparison is **unpaired** — Welch's t-test (not
Student's) plus Mann-Whitney U.

### What the columns mean

| column | what it is |
| --- | --- |
| `fold_change` | The effect. 1.0 is no change, 2.0 is a doubling, 0.5 is a halving. |
| `mean_log10_ratio`, `ci95_log10_*` | The same thing on the log scale the test ran on. Raise 10 to these to get a fold-change interval. |
| `cohens_dz` | Paired effect size (mean/SD of the differences). |
| `t_p`, `wilcoxon_p` | Uncorrected p, parametric and non-parametric. |
| `t_p_fdr`, `wilcoxon_p_fdr` | Benjamini-Hochberg corrected across every row of the table. **Report these.** |
| `n_pairs`, `n_excluded`, `n_zero_diff` | Who actually contributed, and who didn't. |
| `wilcoxon_min_p` | The smallest p Wilcoxon *can* return at this n. See below. |
| `notes` | Why an animal was excluded, in plain words. |

### Four things to know before you report these

**Tests run on log10 band power, not raw power.** Band power is positive and
multiplicative — a vessel's power can double or halve, it cannot go negative — so
a difference of logs (a fold change) is both better behaved and the quantity that
generalises across animals with different baseline vessel sizes. A paired test on
raw power would be dominated by whichever animals have the largest vessels.

**Both a parametric and a non-parametric p are reported, deliberately.** At n=6 a
normality test has almost no power to justify picking one, so you get both. Them
agreeing is your reassurance. Them disagreeing is itself the finding.

**Wilcoxon has a hard floor at small n.** With 6 pairs it cannot return p < 0.031
no matter how large the effect; with 5 pairs, p < 0.063. That is a fact about the
test at that sample size, not about your data, which is why `wilcoxon_min_p` is
in the table. A non-significant Wilcoxon result at n=6 is a statement about n.

**A blank p means the test could not run**, not p=0. It happens when fewer than
two animals paired, or when the change is identical across every animal (no
variance to test). Blanks are excluded from the FDR correction rather than
counted as hypotheses.

### If you turn normalisation off

Raw band power still carries each animal's absolute vessel size. The paired tests
are largely protected — each animal is its own control — but any comparison
*between* animals or regions is confounded by baseline vessel diameter. The app
warns you, and `mean_raw` in `trace_stats.csv` is that baseline if you want it as
a covariate. Normalising is the better option.

## Duplicate columns are checked, and this has already mattered

The tool compares every trace against every other and refuses to be quiet about
exact matches. Two vessels cannot produce identical floating-point values, so an
exact match is a copy-pasted column, not a measurement.

This is not hypothetical. In the first real workbook, **animal 461672's
`SpVB TMX` column was a byte-for-byte copy of its `SpVB pre` column**, all 1173
samples. Nothing else caught it: both columns were fully numeric and the right
length, the animal appeared correctly in both conditions, and the two identical
PSD curves overplotted into what looked like a single condition on screen. Its
contribution to the paired test was a change of exactly zero, which drags the
estimated effect toward "no effect" while still counting toward n.

If you see this error, fix the workbook. No statistical handling repairs a column
that was never recorded.

## Normalisation — upload raw, the pipeline does it

**Upload your raw values. You do not need to normalise anything first.** For
every animal, in every condition, the tool normalises that trace against its
own baseline before computing anything. Nothing is shared between animals or
between conditions.

Pick the mode under **Advanced: preprocessing → Normalise**:

| mode | baseline used |
| --- | --- |
| `dx_x0_median` (default) | That trace's whole-recording median. |
| `dx_x0_mean` | Same, using the mean. |
| `dx_x0_rolling` | A **local** baseline: a centred rolling median, window length of your choosing. |
| `zscore` | Mean-centred, divided by SD. |
| `none` | No normalisation. Power then carries absolute vessel size. |

### "Shouldn't the baseline be local to each analysis window?"

A reasonable worry, and the answer has a hard constraint in it.

Subtracting a local baseline high-passes the trace at roughly 1/window:
anything slower than the baseline window is absorbed into the "baseline" and
removed. So **the baseline window must be longer than the longest period you
want to measure.** Measured here on a pure 0.01 Hz (100 s) oscillation, as the
fraction of amplitude that survives:

| baseline window | amplitude recovered |
| --- | --- |
| 10 s | 15% |
| 30 s | 45% |
| 65 s | 87% |
| 100 s | 100% |

A band edge at 0.005 Hz is a 200 s period, so it needs a baseline window of at
least 200 s. On a 5-minute recording that is two-thirds of the whole trace — at
which point a "local" baseline is barely distinguishable from the global one.
That is not an implementation limit; "local baseline" and "slow fluctuation" are
the same thing at the same timescale.

The app enforces this: with `dx_x0_rolling` it warns whenever the baseline
window is shorter than `1 / lowest band edge`, naming both numbers. The default
is 200 s, matched to the default bands.

**Measured on a real 5-minute workbook**, normalising each 65.5 s analysis
window by its own median instead of the global median moved the myogenic band
(0.04–0.15 Hz) by a median of **4.5%** — but cut ultra-low band power to a
median of **0.49×**, worst case 0.11×. That ultra-low change is destruction, not
correction.

### The thing that matters more here: window length

At 3.91 Hz the default 256-sample window is 65.5 s, so the lowest frequency it
can resolve is **0.0153 Hz**. A band starting at 0.005 Hz is below that — the app
warns, and power there is interpolated rather than measured.

For an ultra-low band on a 5-minute recording, use **multitaper Thomson**: it
uses the whole 300 s trace (lowest resolvable 0.0033 Hz) and still averages
several tapers. On the real workbook that raised ultra-low band power by a median
of **2.2×** versus the 65.5 s Welch window, and cleared the band-edge warning.

A useful side effect: when the whole trace is one window, global and local
baselines are the same thing, so the question above stops mattering.

## Trace variability (SD after normalisation)

`trace_stats.csv` has one row per animal per condition. It reports **two**
standard deviations, because "SD after normalisation" has two answers and they
are not the same number:

| column | what it is |
| --- | --- |
| `sd_normalized` | SD straight after normalisation, **before** the high-pass filter and detrend. Slow drift included. Usually what "how much did the vessel fluctuate" means. |
| `sd_analyzed` | SD of the trace the PSD was actually computed from, after high-pass and detrend. Always the smaller of the two. |
| `mean_raw`, `sd_raw` | The absolute scale, before normalisation — what dx/x₀ divided out. |
| `n_samples`, `duration_sec` | After trimming. |

Use `sd_normalized` for reporting variability. The gap between the two depends
entirely on your high-pass cutoff: at the default 0.005 Hz it is small (about
1.2× on our sample workbook), but a higher cutoff widens it sharply.

Two traps:

- **Units follow the Area/Diameter setting.** With Area selected the tool takes
  the square root first, so `mean_raw` and `sd_raw` are in **diameter** units,
  not area, and both SDs describe a normalised diameter trace.
- **With z-score normalisation, `sd_normalized` is 1.0 for every trace** by
  definition, so it tells you nothing. The app says so on screen. Use `sd_raw`,
  or switch to dx/x₀.

`sd_analyzed` is close to total band power by construction (for a mean-zero
trace, variance is the integral of the PSD — measured agreement on our sample
workbook was about 0.94), so treat it as a check on the PSD rather than as an
independent measure.

## parameters.json

It is both the record of what you ran and the settings file. Load it back in to
restore every control. So "what settings produced this figure" is answerable
from the zip alone — keep it with your results.

## Choosing a method

Method choice changes the reported number. On one of our recordings a plain
periodogram overstated ultra-low band power by about 64% relative to
multitaper, and Welch differed from multitaper by about 12%. Use **Compare
methods** to see that on your own data before you commit to one.

Defaults (Gordon lab preset): 0.005 Hz high-pass 4th order, 256-sample windows,
50% overlap, multitaper Welch NW=3 K=5, up to 0.5 Hz.

Zero-padding changes the spacing of the plotted frequency grid, not the detail
you can actually resolve — that is set by window length.

## Running it yourself

```bash
docker build -t vasomotion-psd vasomotion_app
docker run -p 8080:8080 vasomotion-psd
# then open http://localhost:8080
```

## Log or linear frequency axis?

Depends what you are showing, so the control applies to every figure — group
average, per-animal small multiples and the method comparison.

- **Log** for inspecting low frequencies and 1/f slope. This is the default and
  the right choice most of the time.
- **Linear** to highlight a sharp, discrete peak in a narrow window (say
  0–0.1 Hz), where a log axis compresses the peak into the left margin.

If you are sending figures to someone who will pick one, export both.

## Known gaps

- The compare-methods figure and its table can show different method sets.
- Figures download via **Download Everything** rather than individually.
- The between-region comparison needs exactly two regions; with three or more,
  which pair to compare is your call, so the table is left empty.

Uploaded data is not stored. There is no login and no database; close the tab
and it is gone.
