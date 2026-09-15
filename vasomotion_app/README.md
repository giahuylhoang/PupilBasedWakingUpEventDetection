# Vasomotion PSD tool

Upload vessel traces, get power spectra and band powers you can paste straight
into Prism, SPSS or R. It does not do statistics — it produces the numbers you
run statistics on.

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
2. **Band summary table** — one row per animal per condition, one column per band. This is the one for your stats software.
3. **Full PSD table** — frequency against animal. Collapsed by default, it's large.
4. **Per-animal spectra** — small multiples, so one odd animal is visible.

Everything is shown on screen and downloadable. **Download Everything** gives a
zip of all four plus `parameters.json`.

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

## Known gaps

- The compare-methods figure and its table can show different method sets.
- The log/linear axis control applies to the group-average figure only.
- Figures download via **Download Everything** rather than individually.

Uploaded data is not stored. There is no login and no database; close the tab
and it is gone.
