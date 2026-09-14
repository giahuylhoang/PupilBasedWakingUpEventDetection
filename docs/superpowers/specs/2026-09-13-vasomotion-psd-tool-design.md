# Vasomotion PSD tool: design

Date: 2026-09-13
Status: approved, ready for implementation planning

## Purpose

Members of the lab want to run the vasomotion power-spectral-density analysis on
their own vessel recordings without going through a human. They send area
traces, choose analysis parameters, look at the result on screen, and export
tables they can paste into their own statistics software.

This tool does not do statistics. It produces spectra and band powers. The
group comparison happens in Prism, SPSS, or R, wherever the user already works.

## Users and constraints

**Users are bench scientists who will not open a terminal.** No install, no
command line, no Python environment. If using the tool requires any of those,
they will email a spreadsheet instead and the tool has failed.

**Uploading data is acceptable.** The traces are numeric vessel measurements.
There is no restriction on them leaving the university.

**Longevity is explicitly not a requirement.** The tool needs to be useful now.
If it stops working when the cloud credits run out, that is acceptable; the
analysis modules remain runnable by hand. Do not build a handover mechanism,
a migration path, or multi-account support.

## Input data

Sample file: `data/Tim/Vasomotion analysis compilation_Bal K.xlsx`

Four sheets: `SpVB pre`, `SpVB TMX`, `SmFP pre`, `SmFP TMX`. Sheet names encode
a region (`SpVB`, `SmFP`) and a condition (`pre`, `TMX`).

Each sheet is wide format:

| row | content |
| --- | --- |
| 1 | `Animal ID` then one animal identifier per column |
| 2 | `Sex` then M or F per column |
| 3 onward | sample index in column A, then one value per animal |

In the sample file: 1173 samples, 6 animals per sheet, no missing values.
Values are raw cross-sectional **area** in arbitrary units, ranging roughly 668
to 4856. Per-animal coefficient of variation ranges 2.0 to 15.3 percent.

`pre` and `TMX` sheets for a region contain the **same animal IDs**, so the
design is paired within animal. The two regions contain different animals.

### The critical gap

**The file contains no time column and no sampling rate.** Column A is a sample
index. Without a sampling rate supplied by the user there is no frequency axis,
and a wrong value silently rescales every reported frequency and band power.

Every design decision below that concerns sampling rate follows from this.

## Application flow

Single page, three zones: a sidebar for inputs, a main area for results, and
export controls attached to each result.

Order of operations for the user: upload, set parameters, review on screen,
export.

Results recompute on every parameter change. There is no Run button. At this
data size (about 1173 samples across 6 animals per sheet) computation is
effectively instantaneous, which makes parameter choice something the user
explores rather than commits to blindly.

### Sidebar

Two required inputs, visually separated from everything else because getting
either wrong produces a plausible-looking wrong answer rather than an error:

- **Sampling rate (Hz).** Below the field, a live readout of the consequences:
  number of samples, resulting duration in seconds, lowest resolvable
  frequency, and number of windows available to average. A wrong entry becomes
  visible immediately.
- **Signal is Area or Diameter.** A radio with no default. Area is converted to
  diameter by square root; applying that to diameter data is silently wrong.

Everything else sits under a collapsed Advanced section, described next.

### Main area

In this order:

1. **Group-average spectra figure.** Mean and SEM across animals, per condition,
   per region. The presentable figure.
2. **Band summary table.** One row per animal per condition, one column per
   band. This is the table that goes into statistics software.
3. **Full PSD table.** Frequency against animal. Collapsed by default because it
   is large.
4. **Per-animal spectra.** Small multiples, so a single anomalous animal is
   visible.

All four are displayed on screen as real tables and figures, not only offered
as downloads.

### Export

A download control on each table and figure, plus a single Download Everything
control producing a zip of all four outputs together with `parameters.json`.

## Analysis parameters

Three stages. All are user-selectable; nothing about the method is fixed.

### Stage 1: preprocessing

| Parameter | Options | Default |
| --- | --- | --- |
| Signal is | Area, Diameter | none, must be chosen |
| Normalise | None, dX/X0 median, dX/X0 mean, z-score | dX/X0 median |
| Detrend | None, Constant, Linear | Linear |
| High-pass | Off, or cutoff in Hz plus filter order | 0.005 Hz, 4th order |
| Trim | Drop first N seconds | 0 |

Trim exists because photobleaching and preparation settling occupy the first
minute or two of many recordings, and the high-pass filter is currently the
only defence against them.

### Stage 2: PSD method

A method selector. Sub-parameters appear only for the selected method.

| Method | Sub-parameters |
| --- | --- |
| Periodogram | none |
| Welch | window length, overlap, window function (Hann, Hamming, Blackman, Boxcar) |
| Multitaper Welch | window length, overlap, NW, number of tapers K |
| Multitaper (Thomson) | NW, number of tapers K |

Shared across methods: zero-padding, expressed as a multiplier of the window
length (1x meaning none, 2x and 4x offered), and scaling (density or
spectrum, matching scipy's `scaling` argument).

Zero-padding changes the spacing of the plotted frequency grid, not the
resolvable detail, which is set by window length. The UI states this where the
control appears, because it is a common misunderstanding.

Constraints enforced in the UI: K is capped at `2 * NW - 1`, with the cap shown
rather than silently applied. Window length cannot exceed trace length.

Wherever window length appears, the sampling-rate readout updates to show the
resulting frequency resolution and segment count, so the resolution against
stability trade-off is visible at the moment of choosing.

### Stage 3: output

Maximum frequency (default 0.5 Hz), log or linear axes, and an **editable band
table**. Users define their own bands rather than inheriting the defaults,
because different labs and different vessels use different band edges.

The table is pre-filled with two rows, which the user may edit, delete or add
to:

| Band | Lower | Upper |
| --- | --- | --- |
| ultra_low | 0.005 Hz | 0.04 Hz |
| myogenic | 0.04 Hz | 0.15 Hz |

Band names become column headers in the summary table, so they must be unique
and safe to use as identifiers.

## Compare methods

A toggle that overlays two or three selected methods on the same axes and shows
their band powers side by side with percent differences.

This exists because method choice measurably changes the reported number. On
one recording from the IP3R2 cohort, a single periodogram overstates ultra-low
band power by about 64 percent relative to multitaper, and plain Welch differs
from multitaper by about 12 percent. Letting users see that on their own data is
more convincing than being told, and costs little once the methods already sit
behind one interface.

## Settings and reproducibility

The `parameters.json` written into every results zip **is** the settings file.
It can be loaded back into the app to restore every control to its previous
state. Consequences:

- Every exported result is reproducible by construction.
- "What settings produced this figure" is answerable from the zip alone.

The file records the application version, so an old settings file can be
recognised as predating a change in the analysis code.

A small set of built-in presets is offered in a dropdown:

- **Gordon lab default**: 0.005 Hz high-pass 4th order, 256-sample windows, 50
  percent overlap, multitaper Welch NW=3 K=5, max 0.5 Hz.
- **Raw periodogram**: no windowing, for comparison.
- **Long window**: 512-sample windows, for reaching lower frequencies.

## Architecture

**The analysis core does not import Streamlit.** Six modules, each with one
responsibility and independently testable:

| Module | Responsibility |
| --- | --- |
| `io_excel.py` | wide sheets to a tidy long dataframe: `animal, sex, region, condition, sample, value` |
| `preprocess.py` | area to diameter, normalise, detrend, high-pass, trim |
| `psd_methods.py` | the four methods behind one interface: `(signal, fs, params) -> (freqs, power)` |
| `bands.py` | integrate a PSD over bands to produce the summary table |
| `figures.py` | group-average, per-animal small multiples, method comparison |
| `bundle.py` | assemble tables, figures and `parameters.json` into a zip |

`app.py` contains layout, widgets, and calls into those modules. Nothing else.

`psd_methods.py` is the module that matters most. One interface with four
implementations behind it is what makes the method selector and the
compare-methods feature straightforward, rather than four branching code paths
that drift apart over time.

### Sheet structure detection

The app parses region and condition from sheet names automatically and displays
what it found for confirmation, for example "detected 2 regions x 2 conditions,
6 paired animals each". A manual override is available for files that do not
follow the naming convention.

## Validation

Bad input, not bad code, is the expected failure mode.

- **Sampling rate** must be positive and yield a plausible duration. If the
  implied duration is implausibly short or long, warn with the computed value
  rather than proceeding silently.
- **Window length** cannot exceed trace length.
- **Band edges** must fall inside the resolvable frequency range. Warn when a
  band's lower edge falls below `1 / window_length`, since power there is
  interpolated rather than independently resolved.
- **Non-numeric cells** are reported by animal and row rather than crashing.
- **K against NW** is constrained in the UI as described above.

## Testing

**Analysis core, unit tests with synthetic signals.** A known sine wave at a
known frequency and amplitude must produce a peak at that frequency with the
expected power, for every method. This is the test that catches a wrong
sampling rate, a wrong window length, or a broken normalisation, and it does
not depend on any real data being present.

**End to end, one test on the sample file.** Load
`data/Tim/Vasomotion analysis compilation_Bal K.xlsx`, run with default
parameters, assert the shapes and keys of all four outputs.

## Deployment

A single `Dockerfile`. The image is pushed to Google Cloud Run, which scales to
zero and therefore costs approximately nothing at this usage level.

The same image runs locally with one `docker run` command. Cloud hosting and
local Docker are not two builds; they are two ways of pointing at one artifact.
No authentication, no database, no persisted user data.

## Out of scope

Deliberately excluded, and should not be added without a new decision:

- User accounts, login, or any authentication
- A database, or persistence of uploaded data or results between sessions
- A job queue or any asynchronous processing
- Statistical testing of group differences
- Lomb-Scargle or other irregular-sampling methods, since this data is evenly
  sampled
- Time-frequency analysis such as spectrograms or wavelets
- Event detection, sleep-state classification, or anything from the pupil
  pipeline
- Any handover, migration, or multi-tenant capability

## Open questions

None blocking implementation. The sheet-naming convention is the most likely
source of friction with real user files; the manual override exists for that
reason, and the convention should be revisited once files from more than one
person have been seen.
