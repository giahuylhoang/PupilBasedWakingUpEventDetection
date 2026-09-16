"""Stage 4: statistics on the band-power summary table.

Added after the original spec (which listed statistics as out of scope) at
the lab's request: the point is that nobody has to redo these tests by hand
in another package, so every number a methods section needs is in the
exported table -- n, the test statistic, the raw p, the corrected p, an
effect size, and a confidence interval -- not just a p-value.

Two tables, deliberately kept separate because they answer different
questions and have different designs:

  `paired_band_tests`   Within-animal pre -> post change, per band per
                        region. The workbook's design is genuinely paired
                        (the same animal ID appears in both condition
                        sheets), so this is the test that actually matches
                        the data and it is the primary result.
  `region_change_tests` Between-region difference in that change, per band.
                        Different animals are in each region, so the change
                        scores must be compared UNPAIRED -- pairing them
                        would invent a correspondence between, say, SpVB
                        animal 371146 and SmFP animal 419182 that does not
                        exist.

Design decisions worth knowing before these numbers are reported:

* Tests run on log10 band power, not raw band power. Band power is a
  positive, right-skewed, multiplicative quantity (a vessel's power can
  double or halve, it cannot go negative), so a difference of logs -- i.e.
  a log ratio, reported here as `fold_change` -- is both the better-behaved
  variable for a t-test and the quantity that actually generalises across
  animals with different baseline vessel sizes. A paired test on raw power
  would be dominated by whichever animals happen to have the largest
  vessels.
* Both a parametric and a non-parametric test are reported for every row,
  rather than one being chosen on the analyst's behalf. At these sample
  sizes a normality test has almost no power to justify such a choice, so
  reporting both -- and expecting them to agree -- is the honest option.
  When they disagree, that disagreement is the finding.
* p-values are corrected across every row of each table with
  Benjamini-Hochberg FDR, so the correction spans all bands x regions
  tested in that run rather than each band pretending to be the only
  hypothesis. The uncorrected p stays in the table next to it.
* Wilcoxon signed-rank has a hard floor on the p-value it can return at
  small n (`wilcoxon_min_p`): with 6 pairs, no effect however large can
  produce p < 0.031. That floor is reported per row so a "non-significant"
  non-parametric result at n=6 is not mistaken for evidence of no effect.
* Animals whose two conditions are byte-identical produce a difference of
  exactly zero. That is virtually always a duplicated column in the source
  workbook rather than a real observation (see
  io_excel.check_duplicate_traces), and it biases the paired test toward
  the null, so zero differences are counted in `n_zero_diff` and surfaced
  rather than silently absorbed by scipy's zero-handling.

Benjamini-Hochberg is implemented here in ~15 lines of numpy rather than
imported from statsmodels: statsmodels is not in requirements.txt and
adding it would pull a large dependency (and its own compiled stack) into
the deployed container for one function whose definition is unambiguous.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from vasomotion_app.bands import GROUP_COLUMNS

# Condition names that are recognised as the BASELINE half of a pair, so the
# reported direction is consistently "baseline -> treatment" instead of
# depending on alphabetical accident ('TMX' sorts before 'pre'). Matched
# case-insensitively. A pair with no recognised baseline falls back to sorted
# order, and the chosen direction is always written into the output table's
# `condition_baseline`/`condition_treatment` columns, so the direction of a
# reported fold change is never left implicit.
_BASELINE_NAMES = frozenset({"pre", "baseline", "before", "control", "veh", "vehicle", "day0"})

PAIRED_TEST_COLUMNS = [
    "region", "band",
    "condition_baseline", "condition_treatment",
    "n_pairs", "n_excluded", "n_zero_diff",
    "median_baseline", "median_treatment",
    "fold_change", "mean_log10_ratio",
    "ci95_log10_low", "ci95_log10_high",
    "cohens_dz",
    "t_statistic", "t_p", "t_p_fdr",
    "wilcoxon_statistic", "wilcoxon_p", "wilcoxon_p_fdr",
    "wilcoxon_min_p",
    "notes",
]

REGION_TEST_COLUMNS = [
    "band", "region_a", "region_b",
    "condition_baseline", "condition_treatment",
    "n_a", "n_b",
    "median_log10_ratio_a", "median_log10_ratio_b",
    "fold_change_a", "fold_change_b",
    "welch_t_statistic", "welch_p", "welch_p_fdr",
    "mannwhitney_u", "mannwhitney_p", "mannwhitney_p_fdr",
    "hedges_g",
    "notes",
]


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values, same order as the input.

    NaN entries (a test that could not be run) are passed through as NaN and
    are excluded from the ranking entirely -- they do not inflate the
    correction's denominator, because a test that never ran is not a
    hypothesis that was tested.

    Adjusted values are enforced monotone non-decreasing in p (the standard
    step-up procedure) and clipped at 1.0, so a large raw p can never come
    back with a smaller adjusted value than a smaller raw p.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    n = int(finite.sum())
    if n == 0:
        return out

    idx = np.flatnonzero(finite)
    order = idx[np.argsort(p[idx], kind="stable")]
    ranks = np.arange(1, n + 1, dtype=float)
    scaled = p[order] * n / ranks
    # Step-up: walk from the largest p down, keeping a running minimum, so
    # the result is monotone.
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def wilcoxon_min_p(n_pairs: int) -> float:
    """Smallest two-sided p the exact Wilcoxon signed-rank test can return
    for `n_pairs` non-zero pairs.

    The most extreme outcome is every difference sharing the same sign,
    which has one-sided exact probability 2**-n, so the two-sided floor is
    2**(1 - n): 0.25 at n=3, 0.0625 at n=5, 0.03125 at n=6, 0.0039 at n=9.
    Returns NaN below n=1. This is a property of the test at that sample
    size, not of the data, which is exactly why it belongs in the output
    table -- at n=6 the conventional 0.05 threshold is reachable but 0.01
    is not, no matter how large the true effect.
    """
    if n_pairs < 1:
        return float("nan")
    return float(2.0 ** (1 - n_pairs))


def resolve_condition_order(conditions) -> tuple[str, str]:
    """Pick (baseline, treatment) from exactly two condition labels.

    A label in `_BASELINE_NAMES` (case-insensitive) becomes the baseline;
    otherwise the sorted-first label is used. Raises ValueError for anything
    other than exactly two distinct conditions -- a three-condition design
    is not a paired pre/post comparison and silently picking two of them
    would produce a plausible-looking table answering a question nobody
    asked.
    """
    unique = sorted({str(c) for c in conditions})
    if len(unique) != 2:
        raise ValueError(
            f"a paired pre/post comparison needs exactly 2 conditions, got {len(unique)}: "
            f"{unique}. Reduce the workbook to two conditions, or compare them pairwise "
            "outside this tool."
        )
    lowered = [c.lower() for c in unique]
    for i, name in enumerate(lowered):
        if name in _BASELINE_NAMES:
            other = unique[1 - i]
            return unique[i], other
    return unique[0], unique[1]


def _paired_log_differences(
    baseline_values: np.ndarray, treatment_values: np.ndarray
) -> tuple[np.ndarray, int]:
    """log10(treatment) - log10(baseline) for pairs where both values are
    finite and strictly positive, plus the count of pairs dropped.

    Non-positive band power cannot be log-transformed. It is not an expected
    value for an integrated PSD (power is non-negative, and a band with too
    few resolvable points comes back as NaN from bands.integrate_bands
    rather than as 0.0), so a pair is dropped rather than floored at some
    epsilon -- an invented epsilon would silently set the magnitude of that
    animal's apparent effect.
    """
    a = np.asarray(baseline_values, dtype=float)
    b = np.asarray(treatment_values, dtype=float)
    usable = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    n_excluded = int((~usable).sum())
    return np.log10(b[usable]) - np.log10(a[usable]), n_excluded


def _safe_nanmedian(values: np.ndarray) -> float:
    """Median ignoring NaN, returning NaN for an all-NaN input without
    emitting a RuntimeWarning.

    An all-NaN band is a legitimate input, not a fault: bands.integrate_bands
    returns NaN for a band whose edges fall outside the PSD's frequency
    range, so a band configured above `max_freq` arrives here entirely NaN.
    numpy's `nanmedian` warns ("All-NaN slice encountered") in that case,
    which would print on a perfectly ordinary run and train the reader to
    ignore warnings from this tool.
    """
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _band_names(band_summary: pd.DataFrame, band_names=None) -> list[str]:
    if band_names is not None:
        return list(band_names)
    return [c for c in band_summary.columns if c not in GROUP_COLUMNS]


def paired_band_tests(
    band_summary: pd.DataFrame,
    band_names=None,
    condition_order: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Within-animal paired tests of band power, one row per (region, band).

    `band_summary` is bands.integrate_bands' output: the four GROUP_COLUMNS
    plus one column per band. Only animals present in BOTH conditions of a
    region contribute; an animal recorded in one condition only is counted
    in `n_excluded` and named in `notes`, never silently dropped, because a
    missing post-treatment recording is exactly the kind of thing that
    should be visible before a result is reported (it is how a duplicated
    or absent column shows up as a shrinking n).

    Returns a dataframe with PAIRED_TEST_COLUMNS. FDR correction spans every
    row of the returned table.
    """
    if band_summary.empty:
        return pd.DataFrame(columns=PAIRED_TEST_COLUMNS)

    bands = _band_names(band_summary, band_names)
    baseline, treatment = (
        condition_order
        if condition_order is not None
        else resolve_condition_order(band_summary["condition"].unique())
    )

    rows = []
    for region in sorted(band_summary["region"].unique()):
        region_df = band_summary[band_summary["region"] == region]
        base_df = region_df[region_df["condition"] == baseline].set_index("animal")
        treat_df = region_df[region_df["condition"] == treatment].set_index("animal")
        paired_animals = sorted(
            set(base_df.index) & set(treat_df.index), key=str
        )
        unpaired = sorted(
            set(base_df.index) ^ set(treat_df.index), key=str
        )

        for band in bands:
            notes = []
            if unpaired:
                notes.append(
                    "not in both conditions: " + ", ".join(str(a) for a in unpaired)
                )
            diffs, n_dropped = _paired_log_differences(
                base_df.loc[paired_animals, band].to_numpy(),
                treat_df.loc[paired_animals, band].to_numpy(),
            )
            if n_dropped:
                notes.append(
                    f"{n_dropped} pair(s) dropped: band power was NaN or non-positive, "
                    "so no log ratio exists"
                )
            n_pairs = int(len(diffs))
            n_zero = int(np.count_nonzero(diffs == 0.0))
            if n_zero:
                notes.append(
                    f"{n_zero} pair(s) had an exactly zero change -- check for a duplicated "
                    "column in the source workbook; a true zero biases this test toward "
                    "no effect"
                )

            row = {
                "region": region,
                "band": band,
                "condition_baseline": baseline,
                "condition_treatment": treatment,
                "n_pairs": n_pairs,
                "n_excluded": len(unpaired) + n_dropped,
                "n_zero_diff": n_zero,
                "median_baseline": _safe_nanmedian(
                    base_df.loc[paired_animals, band].to_numpy()
                ) if paired_animals else float("nan"),
                "median_treatment": _safe_nanmedian(
                    treat_df.loc[paired_animals, band].to_numpy()
                ) if paired_animals else float("nan"),
                "wilcoxon_min_p": wilcoxon_min_p(n_pairs - n_zero),
            }
            row.update(_summarise_differences(diffs))
            row.update(_run_paired_tests(diffs))
            row["notes"] = "; ".join(notes)
            rows.append(row)

    out = pd.DataFrame(rows, columns=PAIRED_TEST_COLUMNS)
    if not out.empty:
        out["t_p_fdr"] = benjamini_hochberg(out["t_p"])
        out["wilcoxon_p_fdr"] = benjamini_hochberg(out["wilcoxon_p"])
    return out


def _is_degenerate_variance(diffs: np.ndarray) -> bool:
    """True when the differences are constant to within floating-point
    noise, so no variance-based test is defined on them.

    A plain `std(diffs) > 0` test is not enough. Six differences that are
    each exactly log10(2) in exact arithmetic are NOT bit-identical in
    float64 -- log10(2)-log10(1) and log10(4)-log10(2) differ in the last
    couple of bits -- so their sample SD comes back around 1e-17 instead of
    0. Feeding that to a t-test yields |t| ~ 1e16 and p ~ 1e-100: a
    spectacular, entirely fabricated result whose only input is rounding
    error. Anything whose SD is that small relative to its own mean is
    treated as constant and reported as NaN.

    The 1e-10 relative threshold is far above float64 noise (~1e-16) and far
    below anything a real measurement produces: a genuine SD/|mean| of 1e-10
    would already imply |t| of order 1e10, which is not a number any bench
    recording generates.
    """
    if len(diffs) < 2:
        return True
    sd = float(np.std(diffs, ddof=1))
    if sd == 0.0:
        return True
    mean_abs = abs(float(np.mean(diffs)))
    return sd < 1e-10 * mean_abs


def _summarise_differences(diffs: np.ndarray) -> dict:
    """Effect size and interval for a vector of paired log10 differences.

    `cohens_dz` is mean/SD of the differences (the paired-design effect
    size, not Cohen's d for independent groups). The interval is a t-based
    95% CI on the mean log10 ratio, left on the log scale because that is
    the scale the test ran on; exponentiating its endpoints gives the
    fold-change interval. Both are NaN below 2 pairs, where the SD is
    undefined, and NaN for a variance that is only floating-point noise
    (see `_is_degenerate_variance`) -- a dz of 1e16 is not an effect size.
    """
    n = len(diffs)
    if n == 0:
        return {
            "fold_change": float("nan"), "mean_log10_ratio": float("nan"),
            "ci95_log10_low": float("nan"), "ci95_log10_high": float("nan"),
            "cohens_dz": float("nan"),
        }
    mean_diff = float(np.mean(diffs))
    result = {
        "fold_change": float(10.0 ** mean_diff),
        "mean_log10_ratio": mean_diff,
        "ci95_log10_low": float("nan"),
        "ci95_log10_high": float("nan"),
        "cohens_dz": float("nan"),
    }
    if _is_degenerate_variance(diffs):
        return result
    sd = float(np.std(diffs, ddof=1))
    result["cohens_dz"] = mean_diff / sd
    half_width = scipy_stats.t.ppf(0.975, df=n - 1) * sd / np.sqrt(n)
    result["ci95_log10_low"] = mean_diff - float(half_width)
    result["ci95_log10_high"] = mean_diff + float(half_width)
    return result


def _run_paired_tests(diffs: np.ndarray) -> dict:
    """Paired t-test and Wilcoxon signed-rank on the differences.

    Both are run as one-sample tests against zero on the already-computed
    differences, which is identical to the two-sample paired form and keeps
    the pair-dropping logic in exactly one place (`_paired_log_differences`).

    Every failure mode returns NaN rather than raising: fewer than 2 pairs
    (no variance to test), a variance indistinguishable from floating-point
    noise (`_is_degenerate_variance`), and the all-zero-differences case
    scipy's `wilcoxon` rejects outright. A NaN here is excluded from the FDR
    correction by `benjamini_hochberg`, which is the correct treatment -- a
    test that could not run is not a hypothesis that was tested.

    Wilcoxon is still attempted when the t-test is not, and deliberately so:
    it uses only signs and ranks, so it is unaffected by the vanishing
    variance that makes the t statistic meaningless, and a constant non-zero
    effect is exactly the case where it should return its floor.
    """
    result = {
        "t_statistic": float("nan"), "t_p": float("nan"), "t_p_fdr": float("nan"),
        "wilcoxon_statistic": float("nan"), "wilcoxon_p": float("nan"),
        "wilcoxon_p_fdr": float("nan"),
    }
    if len(diffs) < 2:
        return result

    if not _is_degenerate_variance(diffs):
        t_stat, t_p = scipy_stats.ttest_1samp(diffs, popmean=0.0)
        result["t_statistic"] = float(t_stat)
        result["t_p"] = float(t_p)

    if np.any(diffs != 0.0):
        try:
            w_stat, w_p = scipy_stats.wilcoxon(diffs, zero_method="wilcox")
            result["wilcoxon_statistic"] = float(w_stat)
            result["wilcoxon_p"] = float(w_p)
        except ValueError:
            pass
    return result


def region_change_tests(
    band_summary: pd.DataFrame,
    band_names=None,
    condition_order: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Compare the within-animal change between two regions, per band.

    The change score is each animal's own log10 fold change (the same
    quantity `paired_band_tests` tests against zero), so this asks whether
    the treatment moved one vessel type more than the other. Because the
    two regions contain DIFFERENT animals, the comparison is unpaired:
    Welch's t-test (unequal variances, not Student's) plus Mann-Whitney U.

    Returns one row per band for the single region pair, or an empty table
    when the data does not contain exactly two regions -- with three or more,
    which pair to compare is a question for the analyst, not a default.
    """
    if band_summary.empty:
        return pd.DataFrame(columns=REGION_TEST_COLUMNS)

    regions = sorted(band_summary["region"].unique())
    if len(regions) != 2:
        return pd.DataFrame(columns=REGION_TEST_COLUMNS)

    bands = _band_names(band_summary, band_names)
    baseline, treatment = (
        condition_order
        if condition_order is not None
        else resolve_condition_order(band_summary["condition"].unique())
    )
    region_a, region_b = regions

    def change_scores(region: str, band: str) -> np.ndarray:
        region_df = band_summary[band_summary["region"] == region]
        base_df = region_df[region_df["condition"] == baseline].set_index("animal")
        treat_df = region_df[region_df["condition"] == treatment].set_index("animal")
        animals = sorted(set(base_df.index) & set(treat_df.index), key=str)
        diffs, _ = _paired_log_differences(
            base_df.loc[animals, band].to_numpy(),
            treat_df.loc[animals, band].to_numpy(),
        )
        return diffs

    rows = []
    for band in bands:
        a = change_scores(region_a, band)
        b = change_scores(region_b, band)
        notes = []
        row = {
            "band": band, "region_a": region_a, "region_b": region_b,
            "condition_baseline": baseline, "condition_treatment": treatment,
            "n_a": int(len(a)), "n_b": int(len(b)),
            "median_log10_ratio_a": float(np.median(a)) if len(a) else float("nan"),
            "median_log10_ratio_b": float(np.median(b)) if len(b) else float("nan"),
            "fold_change_a": float(10.0 ** np.mean(a)) if len(a) else float("nan"),
            "fold_change_b": float(10.0 ** np.mean(b)) if len(b) else float("nan"),
            "welch_t_statistic": float("nan"), "welch_p": float("nan"),
            "welch_p_fdr": float("nan"),
            "mannwhitney_u": float("nan"), "mannwhitney_p": float("nan"),
            "mannwhitney_p_fdr": float("nan"),
            "hedges_g": float("nan"),
        }
        if len(a) >= 2 and len(b) >= 2:
            t_stat, t_p = scipy_stats.ttest_ind(a, b, equal_var=False)
            row["welch_t_statistic"] = float(t_stat)
            row["welch_p"] = float(t_p)
            u_stat, u_p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
            row["mannwhitney_u"] = float(u_stat)
            row["mannwhitney_p"] = float(u_p)
            row["hedges_g"] = _hedges_g(a, b)
        else:
            notes.append("needs >= 2 paired animals per region")
        row["notes"] = "; ".join(notes)
        rows.append(row)

    out = pd.DataFrame(rows, columns=REGION_TEST_COLUMNS)
    if not out.empty:
        out["welch_p_fdr"] = benjamini_hochberg(out["welch_p"])
        out["mannwhitney_p_fdr"] = benjamini_hochberg(out["mannwhitney_p"])
    return out


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges' g: Cohen's d on a pooled SD, with the small-sample
    correction factor 1 - 3/(4*(n_a+n_b) - 9) applied. At 6-per-group that
    correction shrinks the estimate by about 8%, which is large enough to
    matter for a reported effect size and is the whole reason Hedges' g is
    preferred over d here.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled <= 0:
        return float("nan")
    d = (np.mean(a) - np.mean(b)) / np.sqrt(pooled)
    correction = 1.0 - 3.0 / (4.0 * (n_a + n_b) - 9.0)
    return float(d * correction)
