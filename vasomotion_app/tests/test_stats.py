"""Tests for vasomotion_app.stats.

The tests that matter most here are the ones pinning behaviour that would be
silently wrong rather than loudly broken: the Benjamini-Hochberg step-up
(which is easy to implement as a non-monotone p*n/rank and never notice), the
Wilcoxon small-n p-value floor (verified against scipy rather than restated
from the formula), and the zero-difference accounting that the real workbook's
duplicated column produces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from vasomotion_app import stats


# --------------------------------------------------------------------------
# Benjamini-Hochberg
# --------------------------------------------------------------------------

def test_bh_uniformly_spaced_pvalues_all_adjust_to_the_same_value():
    # p_i = i/100 for i in 1..5 with n=5: every p*n/rank is exactly 0.05,
    # so the whole set adjusts to 0.05. A hand-checkable fixed point.
    out = stats.benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
    assert np.allclose(out, 0.05)


def test_bh_scales_by_n_over_rank():
    out = stats.benjamini_hochberg([0.001, 0.5])
    assert np.allclose(out, [0.002, 0.5])


def test_bh_preserves_input_order_not_sorted_order():
    out = stats.benjamini_hochberg([0.04, 0.01])
    # sorted: 0.01 at rank 1 -> 0.02; 0.04 at rank 2 -> 0.04
    assert np.allclose(out, [0.04, 0.02])


def test_bh_is_monotone_non_decreasing_in_p():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50)
    adjusted = stats.benjamini_hochberg(p)
    by_p = adjusted[np.argsort(p)]
    assert np.all(np.diff(by_p) >= -1e-12), "adjusted p must not decrease as raw p increases"


def test_bh_clips_at_one():
    assert np.all(stats.benjamini_hochberg([0.9, 0.95, 0.99]) <= 1.0)


def test_bh_passes_nan_through_and_excludes_it_from_the_denominator():
    # Two real tests plus one that could not run. The correction must use
    # n=2, not n=3: a test that never ran is not a hypothesis that was
    # tested, and counting it would make every other p more conservative.
    out = stats.benjamini_hochberg([0.01, float("nan"), 0.02])
    assert np.isnan(out[1])
    assert np.allclose([out[0], out[2]], [0.02, 0.02])


def test_bh_all_nan_returns_all_nan():
    out = stats.benjamini_hochberg([float("nan"), float("nan")])
    assert np.all(np.isnan(out))


def test_bh_empty_input():
    assert stats.benjamini_hochberg([]).shape == (0,)


# --------------------------------------------------------------------------
# Wilcoxon floor
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [(1, 1.0), (3, 0.25), (5, 0.0625), (6, 0.03125), (9, 0.00390625)])
def test_wilcoxon_min_p_values(n, expected):
    assert stats.wilcoxon_min_p(n) == pytest.approx(expected)


def test_wilcoxon_min_p_is_nan_below_one_pair():
    assert np.isnan(stats.wilcoxon_min_p(0))


@pytest.mark.parametrize("n", [5, 6, 7, 8])
def test_wilcoxon_min_p_agrees_with_scipy_on_the_most_extreme_data(n):
    """The floor is verified against scipy, not just restated from the
    formula: all differences the same sign is the most extreme outcome
    available, so scipy's exact p for it IS the floor."""
    diffs = np.arange(1.0, n + 1.0)  # all positive, all distinct ranks
    _, p = scipy_stats.wilcoxon(diffs, zero_method="wilcox")
    assert p == pytest.approx(stats.wilcoxon_min_p(n))


# --------------------------------------------------------------------------
# Condition ordering
# --------------------------------------------------------------------------

def test_resolve_condition_order_puts_pre_first_regardless_of_sort_order():
    # 'TMX' sorts before 'pre' alphabetically, so a naive sorted() would
    # report the fold change upside down.
    assert stats.resolve_condition_order(["TMX", "pre"]) == ("pre", "TMX")
    assert stats.resolve_condition_order(["pre", "TMX"]) == ("pre", "TMX")


def test_resolve_condition_order_is_case_insensitive_about_the_baseline():
    assert stats.resolve_condition_order(["TMX", "Pre"]) == ("Pre", "TMX")


def test_resolve_condition_order_falls_back_to_sorted_when_no_baseline_recognised():
    assert stats.resolve_condition_order(["zeta", "alpha"]) == ("alpha", "zeta")


@pytest.mark.parametrize("conditions", [["only"], ["a", "b", "c"], []])
def test_resolve_condition_order_rejects_anything_but_two_conditions(conditions):
    with pytest.raises(ValueError, match="exactly 2 conditions"):
        stats.resolve_condition_order(conditions)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _band_summary(rows):
    return pd.DataFrame(rows, columns=["animal", "sex", "region", "condition", "alpha"])


def _paired_summary(baseline_values, treatment_values, region="R1"):
    rows = []
    for i, v in enumerate(baseline_values, start=1):
        rows.append([i, "M", region, "pre", v])
    for i, v in enumerate(treatment_values, start=1):
        rows.append([i, "M", region, "TMX", v])
    return _band_summary(rows)


# --------------------------------------------------------------------------
# paired_band_tests
# --------------------------------------------------------------------------

def test_paired_exact_doubling_gives_fold_change_two():
    bs = _paired_summary([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], [2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    out = stats.paired_band_tests(bs)
    row = out.iloc[0]
    assert row["n_pairs"] == 6
    assert row["fold_change"] == pytest.approx(2.0)
    assert row["mean_log10_ratio"] == pytest.approx(np.log10(2.0))
    assert row["condition_baseline"] == "pre"
    assert row["condition_treatment"] == "TMX"


def test_paired_zero_variance_effect_runs_wilcoxon_but_not_the_t_test():
    """A perfectly constant fold change has zero variance in the
    differences, so the t statistic is undefined. It must come back NaN
    rather than as a spuriously enormous value -- while Wilcoxon, which uses
    only signs and ranks, still returns its floor.

    Note this is NOT an exactly-zero SD in float64: log10(2)-log10(1) and
    log10(4)-log10(2) differ in their last bits, so the naive `std > 0`
    guard passes here on about 1e-17 of rounding noise and produces
    |t| ~ 1e16 with p ~ 1e-100. That fabricated p-value is what this test
    exists to prevent."""
    bs = _paired_summary([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], [2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    row = stats.paired_band_tests(bs).iloc[0]
    assert np.isnan(row["t_statistic"]) and np.isnan(row["t_p"])
    assert np.isnan(row["cohens_dz"])
    assert np.isnan(row["ci95_log10_low"]) and np.isnan(row["ci95_log10_high"])
    assert row["wilcoxon_p"] == pytest.approx(0.03125)


def test_degenerate_variance_detects_float_noise_but_not_real_variation():
    """The guard must fire on constant-to-machine-precision data and stay
    out of the way for anything a real recording produces."""
    constant = np.diff(np.log10([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]))
    assert np.std(constant, ddof=1) > 0, "precondition: float noise, not an exact zero SD"
    assert stats._is_degenerate_variance(constant)
    assert stats._is_degenerate_variance(np.zeros(6))
    assert stats._is_degenerate_variance(np.array([1.0]))
    # A tiny but genuine spread: SD/|mean| = 1e-4, far above the threshold.
    real = np.array([0.30103, 0.30106, 0.30100, 0.30104, 0.30102, 0.30105])
    assert not stats._is_degenerate_variance(real)


def test_paired_direction_is_reported_correctly_for_a_decrease():
    bs = _paired_summary([10.0] * 6, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    row = stats.paired_band_tests(bs).iloc[0]
    assert row["fold_change"] < 1.0
    assert row["mean_log10_ratio"] < 0
    assert row["cohens_dz"] < 0


def test_paired_matches_scipy_on_the_log_differences():
    baseline = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    treatment = [1.5, 2.1, 4.0, 3.5, 8.0, 7.0]
    row = stats.paired_band_tests(_paired_summary(baseline, treatment)).iloc[0]
    diffs = np.log10(np.array(treatment)) - np.log10(np.array(baseline))
    t_stat, t_p = scipy_stats.ttest_1samp(diffs, 0.0)
    assert row["t_statistic"] == pytest.approx(t_stat)
    assert row["t_p"] == pytest.approx(t_p)
    assert row["cohens_dz"] == pytest.approx(np.mean(diffs) / np.std(diffs, ddof=1))


def test_paired_ci_brackets_the_mean_log_ratio():
    baseline = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    treatment = [1.5, 2.1, 4.0, 3.5, 8.0, 7.0]
    row = stats.paired_band_tests(_paired_summary(baseline, treatment)).iloc[0]
    assert row["ci95_log10_low"] < row["mean_log10_ratio"] < row["ci95_log10_high"]


def test_paired_counts_an_exactly_zero_change_and_raises_the_wilcoxon_floor():
    """The animal 461672 case: one animal's two conditions are identical, so
    its difference is exactly zero. It must be counted, and the reported
    Wilcoxon floor must reflect the pair scipy's zero_method drops -- 5
    effective pairs, not 6."""
    bs = _paired_summary([1.0, 2.0, 4.0, 8.0, 16.0, 7.0], [2.0, 4.0, 8.0, 16.0, 32.0, 7.0])
    row = stats.paired_band_tests(bs).iloc[0]
    assert row["n_pairs"] == 6
    assert row["n_zero_diff"] == 1
    assert row["wilcoxon_min_p"] == pytest.approx(0.0625)  # 2**(1-5), not 2**(1-6)
    assert "exactly zero change" in row["notes"]


def test_paired_all_zero_differences_returns_nan_not_a_crash():
    """scipy.wilcoxon raises when every difference is zero. That is the
    fully-duplicated-workbook case and must degrade to NaN."""
    bs = _paired_summary([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    row = stats.paired_band_tests(bs).iloc[0]
    assert row["n_zero_diff"] == 3
    assert np.isnan(row["wilcoxon_p"])
    assert np.isnan(row["t_p"])
    assert row["fold_change"] == pytest.approx(1.0)


def test_paired_excludes_an_animal_present_in_only_one_condition():
    rows = [
        [1, "M", "R1", "pre", 1.0], [2, "M", "R1", "pre", 2.0], [3, "M", "R1", "pre", 4.0],
        [1, "M", "R1", "TMX", 2.0], [2, "M", "R1", "TMX", 4.0],
    ]
    row = stats.paired_band_tests(_band_summary(rows)).iloc[0]
    assert row["n_pairs"] == 2
    assert row["n_excluded"] == 1
    assert "not in both conditions: 3" in row["notes"]


def test_paired_drops_pairs_whose_band_power_is_nan_or_non_positive():
    bs = _paired_summary([1.0, 2.0, float("nan"), 0.0], [2.0, 4.0, 8.0, 8.0])
    row = stats.paired_band_tests(bs).iloc[0]
    assert row["n_pairs"] == 2
    assert row["n_excluded"] == 2
    assert "no log ratio exists" in row["notes"]


def test_paired_one_row_per_region_and_band():
    rows = []
    for region in ("R1", "R2"):
        for i in range(1, 4):
            rows.append([i, "M", region, "pre", float(i)])
            rows.append([i, "M", region, "TMX", float(i) * 2])
    df = pd.DataFrame(rows, columns=["animal", "sex", "region", "condition", "alpha"])
    df["beta"] = df["alpha"] * 3
    out = stats.paired_band_tests(df)
    assert len(out) == 4
    assert set(out["region"]) == {"R1", "R2"}
    assert set(out["band"]) == {"alpha", "beta"}


def test_paired_fdr_spans_every_row_of_the_table():
    rows = []
    for region in ("R1", "R2"):
        for i, (b, t) in enumerate(
            zip([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.5, 2.1, 4.0, 3.5, 8.0, 7.0]), start=1
        ):
            rows.append([i, "M", region, "pre", b])
            rows.append([i, "M", region, "TMX", t])
    out = stats.paired_band_tests(_band_summary(rows))
    expected = stats.benjamini_hochberg(out["t_p"])
    assert np.allclose(out["t_p_fdr"], expected, equal_nan=True)
    # Correction over 2 rows must make every adjusted p at least its raw p.
    assert np.all(out["t_p_fdr"] >= out["t_p"] - 1e-12)


def test_paired_returns_declared_columns_and_survives_an_empty_input():
    out = stats.paired_band_tests(pd.DataFrame(columns=["animal", "sex", "region", "condition"]))
    assert list(out.columns) == stats.PAIRED_TEST_COLUMNS
    assert out.empty


def test_paired_respects_an_explicit_condition_order():
    bs = _paired_summary([1.0, 2.0, 4.0], [2.0, 4.0, 8.0])
    flipped = stats.paired_band_tests(bs, condition_order=("TMX", "pre")).iloc[0]
    assert flipped["condition_baseline"] == "TMX"
    assert flipped["fold_change"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# region_change_tests
# --------------------------------------------------------------------------

def _two_region_summary():
    rows = []
    for i, (b, t) in enumerate(zip([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]), start=1):
        rows.append([i, "M", "R1", "pre", b])
        rows.append([i, "M", "R1", "TMX", t])
    for i, (b, t) in enumerate(zip([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]), start=11):
        rows.append([i, "M", "R2", "pre", b])
        rows.append([i, "M", "R2", "TMX", t])
    return _band_summary(rows)


def test_region_change_detects_a_difference_between_regions():
    """R1 doubles, R2 does not change at all -- a clean separation of the
    two regions' change scores."""
    row = stats.region_change_tests(_two_region_summary()).iloc[0]
    assert row["n_a"] == 4 and row["n_b"] == 4
    assert row["fold_change_a"] == pytest.approx(2.0)
    assert row["fold_change_b"] == pytest.approx(1.0)
    assert row["welch_p"] < 0.001
    # Asserted against scipy rather than the exact-test literal (2/70 =
    # 0.0286): region B's change scores are all identically zero, so the
    # ties push scipy off its exact method onto the tie-corrected asymptotic
    # one. Pinning the literal would have pinned the wrong test.
    a = np.log10([2.0, 4.0, 6.0, 8.0]) - np.log10([1.0, 2.0, 3.0, 4.0])
    _, expected_p = scipy_stats.mannwhitneyu(a, np.zeros(4), alternative="two-sided")
    assert row["mannwhitney_p"] == pytest.approx(expected_p)
    assert row["mannwhitney_p"] < 0.05


def test_region_change_uses_welch_not_student():
    """Asserted by matching scipy with equal_var=False. Student's t on these
    change scores gives a different statistic, so this pins the choice."""
    bs = _two_region_summary()
    a = np.log10([2.0, 4.0, 6.0, 8.0]) - np.log10([1.0, 2.0, 3.0, 4.0])
    b = np.zeros(4)
    row = stats.region_change_tests(bs).iloc[0]
    welch = scipy_stats.ttest_ind(a, b, equal_var=False)
    assert row["welch_t_statistic"] == pytest.approx(welch.statistic)
    assert row["welch_p"] == pytest.approx(welch.pvalue)


def test_region_change_is_empty_unless_there_are_exactly_two_regions():
    one_region = _paired_summary([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    out = stats.region_change_tests(one_region)
    assert out.empty
    assert list(out.columns) == stats.REGION_TEST_COLUMNS

    three = pd.concat([
        _paired_summary([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], region=r) for r in ("R1", "R2", "R3")
    ], ignore_index=True)
    assert stats.region_change_tests(three).empty


def test_region_change_needs_two_animals_per_region():
    rows = [
        [1, "M", "R1", "pre", 1.0], [1, "M", "R1", "TMX", 2.0],
        [2, "M", "R2", "pre", 1.0], [2, "M", "R2", "TMX", 2.0],
    ]
    row = stats.region_change_tests(_band_summary(rows)).iloc[0]
    assert np.isnan(row["welch_p"])
    assert "needs >= 2 paired animals per region" in row["notes"]


def test_hedges_g_applies_the_small_sample_correction():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = a + 2.0
    g = stats._hedges_g(a, b)
    pooled = np.sqrt(((5 * np.var(a, ddof=1)) + (5 * np.var(b, ddof=1))) / 10)
    d = (np.mean(a) - np.mean(b)) / pooled
    correction = 1.0 - 3.0 / (4.0 * 12 - 9.0)
    assert g == pytest.approx(d * correction)
    assert abs(g) < abs(d), "the correction must shrink the estimate"
