"""End-to-end test: load the real sample workbook and run the full pipeline
with the Gordon lab default preset. All numbers below were computed directly
from data/Tim/Vasomotion analysis compilation_Bal K.xlsx during planning
(openpyxl inspection + a reference run of the exact module logic in this
plan) -- they are not placeholders.

The workbook has no sampling rate of its own (that is the tool's reason to
exist -- see the spec's "critical gap"); fs=2.0 Hz is this test's own fixed
choice so the pipeline has something to run with, not a fact discovered from
the file. Nothing in this file, or in any module it exercises, reads a
sampling rate off the workbook -- there isn't one. Do not mistake FS below
for something measured; it is an arbitrary, documented assumption.

Beyond the two tests' literal shape/value checks, this file adds identity
checks that only real data can exercise: SpVB and SmFP are two disjoint
6-animal cohorts, and each cohort's exact animal-ID set recurs unchanged
across the pre/TMX condition pair (a repeated-measures design). A bug that
merged, swapped, or crossed conditions or regions could still produce a
table with the right shape and no NaNs -- so shape/NaN checks alone would
not catch it. The animal-ID sets are asserted both on the loader's output
(test 1) and on the pipeline's final band_summary (test 2), so a relabelling
bug introduced anywhere in preprocess -> psd_methods -> bands would also be
caught, not just a bug in io_excel itself. Every Figure run_pipeline returns
is closed in a `finally` block so a failed assertion still cannot leak
figures into later tests.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from vasomotion_app.bundle import PRESETS, psd_params_from_dict, run_pipeline
from vasomotion_app.io_excel import load_workbook_long

SAMPLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "Tim",
    "Vasomotion analysis compilation_Bal K.xlsx",
)
FS = 2.0

# The two regions' animal cohorts (6 animals each, disjoint), verified by
# direct openpyxl inspection to recur identically across the pre/TMX sheet
# pair for that region. Hardcoded here (not re-derived) so a test using them
# fails the same way regardless of which side of the comparison broke.
SPVB_ANIMALS = {371146, 410328, 410331, 410333, 419181, 461672}
SMFP_ANIMALS = {417503, 419182, 419183, 461668, 461669, 461670}

pytestmark = pytest.mark.skipif(
    not os.path.exists(SAMPLE_PATH),
    reason="sample workbook not present in this checkout",
)


def test_sample_file_loads_to_expected_shape():
    df = load_workbook_long(SAMPLE_PATH)
    assert len(df) == 28152  # 4 sheets x 6 animals x 1173 samples
    assert set(df["region"].unique()) == {"SpVB", "SmFP"}
    assert set(df["condition"].unique()) == {"pre", "TMX"}
    assert df["animal"].nunique() == 12
    assert df.groupby(["animal", "region", "condition"]).size().max() == 1173
    assert df.groupby(["animal", "region", "condition"]).size().min() == 1173
    # A specific, real cell: SpVB pre, animal 371146, sample 1 == 4417 (verified via openpyxl)
    cell = df[(df.animal == 371146) & (df.region == "SpVB") & (df.condition == "pre") & (df["sample"] == 1)]
    assert cell.iloc[0]["value"] == 4417.0
    assert cell.iloc[0]["sex"] == "M"

    # Real-data identity check the counts above cannot catch: a bug that
    # merged or swapped the pre/TMX sheets, or crossed the two regions'
    # animal pools, would still leave every count above unchanged (24 groups
    # of 1173 rows, 12 unique animals total either way).
    for condition in ("pre", "TMX"):
        spvb_animals = set(df[(df.region == "SpVB") & (df.condition == condition)]["animal"])
        smfp_animals = set(df[(df.region == "SmFP") & (df.condition == condition)]["animal"])
        assert spvb_animals == SPVB_ANIMALS, f"SpVB {condition} animal set changed: {spvb_animals}"
        assert smfp_animals == SMFP_ANIMALS, f"SmFP {condition} animal set changed: {smfp_animals}"
    assert SPVB_ANIMALS.isdisjoint(SMFP_ANIMALS)

    # Real value range (not the "roughly 668-4856" ballpark quoted going in --
    # the true minimum, from animal 419183/SmFP/pre/sample 920, is 249.0).
    assert df["value"].min() == 249.0
    assert df["value"].max() == 4856.0


def test_run_pipeline_on_sample_file_with_gordon_lab_default():
    df = load_workbook_long(SAMPLE_PATH)
    preset = PRESETS["Gordon lab default"]
    psd_params = psd_params_from_dict(preset["psd"])
    band_defs = {name: tuple(edges) for name, edges in preset["bands"].items()}

    result = run_pipeline(
        df, fs=FS, signal_type="area", normalize=preset["normalize"],
        detrend_method=preset["detrend"], highpass_cutoff=preset["highpass_cutoff"],
        highpass_order=preset["highpass_order"], trim_seconds=preset["trim_seconds"],
        psd_params=psd_params, band_defs=band_defs, max_freq=preset["max_freq"],
    )
    try:
        assert set(result.keys()) == {"full_psd", "band_summary", "figures"}

        full_psd = result["full_psd"]
        assert list(full_psd.columns) == ["animal", "sex", "region", "condition", "frequency", "power"]
        assert len(full_psd) == 1560  # 24 (animal,region,condition) groups x 65 freq points <= 0.5 Hz
        assert full_psd["frequency"].max() == pytest.approx(0.5)
        n_freq_per_group = full_psd.groupby(["animal", "region", "condition"]).size().unique()
        assert list(n_freq_per_group) == [65]

        band_summary = result["band_summary"]
        assert set(band_summary.columns) == {"animal", "sex", "region", "condition", "ultra_low", "myogenic"}
        assert len(band_summary) == 24  # 4 sheets x 6 animals
        assert not band_summary[["ultra_low", "myogenic"]].isna().any().any()

        # "24 rows, no NaN" alone would not catch a bug that dropped one
        # animal's group and duplicated another's -- confirm every row is a
        # distinct (animal, region, condition) triple, and that the set of
        # triples exactly matches full_psd's (not just the count).
        assert not band_summary.duplicated(subset=["animal", "region", "condition"]).any()
        full_psd_groups = set(map(tuple, full_psd[["animal", "region", "condition"]].drop_duplicates().values))
        band_summary_groups = set(map(tuple, band_summary[["animal", "region", "condition"]].values))
        assert band_summary_groups == full_psd_groups

        # Same animal-identity check as the loader test (test 1), now against
        # the pipeline's own output -- confirms preprocess -> psd_methods ->
        # bands carried animal/region/condition through three stages without
        # relabelling or crossing any group.
        for condition in ("pre", "TMX"):
            spvb_animals = set(
                band_summary[(band_summary.region == "SpVB") & (band_summary.condition == condition)]["animal"]
            )
            smfp_animals = set(
                band_summary[(band_summary.region == "SmFP") & (band_summary.condition == condition)]["animal"]
            )
            assert spvb_animals == SPVB_ANIMALS
            assert smfp_animals == SMFP_ANIMALS

        row = band_summary[
            (band_summary.animal == 371146) & (band_summary.region == "SpVB") & (band_summary.condition == "pre")
        ]
        assert len(row) == 1
        assert row.iloc[0]["ultra_low"] == pytest.approx(4.929410283325376e-05, rel=1e-4)
        assert row.iloc[0]["myogenic"] == pytest.approx(2.6505155688225456e-05, rel=1e-4)

        assert set(result["figures"]["group_average"].keys()) == {"SmFP", "SpVB"}
        assert set(result["figures"]["per_animal"].keys()) == {"SmFP", "SpVB"}
    finally:
        for fig in result["figures"]["group_average"].values():
            plt.close(fig)
        for fig in result["figures"]["per_animal"].values():
            plt.close(fig)
