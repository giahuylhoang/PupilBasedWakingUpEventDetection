"""Tests for vasomotion_app.io_excel."""
from __future__ import annotations

import math
from pathlib import Path

import openpyxl
import pytest

from vasomotion_app.io_excel import (
    NonNumericDataError,
    detect_sheet_structure,
    load_workbook_long,
    parse_sheet_name,
)

# The real sample workbook this tool is built for. Tests that read it are
# skipped (not failed) in a checkout that doesn't have the data directory.
REAL_WORKBOOK = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "Tim"
    / "Vasomotion analysis compilation_Bal K.xlsx"
)


def _write_sheet(wb, name, animal_ids, sexes, values):
    ws = wb.create_sheet(name)
    ws.append(["Animal ID", *animal_ids])
    ws.append(["Sex", *sexes])
    for i, row in enumerate(values, start=1):
        ws.append([i, *row])


@pytest.fixture
def simple_workbook(tmp_path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    _write_sheet(
        wb, "RegionA condA",
        animal_ids=[101, 102],
        sexes=["M", "F"],
        values=[[10, 20], [11, 21], [12, 22]],
    )
    _write_sheet(
        wb, "RegionA condB",
        animal_ids=[101, 102],
        sexes=["M", "F"],
        values=[[30, 40], [31, 41], [32, 42]],
    )
    path = tmp_path / "simple.xlsx"
    wb.save(path)
    return str(path)


def test_parse_sheet_name_valid():
    assert parse_sheet_name("SpVB pre") == ("SpVB", "pre")
    assert parse_sheet_name("SmFP TMX") == ("SmFP", "TMX")


def test_parse_sheet_name_invalid():
    assert parse_sheet_name("NoSpaceHere") is None
    assert parse_sheet_name("Too Many Words Here") is None


def test_detect_sheet_structure_all_parsed():
    result = detect_sheet_structure(["SpVB pre", "SpVB TMX", "SmFP pre", "SmFP TMX"])
    assert result["all_parsed"] is True
    assert result["regions"] == ["SmFP", "SpVB"]
    assert result["conditions"] == ["TMX", "pre"]
    assert result["parsed"]["SpVB pre"] == ("SpVB", "pre")


def test_detect_sheet_structure_unparseable():
    result = detect_sheet_structure(["SpVB pre", "WeirdSheetName"])
    assert result["all_parsed"] is False
    assert result["parsed"]["WeirdSheetName"] is None
    assert result["parsed"]["SpVB pre"] == ("SpVB", "pre")


def test_load_workbook_long_shape_and_values(simple_workbook):
    df = load_workbook_long(simple_workbook)
    assert list(df.columns) == ["animal", "sex", "region", "condition", "sample", "value"]
    assert len(df) == 2 * 2 * 3  # 2 sheets x 2 animals x 3 samples

    row = df[(df.animal == 101) & (df.condition == "condA") & (df["sample"] == 1)]
    assert len(row) == 1
    assert row.iloc[0]["value"] == 10.0
    assert row.iloc[0]["sex"] == "M"
    assert row.iloc[0]["region"] == "RegionA"

    row2 = df[(df.animal == 102) & (df.condition == "condB") & (df["sample"] == 3)]
    assert len(row2) == 1
    assert row2.iloc[0]["value"] == 42.0


def test_load_workbook_long_unparseable_sheet_raises(tmp_path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    _write_sheet(wb, "NotAValidName", [101], ["M"], [[5], [6]])
    path = tmp_path / "bad.xlsx"
    wb.save(path)
    with pytest.raises(ValueError, match="NotAValidName"):
        load_workbook_long(str(path))


def test_load_workbook_long_sheet_regions_override(tmp_path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    _write_sheet(wb, "NotAValidName", [101], ["M"], [[5], [6]])
    path = tmp_path / "bad.xlsx"
    wb.save(path)
    df = load_workbook_long(str(path), sheet_regions={"NotAValidName": ("RegionX", "condY")})
    assert len(df) == 2
    assert df.iloc[0]["region"] == "RegionX"
    assert df.iloc[0]["condition"] == "condY"


def test_load_workbook_long_non_numeric_cell_raises(tmp_path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    _write_sheet(wb, "RegionA condA", [101, 102], ["M", "F"], [[10, "oops"], [11, 21]])
    path = tmp_path / "nonnumeric.xlsx"
    wb.save(path)
    with pytest.raises(NonNumericDataError) as excinfo:
        load_workbook_long(str(path))
    assert len(excinfo.value.violations) == 1
    v = excinfo.value.violations[0]
    assert v["sheet"] == "RegionA condA"
    assert v["animal_id"] == 102
    assert v["row"] == 1
    assert v["value"] == "oops"


def test_load_workbook_long_blank_cell_raises(tmp_path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    _write_sheet(wb, "RegionA condA", [101, 102], ["M", "F"], [[10, None], [11, 21]])
    path = tmp_path / "blank.xlsx"
    wb.save(path)
    with pytest.raises(NonNumericDataError) as excinfo:
        load_workbook_long(str(path))
    assert len(excinfo.value.violations) == 1
    v = excinfo.value.violations[0]
    assert v["sheet"] == "RegionA condA"
    assert v["animal_id"] == 102
    assert v["row"] == 1
    assert math.isnan(v["value"])


@pytest.mark.skipif(
    not REAL_WORKBOOK.exists(),
    reason="sample workbook not present in this checkout",
)
def test_real_sample_file_sheet_names_and_shape():
    wb = openpyxl.load_workbook(str(REAL_WORKBOOK), read_only=True)
    try:
        sheet_names = wb.sheetnames
    finally:
        wb.close()
    assert sheet_names == ["SpVB pre", "SpVB TMX", "SmFP pre", "SmFP TMX"]

    df = load_workbook_long(str(REAL_WORKBOOK))
    assert list(df.columns) == ["animal", "sex", "region", "condition", "sample", "value"]
    assert df.shape == (28152, 6)
