"""Read the wide-format vasomotion Excel workbook into a tidy long dataframe.

Each sheet is wide format: row 1 is 'Animal ID' plus one animal identifier
per column, row 2 is 'Sex' plus M/F per column, and row 3 onward is a sample
index in column A followed by one raw value per animal. Sheet names encode a
region and a condition, e.g. 'SpVB pre' -> region='SpVB', condition='pre'.
"""
from __future__ import annotations

import pandas as pd


class NonNumericDataError(Exception):
    """Raised when the workbook contains a non-numeric value in a data cell.

    `violations` is a list of dicts, one per bad cell, each with keys
    'sheet', 'animal_id', 'row', 'value' -- so every offending cell can be
    reported at once instead of failing on the first one found.
    """

    def __init__(self, violations: list[dict]) -> None:
        self.violations = violations
        detail = "; ".join(
            f"sheet={v['sheet']!r} animal_id={v['animal_id']!r} "
            f"row={v['row']} value={v['value']!r}"
            for v in violations
        )
        super().__init__(f"Non-numeric data found in {len(violations)} cell(s): {detail}")


def parse_sheet_name(sheet_name: str) -> tuple[str, str] | None:
    """Split 'SpVB pre' into ('SpVB', 'pre'). Returns None if the sheet name
    is not exactly two whitespace-separated tokens (region, condition)."""
    parts = sheet_name.strip().split()
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def detect_sheet_structure(sheet_names: list[str]) -> dict:
    """Parse region/condition out of every sheet name.

    Returns a dict with keys:
      'parsed': {sheet_name: (region, condition) | None}
      'all_parsed': True if every sheet name parsed successfully
      'regions': sorted list of unique regions found
      'conditions': sorted list of unique conditions found
    """
    parsed = {name: parse_sheet_name(name) for name in sheet_names}
    all_parsed = all(v is not None for v in parsed.values())
    regions = sorted({v[0] for v in parsed.values() if v is not None})
    conditions = sorted({v[1] for v in parsed.values() if v is not None})
    return {
        "parsed": parsed,
        "all_parsed": all_parsed,
        "regions": regions,
        "conditions": conditions,
    }


def load_workbook_long(
    path,
    sheet_regions: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Read every sheet of the workbook at `path` into one tidy long dataframe.

    `path` may be a filesystem path or any file-like object `pandas.read_excel`
    accepts (e.g. the `BytesIO` behind a Streamlit `UploadedFile`).

    Columns: animal, sex, region, condition, sample, value.

    `sheet_regions`, if given, maps sheet name -> (region, condition) and
    overrides automatic parsing for every sheet named in it. Sheets not named
    in `sheet_regions` (or all sheets, if `sheet_regions` is None) fall back
    to `parse_sheet_name`. A ValueError is raised naming any sheet whose
    region/condition could not be determined either way.

    Raises NonNumericDataError if any data cell is not numeric, listing every
    offending cell (sheet, animal_id, row) rather than stopping at the first.
    """
    sheets = pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl")

    overrides = sheet_regions or {}
    unresolved = []
    resolved: dict[str, tuple[str, str]] = {}
    for sheet_name in sheets:
        if sheet_name in overrides:
            resolved[sheet_name] = overrides[sheet_name]
            continue
        parsed = parse_sheet_name(sheet_name)
        if parsed is None:
            unresolved.append(sheet_name)
        else:
            resolved[sheet_name] = parsed
    if unresolved:
        raise ValueError(
            "Could not determine region/condition for sheet(s) "
            f"{unresolved!r} from their names. Pass sheet_regions to override."
        )

    violations: list[dict] = []
    rows = []
    for sheet_name, raw in sheets.items():
        region, condition = resolved[sheet_name]
        animal_ids = raw.iloc[0, 1:].tolist()
        sexes = raw.iloc[1, 1:].tolist()
        samples = raw.iloc[2:, 0].tolist()
        for col_pos, (animal_id, sex) in enumerate(zip(animal_ids, sexes), start=1):
            col_values = raw.iloc[2:, col_pos].tolist()
            for sample, value in zip(samples, col_values):
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    violations.append(
                        {
                            "sheet": sheet_name,
                            "animal_id": animal_id,
                            "row": int(sample) if isinstance(sample, (int, float)) else sample,
                            "value": value,
                        }
                    )
                    continue
                rows.append(
                    {
                        "animal": animal_id,
                        "sex": sex,
                        "region": region,
                        "condition": condition,
                        "sample": int(sample),
                        "value": float(value),
                    }
                )

    if violations:
        raise NonNumericDataError(violations)

    return pd.DataFrame(rows, columns=["animal", "sex", "region", "condition", "sample", "value"])
