"""Read the wide-format vasomotion Excel workbook into a tidy long dataframe.

Each sheet is wide format: row 1 is 'Animal ID' plus one animal identifier
per column, row 2 is 'Sex' plus M/F per column, and row 3 onward is a sample
index in column A followed by one raw value per animal. Sheet names encode a
region and a condition, e.g. 'SpVB pre' -> region='SpVB', condition='pre'.
"""
from __future__ import annotations

import pandas as pd


def _is_numeric(x) -> bool:
    """True if `x` is a real (non-bool, non-NaN) int or float.

    This is the shape every animal ID (row 1 header), sample index (column
    A), and data value in a well-formed sheet must have. Bench spreadsheets
    routinely tack on a trailing 'Mean' or 'SD' summary row or column;
    whichever of those three positions it lands in, the label fails this
    check the same way a stray text cell does -- it is never silently
    treated as a real sample or a real animal.
    """
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not pd.isna(x)


class NonNumericDataError(Exception):
    """Raised when the workbook contains a non-numeric value in a data cell,
    a non-numeric animal-ID column header, or a non-numeric sample-index
    label in column A.

    A blank/missing cell (read back as NaN) counts as a violation too, the
    same as text or any other non-numeric content -- it is reported rather
    than silently written into the output as `NaN`. A trailing 'Mean' or
    'SD' summary row or column is reported the same way -- it is never
    silently treated as a real sample or averaged in as a real animal.

    `violations` is a list of dicts, one per bad cell, each with keys
    'sheet', 'animal_id', 'row', 'value', 'reason' -- so every offending
    cell can be reported at once instead of failing on the first one found.
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


def check_animal_pairing(long_df: pd.DataFrame) -> dict:
    """Check whether every condition within a region was actually measured
    on the same animals (a repeated-measures / paired design), rather than
    assuming it from counts alone.

    Counting `nunique()` animals per region says nothing about whether
    those are the SAME animals across conditions: a region with
    condition="pre" measured on animals {A, B, C} and condition="TMX" on
    an entirely different {D, E, F} has 6 unique animals and 0 actually
    paired, but `groupby("region")["animal"].nunique()` reports 6 either
    way.

    Returns a dict with keys:
      'n_paired_per_region': {region: int} -- the number of animals common
          to every condition present in that region (the intersection
          across conditions' animal-ID sets), which is the true "paired
          animal(s)" count a repeated-measures figure or statistic relies
          on. A region with only one condition is trivially fully paired
          (that condition's own animal count).
      'warnings': list[str] -- one entry per region where the conditions
          do not all share the exact same animal-ID set, naming the region
          and each condition's animal IDs so the mismatch is inspectable,
          not just flagged.
    """
    warnings: list[str] = []
    n_paired_per_region: dict = {}
    for region, region_df in long_df.groupby("region"):
        condition_animals = {
            condition: set(cond_df["animal"].unique())
            for condition, cond_df in region_df.groupby("condition")
        }
        animal_sets = list(condition_animals.values())
        shared = set.intersection(*animal_sets) if animal_sets else set()
        n_paired_per_region[region] = len(shared)
        if len(animal_sets) > 1 and any(s != animal_sets[0] for s in animal_sets):
            detail = "; ".join(
                f"{condition}={sorted(animals)}" for condition, animals in condition_animals.items()
            )
            warnings.append(
                f"Region {region!r}: conditions do not share the same animal IDs ({detail}). "
                f"Only {len(shared)} animal(s) are common to every condition in this region -- "
                "treat paired comparisons here with caution."
            )
    return {"n_paired_per_region": n_paired_per_region, "warnings": warnings}


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

    Raises NonNumericDataError if any data cell is not numeric -- including a
    blank/missing cell, which pandas reads back as NaN -- listing every
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
            # A trailing 'Mean'/'SD' summary COLUMN has a non-numeric header
            # but perfectly numeric values underneath (they're computed
            # averages) -- so this has to be checked once per column here,
            # not folded into the per-cell `value` check below, or it would
            # sail through as a silent extra "animal".
            animal_id_ok = _is_numeric(animal_id)
            col_values = raw.iloc[2:, col_pos].tolist()
            for sample, value in zip(samples, col_values):
                # Likewise, a trailing 'Mean'/'SD' summary ROW has a
                # non-numeric label in column A but a numeric value in this
                # column -- checked independently of `value` so it can never
                # reach the bare `int(sample)` below and crash.
                sample_ok = _is_numeric(sample)
                value_ok = _is_numeric(value)
                if not (animal_id_ok and sample_ok and value_ok):
                    reasons = []
                    if not animal_id_ok:
                        reasons.append(
                            f"column header {animal_id!r} is not a plausible animal ID"
                        )
                    if not sample_ok:
                        reasons.append(
                            f"row label {sample!r} is not a numeric sample index"
                        )
                    if not value_ok:
                        reasons.append("value is missing or non-numeric")
                    violations.append(
                        {
                            "sheet": sheet_name,
                            "animal_id": animal_id,
                            "row": int(sample) if sample_ok else sample,
                            "value": value,
                            "reason": "; ".join(reasons),
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
