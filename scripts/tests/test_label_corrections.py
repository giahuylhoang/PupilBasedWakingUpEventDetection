"""Tests for the pattern-2 correction rule (scripts/propose_label_corrections.py)."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "plc", Path(__file__).resolve().parents[1] / "propose_label_corrections.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def lab(*runs):
    out = []
    for s, n in runs:
        out += [s] * n
    return np.array(out, dtype=object)


def test_pattern2_finds_only_short_nrem_flanked_by_awake():
    l = lab(("Awake", 6), ("NREM", 3), ("Awake", 6), ("NREM", 3), ("REM", 1), ("Awake", 2), ("NREM", 6), ("Awake", 4))
    assert P.pattern2_bouts(l) == [(6, 3)]          # NREM->REM and the 30-s NREM are not pattern 2


def test_pattern1_is_never_a_candidate():
    l = lab(("NREM", 6), ("Awake", 2), ("NREM", 6))  # short wake inside NREM
    assert P.pattern2_bouts(l) == []


def test_decide_requires_pupil_to_stay_above_midpoint():
    assert P.decide(1.0, awake_level=1.0, nrem_level=0.0)        # never left the upper half
    assert P.decide(0.92, awake_level=1.0, nrem_level=0.0)
    assert not P.decide(0.7, awake_level=1.0, nrem_level=0.0)    # real dip below midpoint: keep NREM
    assert not P.decide(1.0, awake_level=0.0, nrem_level=1.0)    # inverted levels: refuse
    assert not P.decide(np.nan, 1.0, 0.0)


def test_typical_levels_use_only_stable_bouts():
    l = lab(("Awake", 6), ("NREM", 2), ("Awake", 1), ("NREM", 6))
    pupil = np.array([1.0] * 6 + [9.0] * 2 + [9.0] + [-1.0] * 6)
    lev = P.typical_levels(l, pupil)
    assert lev["Awake"] == 1.0 and lev["NREM"] == -1.0
