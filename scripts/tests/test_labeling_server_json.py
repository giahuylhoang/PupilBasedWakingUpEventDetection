"""The /api/cycle payload must be valid JSON for a *browser*, not just Python.

Why this test exists: Python's json module happily writes a bare `NaN`, and
`json.loads` reads it back, so every server-side check passed. A browser's
JSON.parse rejects `NaN` outright -- it is not in the JSON grammar -- so the
whole response failed to parse, openCycle() threw before it ever drew anything,
and the cycle rendered as nothing at all while the request logged a clean
HTTP 200. Fourteen cycles were unopenable that way, and the only visible
symptom was an annotator clicking the same row 68 times.

`allow_nan=False` is exactly the browser's strictness, so that is what we
assert. Run:

    ./venv/bin/python -m pytest scripts/tests/test_labeling_server_json.py -q
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TRACE_DIR = ROOT / "data" / "pupil_whisker_arteriole"
requires_traces = pytest.mark.skipif(
    not TRACE_DIR.is_dir() or not any(TRACE_DIR.glob("*_pupil.csv")),
    reason="local trace data not present (it lives in GCS)",
)


@pytest.fixture(scope="module")
def client():
    # Import with auth disabled and no DATA_BUCKET so it reads local files.
    for var in ("APP_USERNAME", "APP_PASSWORD", "DATA_BUCKET"):
        __import__("os").environ.pop(var, None)
    from scripts import labeling_server as ls

    ls.app.config.update(TESTING=True)
    return ls.app.test_client()


def _non_finite_paths(obj, path=""):
    """Every location holding a value JSON.parse would choke on."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _non_finite_paths(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _non_finite_paths(v, f"{path}[{i}]")
    elif isinstance(obj, float) and not math.isfinite(obj):
        yield path


@requires_traces
def test_every_cycle_payload_is_browser_parseable(client):
    keys = [c["key"] for c in client.get("/api/cycles").get_json()]
    assert keys, "no cycles served"

    offenders = {}
    for key in keys:
        r = client.get(f"/api/cycle/{key}")
        assert r.status_code == 200, f"{key} -> HTTP {r.status_code}"
        # The bug was invisible to get_json(); check the raw bytes the browser sees.
        raw = r.get_data(as_text=True)
        try:
            json.loads(raw, parse_constant=_reject)
        except ValueError as exc:
            offenders[key] = str(exc)
    assert not offenders, "payloads a browser cannot parse: " + json.dumps(
        offenders, indent=2
    )


def _reject(token):  # NaN / Infinity / -Infinity reach here
    raise ValueError(f"bare {token} is not valid JSON")


@requires_traces
def test_no_non_finite_floats_anywhere_in_a_payload(client):
    """Same guarantee stated structurally, so a failure names the field."""
    keys = [c["key"] for c in client.get("/api/cycles").get_json()]
    bad = {}
    for key in keys:
        payload = client.get(f"/api/cycle/{key}").get_json()
        paths = list(_non_finite_paths(payload))
        if paths:
            bad[key] = paths[:5]
    assert not bad, f"non-finite values present: {json.dumps(bad, indent=2)}"


@requires_traces
def test_short_traces_keep_every_real_sample(client):
    """The scrub must turn gaps into null, never drop or invent real data.

    Arteriole and calcium are ~1760 points, under MAX_POINTS, so they bypass
    downsampling and are the traces the original bug leaked through.
    """
    import io

    import pandas as pd

    def read_csv_like_server(path):
        raw = path.read_bytes()
        try:
            return pd.read_csv(io.BytesIO(raw), encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(raw), encoding="latin-1", engine="python")

    keys = [c["key"] for c in client.get("/api/cycles").get_json()]
    checked = 0
    for key in keys:
        payload = client.get(f"/api/cycle/{key}").get_json()
        for suffix in ("arteriole-diameter", "calcium"):
            csv_path = TRACE_DIR / f"{key}_{suffix}.csv"
            trace = next(
                (t for t in payload["traces"] if t["id"] == suffix), None
            )
            if trace is None or not csv_path.exists():
                continue
            col = read_csv_like_server(csv_path).iloc[:, -1]
            expected = int(pd.to_numeric(col, errors="coerce").notna().sum())
            got = sum(1 for v in trace["y"] if v is not None)
            assert got == expected, (
                f"{key}/{suffix}: {got} real samples served, {expected} in the CSV"
            )
            checked += 1
    assert checked > 0, "no short traces were checked"
