"""Browser-based Awake/NREM/REM labeling tool (Turner 2023 style).

Serves a single-page app that lets a colleague view all traces for one cycle
(pupil diameter, eye-motion speed, whisker, arteriole diameter, calcium),
navigate 5-second segments, and label each segment Awake / NREM / REM / Unclear.

Why a backend (not a static HTML file): the browser cannot list a server
directory or write files to disk. This tiny Flask app does three things:
  * /api/cycles            -> list every cycle (recording) available to label
  * /api/cycle/<key>       -> traces (downsampled) + 5 s bins for one cycle
  * /api/labels (GET/POST) -> load / save a label CSV per annotator+cycle

Data sources (read-only):
  data/pupil_whisker_arteriole/<key>_{pupil,whisker-gradient,
      arteriole-diameter,calcium}.csv          (the 5 raw traces)
  data/pupil_eye_motion/<key>_eyemotion_perframe.csv   (diameter + eye_speed)
  data/pupil_eye_motion/<key>_eyemotion_5s_features.csv (5 s bin grid)

Labels are written to:
  data/labels/<annotator>/<key>_labels.csv
      columns: key, bin, t_start, t_end, label, annotator, ts

Run:
  .venv/bin/python scripts/labeling_server.py            # default port 5050
  PORT=8000 .venv/bin/python scripts/labeling_server.py  # custom port
  open http://127.0.0.1:5050

Note: port 5000 is avoided by default because macOS AirPlay Receiver
(ControlCenter / "AirTunes") occupies it and returns 403 to all requests.
"""
from __future__ import annotations

import csv
import datetime as _dt
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "data" / "pupil_whisker_arteriole"
EYE_DIR = ROOT / "data" / "pupil_eye_motion"
LABEL_DIR = ROOT / "data" / "labels"
LABEL_DIR.mkdir(parents=True, exist_ok=True)

BIN_SEC = 5.0          # Turner 5 s classification bins
MAX_POINTS = 4000      # max points sent per trace (downsample for the browser)

# Modalities found in pupil_whisker_arteriole/, by filename suffix.
# label = what the annotator sees; each file carries its own `time` column,
# so traces with different sample rates stay correctly aligned on the x-axis.
TRACE_SUFFIXES = [
    ("pupil", "Pupil diameter"),
    ("whisker-gradient", "Whisker (gradient)"),
    ("arteriole-diameter", "Arteriole diameter"),
    ("calcium", "Calcium"),
    ("ne", "NE (norepinephrine)"),
]

KEY_RE = re.compile(r"^(.*)_pupil\.csv$")

app = Flask(__name__)

# --------------------------------------------------------------------------- #
# Authentication (login channel; credentials from environment variables)
# --------------------------------------------------------------------------- #
# Set these at deploy time. If both are unset, auth is DISABLED (local dev).
#   APP_USERNAME   - the login username
#   APP_PASSWORD   - the login password
#   APP_SECRET_KEY - Flask session signing key (random string; auto-generated
#                    if unset, but then sessions reset on every restart)
import hmac
import secrets

AUTH_USER = os.environ.get("APP_USERNAME", "")
AUTH_PASS = os.environ.get("APP_PASSWORD", "")
AUTH_ENABLED = bool(AUTH_USER and AUTH_PASS)
app.secret_key = os.environ.get("APP_SECRET_KEY") or secrets.token_hex(32)
# Harden the session cookie.
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    # Secure cookie only over HTTPS; Cloud Run terminates TLS, set via env.
    SESSION_COOKIE_SECURE=os.environ.get("APP_COOKIE_SECURE", "0") == "1",
)

# Imported here so the auth code above can be defined before use.
from flask import redirect, session, url_for  # noqa: E402

LOGIN_HTML = """<!doctype html><html lang=en><head><meta charset=utf-8>
<title>Sign in — Sleep labeling</title>
<style>
 body{margin:0;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
   background:#0f1115;color:#e8eaed;display:grid;place-items:center;height:100vh}
 form{background:#181b22;border:1px solid #2a2f3a;border-radius:12px;
   padding:28px 26px;width:300px}
 h1{font-size:16px;margin:0 0 16px}
 label{display:block;font-size:12px;color:#9aa0aa;margin:12px 0 4px}
 input{width:100%;box-sizing:border-box;padding:9px;border-radius:7px;
   border:1px solid #2a2f3a;background:#222732;color:#e8eaed;font:inherit}
 button{margin-top:18px;width:100%;padding:10px;border:0;border-radius:7px;
   background:#4da3ff;color:#001;font-weight:600;font:inherit;cursor:pointer}
 .err{color:#e0607e;font-size:13px;margin-top:12px;min-height:16px}
</style></head><body>
<form method=post>
 <h1>Sleep-state labeling</h1>
 <label>Username</label><input name=username autofocus autocomplete=username>
 <label>Password</label><input name=password type=password autocomplete=current-password>
 <button>Sign in</button>
 <div class=err>%ERR%</div>
</form></body></html>"""


@app.route("/login", methods=["GET", "POST"])
def login():
    if not AUTH_ENABLED:
        return redirect(url_for("index"))
    err = ""
    if request.method == "POST":
        u = request.form.get("username", "")
        p = request.form.get("password", "")
        # constant-time comparison to avoid timing leaks
        ok = hmac.compare_digest(u, AUTH_USER) and hmac.compare_digest(p, AUTH_PASS)
        if ok:
            session["auth"] = True
            session.permanent = True
            return redirect(url_for("index"))
        err = "Incorrect username or password."
    return Response(LOGIN_HTML.replace("%ERR%", err), mimetype="text/html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.before_request
def _require_login():
    if not AUTH_ENABLED:
        return None  # auth disabled (local dev)
    if request.endpoint in ("login", "logout"):
        return None
    if session.get("auth"):
        return None
    # Unauthenticated: API calls get 401 JSON; pages get the login screen.
    if request.path.startswith("/api/"):
        return jsonify({"error": "auth required"}), 401
    return redirect(url_for("login"))


# --------------------------------------------------------------------------- #
# Storage layer: local filesystem (dev) or Google Cloud Storage (production).
# Selected by the DATA_BUCKET env var. All trace/label I/O goes through here so
# the rest of the app is storage-agnostic. Logical paths are POSIX-style and
# relative to the data/ root, e.g. "pupil_whisker_arteriole/<key>_pupil.csv"
# and "labels/<annotator>/<key>_labels.csv".
# --------------------------------------------------------------------------- #
DATA_BUCKET = os.environ.get("DATA_BUCKET", "")          # e.g. "gordonlab-labeling-data"
DATA_PREFIX = os.environ.get("DATA_PREFIX", "data")      # object-name prefix in the bucket
USE_GCS = bool(DATA_BUCKET)

_gcs_bucket = None
if USE_GCS:
    from google.cloud import storage as _gcs  # lazy import; only needed in prod
    _gcs_bucket = _gcs.Client().bucket(DATA_BUCKET)


def _blob(relpath: str):
    return _gcs_bucket.blob(f"{DATA_PREFIX}/{relpath}")


def store_exists(relpath: str) -> bool:
    if USE_GCS:
        return _blob(relpath).exists()
    return (ROOT / "data" / relpath).exists()


def store_read_bytes(relpath: str) -> bytes | None:
    """Return raw bytes of a stored file, or None if absent."""
    if USE_GCS:
        b = _blob(relpath)
        if not b.exists():
            return None
        return b.download_as_bytes()
    p = ROOT / "data" / relpath
    if not p.exists():
        return None
    return p.read_bytes()


def store_write_text(relpath: str, text: str) -> None:
    if USE_GCS:
        _blob(relpath).upload_from_string(text, content_type="text/csv")
    else:
        p = ROOT / "data" / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def store_list_pupil_keys() -> list[str]:
    """All recording keys that have a pupil trace."""
    keys = []
    if USE_GCS:
        prefix = f"{DATA_PREFIX}/pupil_whisker_arteriole/"
        for b in _gcs_bucket.list_blobs(prefix=prefix):
            name = b.name[len(prefix):]
            m = KEY_RE.match(name)
            if m:
                keys.append(m.group(1))
    else:
        for f in sorted((ROOT / "data" / "pupil_whisker_arteriole").glob("*_pupil.csv")):
            m = KEY_RE.match(f.name)
            if m:
                keys.append(m.group(1))
    return sorted(keys)


def _read_csv_df(relpath: str):
    """Read a stored CSV into a DataFrame (utf-8, latin-1 fallback), or None."""
    raw = store_read_bytes(relpath)
    if raw is None:
        return None
    import io as _io
    try:
        return pd.read_csv(_io.BytesIO(raw), encoding="utf-8")
    except (UnicodeDecodeError, pd.errors.ParserError):
        return pd.read_csv(_io.BytesIO(raw), encoding="latin-1", engine="python")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def list_keys() -> list[str]:
    """Every recording key that has at least a pupil trace."""
    return store_list_pupil_keys()


def _list_keys_local_unused() -> list[str]:
    keys = []
    for f in sorted(TRACE_DIR.glob("*_pupil.csv")):
        m = KEY_RE.match(f.name)
        if m:
            keys.append(m.group(1))
    return keys


def parse_key(key: str) -> dict:
    """Split <Cohort>_<Group>_<date>_cycle<N> into parts (best-effort)."""
    cohort = group = date = cycle = ""
    mcyc = re.search(r"_cycle_?(\d+)$", key)
    if mcyc:
        cycle = mcyc.group(1)
    mdate = re.search(r"(\d{4}\.\d{2}\.\d{2})", key)
    if mdate:
        date = mdate.group(1)
    # everything before the date is cohort/group
    head = key.split("_" + date)[0] if date else key
    parts = head.split("_")
    if parts:
        cohort = parts[0]
        if len(parts) > 1:
            group = "_".join(parts[1:])
    return {"cohort": cohort, "group": group, "date": date, "cycle": cycle}


def _eye_key(key: str) -> str:
    """pupil_whisker_arteriole keys are 'Extra_GCaMP8m_<date>_cycleN';
    eye-motion keys drop the leading 'Extra_' and use 'GCaMP8m_<date>_cycleN'.
    Map between them when looking for the eye-motion file."""
    if store_exists(f"pupil_eye_motion/{key}_eyemotion_perframe.csv"):
        return key
    stripped = re.sub(r"^Extra_", "", key)
    if store_exists(f"pupil_eye_motion/{stripped}_eyemotion_perframe.csv"):
        return stripped
    return key


def _finite_points(ts: list, ys: list) -> tuple[list, list]:
    """Make a trace safe to serialise as JSON.

    A non-finite VALUE becomes None: the frontend already treats null as "lift
    the pen", so the trace draws with a gap instead of a fake datapoint.
    A non-finite TIME drops the point entirely -- it cannot be placed on the
    x-axis, and the frontend only null-checks y.
    """
    out_t, out_y = [], []
    for tv, yv in zip(ts, ys):
        if tv is None or not math.isfinite(tv):
            continue
        out_t.append(float(tv))
        out_y.append(None if yv is None or not math.isfinite(yv) else float(yv))
    return out_t, out_y


def downsample(t: np.ndarray, y: np.ndarray, max_points: int = MAX_POINTS):
    """Reduce a trace to <= max_points using block min/max so spikes survive.

    For each block we emit two points (min then max at their real times),
    preserving the visual envelope far better than naive striding.
    """
    n = len(t)
    if n <= max_points:
        # Short traces (arteriole ~1761 points, calcium ~1760) are sent as-is.
        # They must still be scrubbed: Python's json writes a bare NaN, which is
        # NOT valid JSON. The browser's JSON.parse rejects the whole response,
        # so openCycle() throws and the cycle renders as nothing at all -- while
        # the request still looks like a clean HTTP 200 in the logs. That is
        # exactly how four arteriole traces (one empty trailing CSV cell each)
        # made five cycles silently unopenable.
        return _finite_points(t.tolist(), y.tolist())
    n_blocks = max(1, max_points // 2)
    idx = np.array_split(np.arange(n), n_blocks)
    out_t, out_y = [], []
    for blk in idx:
        if blk.size == 0:
            continue
        yb = y[blk]
        finite = np.isfinite(yb)
        if not finite.any():
            j = blk[0]
            out_t.append(float(t[j])); out_y.append(None)
            continue
        bf = blk[finite]; yf = yb[finite]
        jmin = bf[int(np.argmin(yf))]
        jmax = bf[int(np.argmax(yf))]
        for j in sorted((jmin, jmax)):
            out_t.append(float(t[j]))
            out_y.append(float(y[j]))
    # Long traces rarely leak a NaN (an all-NaN block already becomes None),
    # but a non-finite timestamp would still slip through, so scrub both paths.
    return _finite_points(out_t, out_y)


def load_trace(relpath: str):
    """Read a time/value csv (by storage relpath) -> (time, value, needs_rescale).

    Two shapes are supported:
      * 2 columns (time, value): timestamps are real -> needs_rescale=False.
      * 1 column (value only, e.g. the NE whisker-gradient files, which are
        headerless and carry no time column): a synthetic index-based time
        axis is returned and needs_rescale=True, so the caller can stretch it
        linearly across the cycle duration to align with the other traces.

    Handles BOM, unnamed value columns, and non-UTF-8 bytes.
    Returns None if the file is absent or has no usable numeric column.
    """
    df = _read_csv_df(relpath)
    if df is None:
        return None
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    tcol = df.columns[0]
    vcol = df.columns[1] if len(df.columns) > 1 else None
    if vcol is None:
        # Single-column, values-only file: no timestamps available.
        # NOTE: pandas already consumed the first data row as the header when
        # the file is headerless, so re-read without a header to keep all rows.
        raw = store_read_bytes(relpath)
        if raw is None:
            return None
        import io as _io
        try:
            df1 = pd.read_csv(_io.BytesIO(raw), header=None, encoding="utf-8")
        except (UnicodeDecodeError, pd.errors.ParserError):
            df1 = pd.read_csv(_io.BytesIO(raw), header=None,
                              encoding="latin-1", engine="python")
        y = pd.to_numeric(df1.iloc[:, 0], errors="coerce").to_numpy()
        ok = np.isfinite(y)
        y = y[ok]
        t = np.arange(len(y), dtype=float)  # index-based; rescaled by caller
        return t, y, True
    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy()
    y = pd.to_numeric(df[vcol], errors="coerce").to_numpy()
    ok = np.isfinite(t)
    return t[ok], y[ok], False


def _label_relpath(annotator: str, key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", annotator or "anon")
    return f"labels/{safe}/{key}_labels.csv"


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.route("/api/cycles")
def api_cycles():
    annotator = request.args.get("annotator", "anon")
    out = []
    for key in list_keys():
        info = parse_key(key)
        ldf = _read_csv_df(_label_relpath(annotator, key))
        n_done = 0
        if ldf is not None:
            try:
                n_done = int((ldf["label"].astype(str) != "").sum())
            except Exception:
                n_done = 0
        info.update({"key": key, "labeled": n_done})
        out.append(info)
    return jsonify(out)


@app.route("/api/cycle/<key>")
def api_cycle(key: str):
    if key not in list_keys():
        return jsonify({"error": "unknown key"}), 404

    traces = []
    # 1) eye-motion: diameter (z) and eye_speed come from the per-frame file
    ekey = _eye_key(key)
    edf = _read_csv_df(f"pupil_eye_motion/{ekey}_eyemotion_perframe.csv")
    duration = 0.0
    if edf is not None:
        et = pd.to_numeric(edf["time"], errors="coerce").to_numpy()
        duration = max(duration, float(np.nanmax(et)) if et.size else 0.0)
        for col, lab in [("diameter_z", "Pupil diameter (z)"),
                         ("eye_speed", "Eye motion (speed)")]:
            if col in edf.columns:
                y = pd.to_numeric(edf[col], errors="coerce").to_numpy()
                dt, dy = downsample(et, y)
                traces.append({"id": col, "label": lab, "t": dt, "y": dy})

    # 2) raw traces from pupil_whisker_arteriole (each with own sample rate)
    #    Single-column files (needs_rescale=True) have no timestamps; defer them
    #    until `duration` is known, then stretch linearly across the cycle.
    pending_rescale = []
    for suf, lab in TRACE_SUFFIXES:
        tr = load_trace(f"pupil_whisker_arteriole/{key}_{suf}.csv")
        if tr is None:
            continue
        t, y, needs_rescale = tr
        if needs_rescale:
            pending_rescale.append((suf, lab, t, y))
            continue
        if t.size:
            duration = max(duration, float(np.nanmax(t)))
        dt, dy = downsample(t, y)
        traces.append({"id": suf, "label": lab, "t": dt, "y": dy})

    # Now stretch any single-column traces (e.g. NE whisker) across the full
    # cycle duration so they align on the same x-axis as the other traces.
    for suf, lab, t, y in pending_rescale:
        if y.size and duration > 0:
            n = y.size
            t = np.linspace(0.0, duration, n) if n > 1 else np.array([0.0])
        dt, dy = downsample(t, y)
        traces.append({"id": suf, "label": lab, "t": dt, "y": dy})

    # 5 s bin grid (prefer the eye-motion feature file; else synthesize)
    bins = []
    fdf = _read_csv_df(f"pupil_eye_motion/{ekey}_eyemotion_5s_features.csv")
    if fdf is not None:
        for _, r in fdf.iterrows():
            ts = float(r["t_start"])
            bins.append({"bin": int(r["bin"]), "t_start": ts,
                         "t_end": ts + BIN_SEC,
                         "motion_sum": float(r.get("motion_sum", np.nan)),
                         "diam_z_mean": float(r.get("diam_z_mean", np.nan))})
    if not bins and duration > 0:
        n = int(np.ceil(duration / BIN_SEC))
        for i in range(n):
            bins.append({"bin": i, "t_start": i * BIN_SEC,
                         "t_end": (i + 1) * BIN_SEC,
                         "motion_sum": None, "diam_z_mean": None})

    return jsonify({
        "key": key, "info": parse_key(key),
        "duration": duration, "bin_sec": BIN_SEC,
        "traces": traces, "bins": bins,
    })


@app.route("/api/labels", methods=["GET", "POST"])
def api_labels():
    if request.method == "GET":
        annotator = request.args.get("annotator", "anon")
        key = request.args.get("key", "")
        raw = store_read_bytes(_label_relpath(annotator, key))
        labels = {}
        if raw is not None:
            import io as _io
            for row in csv.DictReader(_io.StringIO(raw.decode("utf-8", "replace"))):
                try:
                    labels[int(row["bin"])] = row["label"]
                except (KeyError, ValueError):
                    continue
        return jsonify({"labels": labels})

    data = request.get_json(force=True)
    annotator = data.get("annotator", "anon")
    key = data.get("key", "")
    labels: dict = data.get("labels", {})  # {bin: label}
    bins_meta: dict = data.get("bins", {})  # {bin: {t_start,t_end}}
    if key not in list_keys():
        return jsonify({"error": "unknown key"}), 400
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    import io as _io
    buf = _io.StringIO()
    w = csv.writer(buf)
    w.writerow(["key", "bin", "t_start", "t_end", "label", "annotator", "ts"])
    for b in sorted(labels, key=lambda x: int(x)):
        meta = bins_meta.get(str(b), bins_meta.get(b, {}))
        t0 = meta.get("t_start", float(b) * BIN_SEC)
        t1 = meta.get("t_end", (float(b) + 1) * BIN_SEC)
        w.writerow([key, b, t0, t1, labels[b], annotator, ts])
    relpath = _label_relpath(annotator, key)
    store_write_text(relpath, buf.getvalue())
    return jsonify({"ok": True, "saved": len(labels), "path": relpath})


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


# INDEX_HTML is injected from the frontend file at import time (built next task).
INDEX_HTML = "<h1>Frontend not built yet</h1>"
_front = Path(__file__).with_name("labeling_frontend.html")
if _front.exists():
    INDEX_HTML = _front.read_text()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Serving {len(list_keys())} cycles from {TRACE_DIR}")
    print(f"Auth: {'ENABLED (login required)' if AUTH_ENABLED else 'DISABLED (no APP_USERNAME/APP_PASSWORD set)'}")
    print(f"Open http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
