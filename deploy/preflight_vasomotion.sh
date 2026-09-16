#!/usr/bin/env bash
# Pre-deploy gate for the vasomotion PSD tool.
#
# Why this exists: commit e92d983 passed 247/247 tests on my machine while
# being genuinely broken for everyone else. It committed the *callers* of
# preprocess.trace_stats_table but not the definition, and because app.py
# reaches it as an attribute (preprocess.trace_stats_table, not a from-import)
# nothing failed at import time. A clean checkout started fine and raised
# AttributeError the moment a user pressed Run. "My tests pass" and "the
# artifact works" were different claims, and only a clean checkout could
# tell them apart.
#
# So this script never tests the working tree. It tests a detached worktree at
# HEAD -- exactly what Cloud Build will upload -- and then boots the app from a
# replica of the container's file layout.
#
# Usage:
#   ./deploy/preflight_vasomotion.sh      # run the gate alone
#   ./deploy/deploy_vasomotion.sh         # runs this first, refuses to deploy if it fails
#
# Exit codes: 0 = safe to deploy, non-zero = do not deploy. A check that cannot
# run is a failure, never a silent pass -- an earlier version of this used
# `exit 0` when Docker was unavailable, which made a skip look like a success.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="$REPO/venv/bin/python"
PORT="${PREFLIGHT_PORT:-8899}"
WT=""
APP=""
SERVER_PID=""

fail() { echo; echo "PREFLIGHT FAILED: $*"; exit 1; }
step() { echo; echo "==> $*"; }

cleanup() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  if [ -n "$WT" ] && [ -d "$WT" ]; then
    git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true
    rm -rf "$(dirname "$WT")" 2>/dev/null || true
  fi
  [ -n "$APP" ] && rm -rf "$(dirname "$APP")" 2>/dev/null || true
}
trap cleanup EXIT

cd "$REPO"

# ---------------------------------------------------------------------------
step "1/6  the deploy must not depend on files only present locally"
# Cloud Build uploads committed content. Anything uncommitted under
# vasomotion_app/ is, by definition, not going into the image -- which is the
# exact shape of the e92d983 bug.
DIRTY="$(git status --porcelain -- vasomotion_app | grep -v '^?? ' || true)"
if [ -n "$DIRTY" ]; then
  echo "$DIRTY"
  fail "tracked files under vasomotion_app/ have uncommitted changes; commit them or the image will not contain them"
fi
UNTRACKED="$(git status --porcelain -- vasomotion_app | grep '^?? ' || true)"
if [ -n "$UNTRACKED" ]; then
  echo "$UNTRACKED"
  fail "untracked files under vasomotion_app/; commit them or delete them, do not deploy while it is ambiguous"
fi
echo "    clean: HEAD = $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"

# ---------------------------------------------------------------------------
step "2/6  full test suite in a clean detached worktree at HEAD"
[ -x "$PY" ] || fail "no venv interpreter at $PY (bare 'python' on this machine lacks numpy/scipy)"
WT="$(mktemp -d "${TMPDIR:-/tmp}/predeploy.XXXXXX")/wt"
git worktree add --detach "$WT" HEAD >/dev/null 2>&1 \
  || fail "could not create a worktree at HEAD"
find "$WT" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
# Tests that need the (gitignored) lab workbook skip here; that is expected and
# is why the count below is lower than a run with data/Tim/ present.
if ! ( cd "$WT" && PYTHONPATH="$WT" "$PY" -B -m pytest vasomotion_app/tests -q 2>&1 | tail -6 ); then
  fail "test suite failed in a clean checkout of HEAD"
fi

# ---------------------------------------------------------------------------
step "3/6  Dockerfile COPY covers every module"
# A previous Dockerfile listed modules by name and silently omitted stats.py.
# The glob makes that impossible, so assert the glob is still there.
grep -qE '^COPY \*\.py /app/vasomotion_app/' "$WT/vasomotion_app/Dockerfile" \
  || fail "Dockerfile no longer COPYs *.py; a per-file list can silently omit a new module"
echo "    COPY *.py present; $(ls "$WT/vasomotion_app"/*.py | wc -l | tr -d ' ') modules will ship"

# ---------------------------------------------------------------------------
step "4/6  every third-party import is pinned in requirements.txt"
MISSING="$( cd "$WT/vasomotion_app" && "$PY" - <<'PYEOF'
import ast, pathlib, sys
local = {"vasomotion_app","app","bands","bundle","figures","io_excel",
         "preprocess","psd_methods","stats"}
pinned = {l.split("==")[0].strip().lower()
          for l in pathlib.Path("requirements.txt").read_text().splitlines()
          if l.strip() and not l.startswith("#")}
missing = set()
for p in pathlib.Path(".").glob("*.py"):
    for node in ast.walk(ast.parse(p.read_text())):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]] if node.level == 0 else []
        else:
            continue
        for n in names:
            if not n or n in local or n in sys.stdlib_module_names or n in pinned:
                continue
            missing.add(n)
print(",".join(sorted(missing)))
PYEOF
)"
[ -z "$MISSING" ] || fail "imported but not pinned in requirements.txt: $MISSING"
echo "    all third-party imports pinned"

# ---------------------------------------------------------------------------
step "5/6  modules compile under the container's Python version"
BASE_PY_VER="$(grep -oE '^FROM python:[0-9]+\.[0-9]+' "$WT/vasomotion_app/Dockerfile" | grep -oE '[0-9]+\.[0-9]+' || true)"
LOCAL_VER="$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo "    container Python $BASE_PY_VER, local test Python $LOCAL_VER"
CONTAINER_PY="$(command -v "python$BASE_PY_VER" || true)"
if [ -n "$CONTAINER_PY" ]; then
  ( cd "$WT/vasomotion_app" && "$CONTAINER_PY" -m py_compile *.py ) \
    || fail "syntax not valid on Python $BASE_PY_VER (the container's version)"
  find "$WT/vasomotion_app" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
  echo "    all modules compile on $BASE_PY_VER"
elif [ "$BASE_PY_VER" = "$LOCAL_VER" ]; then
  echo "    local interpreter IS $BASE_PY_VER; covered by step 2"
else
  fail "no python$BASE_PY_VER on PATH to syntax-check against the container's version (install it, or this check cannot vouch for the image)"
fi

# ---------------------------------------------------------------------------
step "6/6  boot the app from a replica of the container's file layout"
# Reproduces the Dockerfile: COPY *.py -> /app/vasomotion_app/, PYTHONPATH=/app.
# This is what proves PYTHONPATH is load-bearing and the nested package layout
# actually resolves outside the repo.
APP="$(mktemp -d "${TMPDIR:-/tmp}/containerlayout.XXXXXX")/app"
mkdir -p "$APP/vasomotion_app"
cp "$WT/vasomotion_app"/*.py "$APP/vasomotion_app/"
cp "$WT/vasomotion_app/requirements.txt" "$APP/"

( cd "$APP" && env -u PYTHONPATH PYTHONPATH="$APP" "$PY" -B - <<PYEOF
import importlib, os, sys
root = os.path.realpath("$APP")
bad = []
for m in ["bands","bundle","figures","io_excel","preprocess","psd_methods","stats","app"]:
    f = os.path.realpath(importlib.import_module("vasomotion_app."+m).__file__)
    if not f.startswith(root):
        bad.append((m, f))
if bad:
    sys.exit("resolved from outside the container tree: %r" % (bad,))
PYEOF
) >/dev/null 2>&1 || fail "modules do not import from the container layout (or resolved from the repo instead)"
echo "    all 8 modules import from the container tree, not the repo"

( cd "$APP" && env -u PYTHONPATH PYTHONPATH="$APP" \
    "$PY" -m streamlit run vasomotion_app/app.py \
    --server.port="$PORT" --server.address=127.0.0.1 --server.headless=true \
    >"${TMPDIR:-/tmp}/preflight_streamlit.log" 2>&1 & echo $! >"${TMPDIR:-/tmp}/preflight_pid" )
SERVER_PID="$(cat "${TMPDIR:-/tmp}/preflight_pid")"

READY=""
for _ in $(seq 1 45); do
  sleep 1
  if curl -fsS "http://127.0.0.1:$PORT/_stcore/health" >/dev/null 2>&1; then READY=1; break; fi
done
[ -n "$READY" ] || { tail -20 "${TMPDIR:-/tmp}/preflight_streamlit.log"; fail "app never became healthy in the container layout"; }

CODE="$(curl -s -o "${TMPDIR:-/tmp}/preflight_index.html" -w '%{http_code}' "http://127.0.0.1:$PORT/")"
[ "$CODE" = "200" ] || fail "app root returned HTTP $CODE from the container layout"
grep -q '<title>' "${TMPDIR:-/tmp}/preflight_index.html" \
  || fail "app root returned no HTML title; it is not serving the Streamlit page"
if grep -iqE 'traceback|modulenotfound|attributeerror' "${TMPDIR:-/tmp}/preflight_streamlit.log"; then
  grep -iE -A3 'traceback|modulenotfound|attributeerror' "${TMPDIR:-/tmp}/preflight_streamlit.log" | head -20
  fail "the server log contains an import/attribute error"
fi
echo "    health 200, root 200, no errors in the server log"

echo
echo "==================================================================="
echo "  PREFLIGHT PASSED for $(git rev-parse --short HEAD) -- safe to deploy"
echo "==================================================================="
