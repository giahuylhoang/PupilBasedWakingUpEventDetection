#!/usr/bin/env bash
# Smoke test for vasomotion_app/Dockerfile: build the image, run it, and
# confirm Streamlit's built-in health endpoint responds. This is
# infrastructure, not a pytest-testable unit, so it is a standalone script
# rather than a test collected by pytest.
#
# Requires Docker Desktop/Engine running locally. Run from anywhere; it
# cd's to vasomotion_app/ itself, so relative paths below are all relative
# to that directory.
set -euo pipefail

# vasomotion_app/tests/test_docker_smoke.sh -> vasomotion_app/. This is also
# the Docker build context, so `docker build . ` below picks up
# vasomotion_app/Dockerfile and the module files it COPYs by default.
cd "$(dirname "$0")/.."

IMAGE="vasomotion-psd-tool-smoketest"
CONTAINER="vasomotion-psd-tool-smoketest-container"
# Host-side port for this smoke test only; arbitrary and unrelated to the
# container's own $PORT (which defaults to 8080 -- see Dockerfile ENV).
# Picked high and specific to avoid colliding with a real dev server.
HOST_PORT=8599

# --- Step 1: build the image from vasomotion_app/Dockerfile. -----------
# Fails loudly (via `set -e`) if requirements.txt fails to resolve or a
# COPY path is missing -- there is nothing running yet to clean up if this
# step fails, so no trap is needed until after `docker run` below.
docker build -t "$IMAGE" .

# --- Step 2: clear out any leftover container from a previous run. -----
# Ignore failure: this is a "make sure it's gone" step, not a check, so a
# missing container (the common case) must not abort the script under
# `set -e`.
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# --- Step 3: run the image, mapping the container's $PORT (default 8080,
# from the Dockerfile) to HOST_PORT on the host. -d backgrounds it so the
# script can go on to poll it below instead of blocking forever.
docker run -d --name "$CONTAINER" -p "${HOST_PORT}:8080" "$IMAGE"

# --- Step 4: guarantee cleanup. ------------------------------------------
# From here on, no matter how the script exits (success, health-check
# timeout, or an unexpected error), stop and remove the container so
# repeated runs of this script never collide with a stale one.
trap 'docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

# --- Step 5: poll Streamlit's health endpoint until it answers. ---------
# `/_stcore/health` is Streamlit's own built-in liveness route (added
# automatically by the server, not something this app defines) -- a bare
# 200 with no body means the process is up and serving. Retry for up to
# 30s to cover slow image-cache-cold container starts; `curl -sf` treats
# any non-2xx status as failure without printing the response body.
for _ in $(seq 1 30); do
  if curl -sf "http://localhost:${HOST_PORT}/_stcore/health" >/dev/null; then
    echo "OK: health check passed"
    exit 0
  fi
  sleep 1
done

# --- Step 6: the loop above only falls through on timeout. ---------------
# Dump the container's logs before exiting non-zero -- with no health
# check having ever passed, stdout/stderr from Streamlit (e.g. a Python
# traceback from a broken import) is the first thing you want to see, and
# the EXIT trap above still removes the container after this prints.
echo "FAIL: health check did not pass within 30s"
docker logs "$CONTAINER" || true
exit 1
