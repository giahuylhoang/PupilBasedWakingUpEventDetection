#!/usr/bin/env bash
# Deploy the vasomotion PSD tool to Google Cloud Run.
#
# The spec says: no authentication, no database, no persisted data, and
# longevity is explicitly not a requirement. So this is deliberately the
# smallest possible deployment -- one service, scale to zero, public URL.
#
# Prerequisite (interactive, run it yourself):
#   gcloud auth login
#
# Then:
#   ./deploy/deploy_vasomotion.sh
#
# This refuses to deploy unless ./deploy/preflight_vasomotion.sh passes first
# (clean-checkout tests + a container-layout boot). Run the preflight alone any
# time with:
#   ./deploy/preflight_vasomotion.sh
#
# To tear it down when the credits run out:
#   gcloud run services delete vasomotion-psd --region northamerica-northeast2
set -euo pipefail

PROJECT="rockyridgeai-dental"
REGION="northamerica-northeast2"      # Toronto
SERVICE="vasomotion-psd"
SOURCE="vasomotion_app"               # build context: the Dockerfile's COPY
                                      # paths are relative to this directory

cd "$(dirname "$0")/.."

# Gate the deploy on a clean-checkout verification. This is not ceremony: an
# earlier commit passed the whole suite locally while being broken for everyone
# else, because the working tree held a file the commit did not. The preflight
# tests a detached worktree at HEAD instead, so "my tests pass" and "the
# artifact works" cannot diverge silently again. Set SKIP_PREFLIGHT=1 only if
# you have already run it in this session and nothing has changed since.
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  echo "==> preflight (clean checkout of HEAD)"
  ./deploy/preflight_vasomotion.sh || {
    echo
    echo "Refusing to deploy: preflight failed above. Fix it rather than"
    echo "setting SKIP_PREFLIGHT -- this URL is public."
    exit 1
  }
else
  echo "==> preflight SKIPPED by SKIP_PREFLIGHT=1"
fi

echo "==> project $PROJECT / region $REGION / service $SERVICE"

# Cloud Run source deploys go through Cloud Build and land the image in
# Artifact Registry, so all three APIs are needed. Enabling an already-enabled
# API is a no-op, so this is safe to re-run.
echo "==> enabling APIs (no-op if already enabled)"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --project "$PROJECT"

echo "==> deploying from source"
# --memory 1Gi: numpy + scipy + matplotlib + pandas + streamlit resident is
#   comfortably over the 512Mi default once a workbook is loaded.
# --timeout 3600: Streamlit holds a websocket open for the life of the
#   session; the 300s default would drop users mid-analysis.
# --session-affinity: pins a user to one instance, so their session_state
#   (uploaded file, chosen parameters) survives across requests.
# --min-instances 0: scales to zero, which is the whole cost argument.
gcloud run deploy "$SERVICE" \
  --source "$SOURCE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 3600 \
  --session-affinity \
  --min-instances 0 \
  --max-instances 4 \
  --port 8080

URL=$(gcloud run services describe "$SERVICE" \
        --project "$PROJECT" --region "$REGION" \
        --format='value(status.url)')

echo
echo "==================================================================="
echo "  Live at: $URL"
echo "==================================================================="
echo
echo "Checking it actually serves..."
curl -sS -o /dev/null -w "  /_stcore/health -> HTTP %{http_code}\n" "$URL/_stcore/health" || true
# health alone is not enough: it returns 200 even when the app script itself
# raises on first render, so fetch the page and confirm it is really Streamlit.
BODY=$(mktemp)
curl -sS -o "$BODY" -w "  /               -> HTTP %{http_code}\n" "$URL/" || true
if grep -q '<title>' "$BODY"; then
  echo "  page title      -> $(grep -o '<title>[^<]*</title>' "$BODY" | head -1)"
else
  echo "  page title      -> MISSING; the app may be failing on first render"
fi
rm -f "$BODY"
echo
echo "Smoke-test it yourself before sharing the link: upload a workbook, set the"
echo "sampling rate, and confirm a figure renders. No automated check covers that."
