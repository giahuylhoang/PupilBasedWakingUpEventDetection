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
# To tear it down when the credits run out:
#   gcloud run services delete vasomotion-psd --region northamerica-northeast2
set -euo pipefail

PROJECT="rockyridgeai-dental"
REGION="northamerica-northeast2"      # Toronto
SERVICE="vasomotion-psd"
SOURCE="vasomotion_app"               # build context: the Dockerfile's COPY
                                      # paths are relative to this directory

cd "$(dirname "$0")/.."

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
