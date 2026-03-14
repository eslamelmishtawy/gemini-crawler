#!/usr/bin/env bash
# deploy.sh — Automated Cloud Run deployment for UI Navigator
#
# Usage:
#   ./deploy.sh                          # Deploy with defaults
#   ./deploy.sh --project my-gcp-project # Override project ID
#   ./deploy.sh --region europe-west1    # Override region
#
# Prerequisites:
#   - gcloud CLI authenticated (gcloud auth login)
#   - Docker or Cloud Build enabled
#   - Required APIs: Cloud Run, Artifact Registry, Firestore, Cloud Storage

set -euo pipefail

# ---- Defaults (override via flags or env vars) ----
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${CLOUD_RUN_REGION:-us-central1}"
SERVICE_NAME="ui-navigator"
MEMORY="2Gi"
TIMEOUT="300"

# ---- Parse flags ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --project) PROJECT_ID="$2"; shift 2 ;;
    --region)  REGION="$2";     shift 2 ;;
    --memory)  MEMORY="$2";     shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

if [[ -z "$PROJECT_ID" ]]; then
  echo "ERROR: No GCP project ID. Set GCP_PROJECT_ID or use --project flag."
  exit 1
fi

echo "=== UI Navigator — Cloud Run Deployment ==="
echo "  Project : $PROJECT_ID"
echo "  Region  : $REGION"
echo "  Service : $SERVICE_NAME"
echo "  Memory  : $MEMORY"
echo ""

# ---- Enable required APIs ----
echo "[1/4] Enabling required GCP APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  firestore.googleapis.com \
  storage.googleapis.com \
  --project "$PROJECT_ID" --quiet

# ---- Install Playwright browsers locally (for local dev) ----
echo "[2/4] Checking Playwright browsers..."
if command -v playwright &>/dev/null; then
  playwright install chromium 2>/dev/null || true
fi

# ---- Deploy to Cloud Run from source ----
echo "[3/4] Deploying to Cloud Run (building from source)..."
gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --memory "$MEMORY" \
  --timeout "$TIMEOUT" \
  --allow-unauthenticated \
  --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID" \
  --quiet

# ---- Verify deployment ----
echo "[4/4] Verifying deployment..."
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format "value(status.url)")

echo ""
echo "=== Deployment Complete ==="
echo "  Service URL: $SERVICE_URL"
echo ""

# Health check
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL/health")
if [[ "$HTTP_STATUS" == "200" ]]; then
  echo "  Health check: PASSED"
else
  echo "  Health check: FAILED (HTTP $HTTP_STATUS)"
  exit 1
fi
