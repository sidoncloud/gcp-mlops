#!/usr/bin/env bash
#
# Deploy the Support Ticket Triage API to Cloud Run.
# Run these commands one at a time, or execute the whole script with: bash deploy-commands.sh
set -euo pipefail

# ----------------------------------------------------------------------------
# Configuration - edit these for your project.
# ----------------------------------------------------------------------------
PROJECT_ID="your-project-id"
REGION="us-central1"
SERVICE_NAME="ticket-classification-app"
REPO="gemini-labs"                 # Artifact Registry repository name
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE_NAME}:latest"

gcloud config set project "${PROJECT_ID}"

# ----------------------------------------------------------------------------
# Grant the Cloud Run service account permission to call Vertex AI.
# Cloud Run runs as the Compute Engine default service account unless told
# otherwise. roles/aiplatform.user lets it invoke Gemini.
# ----------------------------------------------------------------------------
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/aiplatform.user"

# ----------------------------------------------------------------------------
# Enable APIs and create the Artifact Registry repo (idempotent).
# ----------------------------------------------------------------------------
gcloud services enable run.googleapis.com aiplatform.googleapis.com artifactregistry.googleapis.com

gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Gemini Cloud Run labs" || true

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ----------------------------------------------------------------------------
# Build and push the image.
# --platform linux/amd64 forces an x86 build. This matters on Apple Silicon
# (M-series) Macs, whose native arm64 images will not run on Cloud Run's x86.
# ----------------------------------------------------------------------------
docker build --platform linux/amd64 -t "${IMAGE}" .
docker push "${IMAGE}"

# ----------------------------------------------------------------------------
# Deploy to Cloud Run.
# ----------------------------------------------------------------------------
gcloud run deploy "${SERVICE_NAME}" \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --platform=managed \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION}"

# ----------------------------------------------------------------------------
# Test it (uncomment and run after deploy).
# ----------------------------------------------------------------------------
# SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region="${REGION}" --format="value(status.url)")
#
# curl -X POST "${SERVICE_URL}/classify" \
#   -H "Content-Type: application/json" \
#   -d '{"message": "I was charged twice for my subscription this month and need a refund."}'
