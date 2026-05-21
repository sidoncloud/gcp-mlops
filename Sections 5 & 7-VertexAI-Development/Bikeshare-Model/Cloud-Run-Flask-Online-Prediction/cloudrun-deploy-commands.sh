#!/bin/bash
# Bikeshare Model - Cloud Run Deployment Commands
# Updated: Uses Artifact Registry, Python 3.12, environment variables

# ---- Configuration ----
PROJECT_ID="YOUR_PROJECT_ID"
PROJECT_NUMBER="YOUR_PROJECT_NUMBER"
REGION="us-central1"
REPO_NAME="vertex-ai-models"
SERVICE_NAME="bikeshare-online-predict"
ENDPOINT_ID="YOUR_ENDPOINT_ID"

# Step 0 - Grant Vertex AI permissions to Cloud Run service account
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/aiplatform.user"

# Step 1 - Create Artifact Registry repo (if not already created)
gcloud artifacts repositories create ${REPO_NAME} \
  --repository-format=docker \
  --location=${REGION} \
  --project=${PROJECT_ID} \
  --description="Vertex AI model serving containers" \
  --quiet 2>/dev/null || true

# Configure Docker for Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Step 2 - Build the Docker image
docker build -t ${SERVICE_NAME} .

# OPTIONAL - Apple Silicon (M1/M2/M3) Macs only:
# Cloud Run runs containers on x86_64. The default build on Apple Silicon produces
# an ARM64 image which fails to start on Cloud Run ("container failed to start
# and listen on the port"). If you are on an Apple Silicon Mac, comment out the
# line above and use the line below instead:
# docker build --platform linux/amd64 -t ${SERVICE_NAME} .

# Step 3 - Tag and push to Artifact Registry
docker tag ${SERVICE_NAME} ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}

# Step 4 - Deploy to Cloud Run with environment variables
gcloud run deploy ${SERVICE_NAME} \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME} \
  --region=${REGION} \
  --project=${PROJECT_ID} \
  --set-env-vars="PROJECT_ID=${PROJECT_ID},REGION=${REGION},ENDPOINT_ID=${ENDPOINT_ID}" \
  --allow-unauthenticated

# Example test command (replace URL with your Cloud Run URL):
# curl -X POST -H "Content-Type: application/json" \
#   -d '{"instance": [0.24, 0.81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]}' \
#   https://bikeshare-online-predict-XXXXX.a.run.app/predict
