#!/bin/bash
# Bikeshare Model - Custom Container Build & Deploy Commands
# Updated: Uses Artifact Registry instead of Container Registry
#          Uses Python 3.12 base image

# ---- Configuration ----
PROJECT_ID="YOUR_PROJECT_ID"
REGION="us-central1"
REPO_NAME="vertex-ai-models"        # Artifact Registry repository name
IMAGE_NAME="vertex-bikeshare-model"

# Step 0 - Create Artifact Registry repository (one-time setup)
gcloud artifacts repositories create ${REPO_NAME} \
  --repository-format=docker \
  --location=${REGION} \
  --project=${PROJECT_ID} \
  --description="Vertex AI model training containers"

# Configure Docker authentication for Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Step 1 - Build the Docker image
# --platform linux/amd64 is required so the image runs on Vertex AI training VMs (x86_64).
# Without this flag, builds on Apple Silicon (M1/M2/M3) Macs produce ARM64 images
# that fail with "exec format error" on Vertex AI.
docker build --platform linux/amd64 -t ${IMAGE_NAME} .

# Step 2 - Tag the image for Artifact Registry
docker tag ${IMAGE_NAME} ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}

# Step 3 - Push the image to Artifact Registry
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}

# Step 4 - Submit a custom model training job to Vertex AI
gcloud ai custom-jobs create \
  --region=${REGION} \
  --project=${PROJECT_ID} \
  --worker-pool-spec=replica-count=1,machine-type='n1-standard-4',container-image-uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}" \
  --display-name=bike-sharing-model-training
