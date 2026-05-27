#!/bin/bash
# 01_enable_apis_and_bucket.sh
# Enable required GCP APIs and create the GCS bucket that holds training data,
# unseen data, and model artifacts for this lab.

set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID, e.g. export PROJECT_ID=my-gcp-project}"
REGION="${REGION:-us-central1}"
BUCKET_NAME="${BUCKET_NAME:-${PROJECT_ID}-order-return-ml}"

echo "Project:  $PROJECT_ID"
echo "Region:   $REGION"
echo "Bucket:   gs://$BUCKET_NAME"

echo "Enabling APIs..."
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  compute.googleapis.com \
  notebooks.googleapis.com \
  --project "$PROJECT_ID"

echo "Creating bucket..."
if gcloud storage buckets describe "gs://$BUCKET_NAME" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "Bucket gs://$BUCKET_NAME already exists, skipping create."
else
  gcloud storage buckets create "gs://$BUCKET_NAME" \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --uniform-bucket-level-access
fi

echo "Done."
