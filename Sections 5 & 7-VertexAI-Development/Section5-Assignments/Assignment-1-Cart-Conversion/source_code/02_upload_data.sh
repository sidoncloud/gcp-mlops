#!/bin/bash
# 02_upload_data.sh
# Upload the local training CSV and unseen sessions CSV to GCS under /data/ and /new/.

set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
BUCKET_NAME="${BUCKET_NAME:-${PROJECT_ID}-cart-conversion-ml}"

DATASET_DIR="$(cd "$(dirname "$0")/../dataset" && pwd)"

echo "Uploading training data..."
gcloud storage cp "$DATASET_DIR/cart_sessions_train.csv" "gs://$BUCKET_NAME/data/cart_sessions_train.csv"

echo "Uploading unseen data (simulates newly-arrived sessions)..."
gcloud storage cp "$DATASET_DIR/cart_sessions_new.csv" "gs://$BUCKET_NAME/new/cart_sessions_new.csv"

echo "Listing bucket contents:"
gcloud storage ls --recursive "gs://$BUCKET_NAME/"
