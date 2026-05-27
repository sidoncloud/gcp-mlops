#!/bin/bash
# 02_upload_data.sh
# Upload the local training CSV and unseen orders CSV to GCS under /data/ and /new/.

set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
BUCKET_NAME="${BUCKET_NAME:-${PROJECT_ID}-order-return-ml}"

DATASET_DIR="$(cd "$(dirname "$0")/../dataset" && pwd)"

echo "Uploading training data..."
gcloud storage cp "$DATASET_DIR/orders_train.csv" "gs://$BUCKET_NAME/data/orders_train.csv"

echo "Uploading unseen data (simulates newly-placed orders)..."
gcloud storage cp "$DATASET_DIR/orders_new.csv" "gs://$BUCKET_NAME/new/orders_new.csv"

echo "Listing bucket contents:"
gcloud storage ls --recursive "gs://$BUCKET_NAME/"
