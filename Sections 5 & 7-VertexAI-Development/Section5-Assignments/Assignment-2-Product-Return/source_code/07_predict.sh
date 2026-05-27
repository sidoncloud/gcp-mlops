#!/bin/bash
# 07_predict.sh
# Send `request.json` to the deployed Vertex AI endpoint via curl and print the
# prediction response. Uses a short-lived access token from gcloud.

set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION, e.g. export REGION=us-central1}"
: "${ENDPOINT_ID:?Set ENDPOINT_ID (printed by 05_deploy_endpoint.py)}"

REQUEST_FILE="${REQUEST_FILE:-request.json}"

if [ ! -f "$REQUEST_FILE" ]; then
  echo "Error: $REQUEST_FILE not found. Run 06_build_request.py first." >&2
  exit 1
fi

TOKEN=$(gcloud auth print-access-token)
URL="https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}:predict"

echo "POST $URL"
echo

curl -sS -X POST \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  "$URL" \
  -d @"$REQUEST_FILE"

echo
