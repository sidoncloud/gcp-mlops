#!/bin/bash
# Cart Conversion - Live Online Prediction via curl
# Calls the deployed Vertex AI endpoint with one row of unseen GCS data.
# Authenticates with a short-lived access token from gcloud.

# ---- Configuration ----
PROJECT_ID="YOUR_PROJECT_ID"
REGION="us-central1"
ENDPOINT_ID="YOUR_ENDPOINT_ID"   # Printed by python-sdk-training-deployment.py

# The "instances" array below is one pre-encoded row pulled from
# gs://<your-bucket>/cart-conversion/cart_sessions_new.csv. The column order is:
#
#   session_duration_seconds, pages_viewed, items_in_cart, cart_total_value,
#   unique_categories_viewed, has_discount_code, is_returning_customer,
#   hour_of_day, day_of_week, previous_purchases_count,
#   avg_time_per_page_seconds, added_to_wishlist, used_search,
#   device_type_mobile, device_type_tablet,
#   traffic_source_email, traffic_source_organic, traffic_source_paid
#
# Replace these values with any row from cart_sessions_new.csv.

curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{"instances": [[300, 5, 3, 150.50, 2, 1, 1, 14, 3, 8, 60.10, 0, 1, 1, 0, 0, 1, 0]]}' \
  "https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}:predict"
