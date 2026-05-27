#!/bin/bash
# Product Return - Live Online Prediction via curl
# Calls the deployed Vertex AI endpoint with one row of unseen GCS data.
# Authenticates with a short-lived access token from gcloud.

# ---- Configuration ----
PROJECT_ID="YOUR_PROJECT_ID"
REGION="us-central1"
ENDPOINT_ID="YOUR_ENDPOINT_ID"   # Printed by python-sdk-training-deployment.py

# The "instances" array below is one pre-encoded row pulled from
# gs://<your-bucket>/order-return/orders_new.csv. The column order is:
#
#   order_total, num_items, item_price, discount_applied_percent, shipping_days,
#   product_avg_rating, customer_past_order_count, customer_past_return_rate,
#   customer_tenure_days, is_first_purchase, used_size_guide, promo_used,
#   weekend_order,
#   product_category_Books, product_category_Clothing, product_category_Electronics,
#   product_category_Home, product_category_Sports, product_category_Toys,
#   payment_method_credit_card, payment_method_debit_card,
#   payment_method_gift_card, payment_method_paypal
#
# Replace these values with any row from orders_new.csv.

curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{"instances": [[249.50, 2, 124.75, 10, 3, 4.2, 12, 0.15, 320, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]]}' \
  "https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}:predict"
