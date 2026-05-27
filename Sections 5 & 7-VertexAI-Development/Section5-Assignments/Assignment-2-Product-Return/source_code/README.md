# Predicting Product Returns for Online Retail on Vertex AI

**Lab ID:** `LAB_ECOM_VERTEXAI_002`

End-to-end MLOps on Vertex AI: train a custom scikit-learn classifier on
historical orders, register it in the Vertex AI Model Registry, deploy it to
an online endpoint, and score unseen orders live with curl.

## Before You Start

Before running any script, export these environment variables in your shell:

```bash
export PROJECT_ID="<your-gcp-project-id>"
export REGION="us-central1"
export BUCKET_NAME="${PROJECT_ID}-order-return-ml"
```

## Run Order

| Step | File                          | What it does                                                            |
|------|-------------------------------|-------------------------------------------------------------------------|
| 1    | `01_enable_apis_and_bucket.sh`| Enables required APIs and creates the GCS bucket                        |
| 2    | `02_upload_data.sh`           | Uploads training + unseen CSVs to GCS                                   |
| 3    | `03_submit_training.py`       | Submits a Vertex AI Custom Training Job (runs `trainer/task.py`)        |
| 4    | `04_register_model.py`        | Registers the trained artifact in Model Registry with serving container |
| 5    | `05_deploy_endpoint.py`       | Creates an endpoint and deploys the model version                       |
| 6    | `06_build_request.py`         | Reads unseen orders from GCS and builds `request.json`                  |
| 7    | `07_predict.sh`               | curl-POSTs the request to the live endpoint, prints predictions         |
| 99   | `99_cleanup.py`               | Deletes endpoint, model, and (optionally) the bucket                    |

## Tech Stack

- **Vertex AI Workbench**: managed JupyterLab for data exploration
- **Vertex AI Custom Training**: runs `trainer/task.py` in a pre-built sklearn container
- **Vertex AI Model Registry**: stores the versioned model artifact with its serving image
- **Vertex AI Endpoints**: managed online serving with autoscaling
- **scikit-learn 1.6**: RandomForestClassifier wrapped in a Pipeline
- **Google Cloud Storage**: stores training data, unseen data, and model artifacts
- **gcloud CLI**: used to mint short-lived access tokens for curl

## Feature Order

The pipeline expects inputs in exactly this order (see `trainer/task.py` and
`06_build_request.py`):

1. order_total
2. num_items
3. item_price
4. discount_applied_percent
5. shipping_days
6. category_code (Clothing=0, Electronics=1, Home=2, Sports=3, Books=4, Beauty=5, Toys=6)
7. product_avg_rating
8. customer_past_order_count
9. customer_past_return_rate
10. customer_tenure_days
11. payment_code (credit_card=0, paypal=1, debit_card=2, apple_pay=3, gift_card=4)
12. is_first_purchase
13. used_size_guide
14. promo_used
15. weekend_order
