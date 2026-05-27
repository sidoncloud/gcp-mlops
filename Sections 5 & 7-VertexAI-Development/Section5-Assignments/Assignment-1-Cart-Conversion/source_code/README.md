# Real-Time Cart Conversion Scoring on Vertex AI

**Lab ID:** `LAB_ECOM_VERTEXAI_001`

End-to-end MLOps on Vertex AI: train a custom scikit-learn classifier on
historical cart sessions, register it in the Vertex AI Model Registry, deploy
it to an online endpoint, and score unseen sessions live with curl.

## Before You Start

Before running any script, export these environment variables in your shell:

```bash
export PROJECT_ID="<your-gcp-project-id>"
export REGION="us-central1"
export BUCKET_NAME="${PROJECT_ID}-cart-conversion-ml"
```

## Run Order

| Step | File                          | What it does                                                            |
|------|-------------------------------|-------------------------------------------------------------------------|
| 1    | `01_enable_apis_and_bucket.sh`| Enables required APIs and creates the GCS bucket                        |
| 2    | `02_upload_data.sh`           | Uploads training + unseen CSVs to GCS                                   |
| 3    | `03_submit_training.py`       | Submits a Vertex AI Custom Training Job (runs `trainer/task.py`)        |
| 4    | `04_register_model.py`        | Registers the trained artifact in Model Registry with serving container |
| 5    | `05_deploy_endpoint.py`       | Creates an endpoint and deploys the model version                       |
| 6    | `06_build_request.py`         | Reads unseen sessions from GCS and builds `request.json`                |
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

1. session_duration_seconds
2. pages_viewed
3. items_in_cart
4. cart_total_value
5. unique_categories_viewed
6. has_discount_code
7. is_returning_customer
8. device_code  (mobile=0, desktop=1, tablet=2)
9. traffic_code (organic=0, paid=1, direct=2, email=3)
10. hour_of_day
11. day_of_week
12. previous_purchases_count
13. avg_time_per_page_seconds
14. added_to_wishlist
15. used_search
