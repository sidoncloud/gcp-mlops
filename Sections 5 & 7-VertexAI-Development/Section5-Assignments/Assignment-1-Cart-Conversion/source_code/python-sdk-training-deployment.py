"""
Cart Conversion - Vertex AI Python SDK: Train, Upload, Deploy, Predict
End-to-end orchestrator: submit the Vertex AI Custom Training Job, upload the
trained model to the Model Registry, deploy it to an online endpoint, and score
unseen cart sessions read from GCS.

Updated: google-cloud-aiplatform>=1.60.0, Python 3.12
"""

import pandas as pd
from google.cloud import aiplatform

# ---- Configuration ----
PROJECT_ID = "YOUR_PROJECT_ID"          # Replace with your GCP project ID
REGION = "us-central1"
BUCKET_NAME = "YOUR_BUCKET_NAME"        # Replace with your GCS bucket name
STAGING_BUCKET = f"gs://{BUCKET_NAME}"

# =====================================================
# Step 1: Initialize Vertex AI SDK
# =====================================================
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

# =====================================================
# Step 2: Submit Custom Training Job
# Uses a prebuilt scikit-learn training container from Artifact Registry.
# =====================================================
job = aiplatform.CustomTrainingJob(
    display_name="cart-conversion-training-job",
    script_path="model-training-code.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-5:latest",
    requirements=["gcsfs"],
)

job.run(
    replica_count=1,
    machine_type="n1-standard-4",
    sync=True,
)
job.wait()

# =====================================================
# Step 3: Upload Model to Vertex AI Model Registry
# =====================================================
display_name = "cart-conversion-classifier"
artifact_uri = f"gs://{BUCKET_NAME}/cart-conversion-rf-artifact/"
serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"

model = aiplatform.Model.upload(
    display_name=display_name,
    artifact_uri=artifact_uri,
    serving_container_image_uri=serving_container_image_uri,
    sync=True,
)

# =====================================================
# Step 4: Deploy Model to Vertex AI Endpoint
# =====================================================
deployed_model_display_name = "cart-conversion-endpoint"
traffic_split = {"0": 100}
machine_type = "n1-standard-2"
min_replica_count = 1
max_replica_count = 1

endpoint = model.deploy(
    deployed_model_display_name=deployed_model_display_name,
    traffic_split=traffic_split,
    machine_type=machine_type,
    min_replica_count=min_replica_count,
    max_replica_count=max_replica_count,
)

# =====================================================
# Step 5: Online Prediction on Unseen GCS Data
# Reads the pre-encoded unseen sessions CSV from your bucket and sends the
# first 5 rows to the live endpoint. The CSV is already in the model's
# expected feature order, so no client-side preprocessing is required.
# =====================================================
unseen_uri = f"gs://{BUCKET_NAME}/cart-conversion/cart_sessions_new.csv"
new_df = pd.read_csv(unseen_uri)
instances_list = new_df.head(5).values.tolist()
prediction = endpoint.predict(instances_list)
print("Online prediction:", prediction)

endpoint_id = endpoint.name.split("/")[-1]
print(f"\nEndpoint ID: {endpoint_id}")
print("Paste this Endpoint ID into curl-test.sh to call the endpoint from your terminal.")
