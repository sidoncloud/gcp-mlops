"""
Credit Scoring End-to-End Kubeflow Pipeline (assignment solution)

One pipeline that does everything:
  1. Validate the German Credit dataset schema
  2. Train an XGBoost classifier, log accuracy / precision / recall / F1
  3. Gate on F1 > 0.70 before promoting the model
  4. Upload the model to the Vertex AI Model Registry
  5. Deploy it to an endpoint
  6. Send a sample prediction to confirm the endpoint is live

After the run, undeploy and delete the endpoint manually from the
Vertex AI console (or via gcloud) so the n1-standard-4 stops billing.
"""

from kfp import dsl, compiler
from kfp.dsl import (Output, Metrics, ClassificationMetrics, component)
from google.cloud.aiplatform import pipeline_jobs
from typing import NamedTuple
import os


# =============================================================================
# Configuration
# =============================================================================
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
REGION = "us-central1"
BUCKET_NAME = f"{PROJECT_ID}-kubeflow-v1"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/credit-scoring-pipeline"
DATA_PATH = f"gs://{BUCKET_NAME}/credit-scoring/credit_files.csv"


# =============================================================================
# Component 1: Validate Input Dataset
# =============================================================================
@component(
    packages_to_install=["gcsfs", "pandas", "google-cloud-storage"]
)
def validate_input_ds(
    filename: str,
) -> NamedTuple("output", [("input_validation", str)]):
    import logging
    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Reading file: {filename}")

    df = pd.read_csv(filename)
    expected_num_cols = 21
    num_cols = len(df.columns)
    logging.info(f"Number of columns: {num_cols}")

    input_validation = "true"
    if num_cols != expected_num_cols:
        input_validation = "false"

    expected_col_names = [
        'CREDIT_REQUEST_ID', 'CREDIT_AMOUNT', 'CREDIT_DURATION', 'PURPOSE',
        'INSTALLMENT_COMMITMENT', 'OTHER_PARTIES', 'CREDIT_STANDING',
        'CREDIT_SCORE', 'CHECKING_BALANCE', 'SAVINGS_BALANCE',
        'EXISTING_CREDITS', 'ASSETS', 'HOUSING', 'QUALIFICATION', 'JOB_HISTORY',
        'AGE', 'SEX', 'MARITAL_STATUS', 'NUM_DEPENDENTS', 'RESIDENCE_SINCE',
        'OTHER_PAYMENT_PLANS'
    ]
    if set(df.columns) != set(expected_col_names):
        input_validation = "false"

    return (input_validation,)


# =============================================================================
# Component 2: Train XGBoost and Gate on F1
# =============================================================================
@component(
    packages_to_install=[
        "google-cloud-aiplatform", "gcsfs", "xgboost==2.1.1",
        "scikit-learn>=1.3", "pandas", "google-cloud-storage"
    ]
)
def train_model_component(
    bucket_name: str,
    max_depth: int,
    learning_rate: float,
    n_estimators: int,
    f1_threshold: float,
    metrics: Output[Metrics],
    performance_metrics: Output[ClassificationMetrics],
) -> NamedTuple("output", [("model_validation", str)]):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        confusion_matrix, precision_score, recall_score,
        accuracy_score, f1_score,
    )
    from xgboost import XGBClassifier
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # ---- Categorical encoding helpers ----
    def purpose_encode(x):
        return {"Consumer Goods": 1, "Vehicle": 2, "Tuition": 3, "Business": 4, "Repairs": 5}.get(x, 0)

    def other_parties_encode(x):
        return {"Guarantor": 1, "Co-Applicant": 2}.get(x, 0)

    def qualification_encode(x):
        return {"unskilled": 1, "skilled": 2, "highly skilled": 3}.get(x, 0)

    def credit_standing_encode(x):
        return 1 if x == "good" else 0

    def assets_encode(x):
        return {"Vehicle": 1, "Investments": 2, "Home": 3}.get(x, 0)

    def housing_encode(x):
        return {"rent": 1, "own": 2}.get(x, 0)

    def marital_status_encode(x):
        return {"Married": 1, "Single": 2}.get(x, 0)

    def other_payment_plans_encode(x):
        return {"bank": 1, "stores": 2}.get(x, 0)

    def sex_encode(x):
        return 1 if x == "M" else 0

    def preprocess_data(df):
        df["PURPOSE_CODE"] = df["PURPOSE"].apply(purpose_encode)
        df["OTHER_PARTIES_CODE"] = df["OTHER_PARTIES"].apply(other_parties_encode)
        df["QUALIFICATION_CODE"] = df["QUALIFICATION"].apply(qualification_encode)
        df["CREDIT_STANDING_CODE"] = df["CREDIT_STANDING"].apply(credit_standing_encode)
        df["ASSETS_CODE"] = df["ASSETS"].apply(assets_encode)
        df["HOUSING_CODE"] = df["HOUSING"].apply(housing_encode)
        df["MARITAL_STATUS_CODE"] = df["MARITAL_STATUS"].apply(marital_status_encode)
        df["OTHER_PAYMENT_PLANS_CODE"] = df["OTHER_PAYMENT_PLANS"].apply(other_payment_plans_encode)
        df["SEX_CODE"] = df["SEX"].apply(sex_encode)

        df = df.drop(columns=[
            "PURPOSE", "OTHER_PARTIES", "QUALIFICATION", "CREDIT_STANDING",
            "ASSETS", "HOUSING", "MARITAL_STATUS", "OTHER_PAYMENT_PLANS", "SEX",
        ])
        return df

    def save_model_artifact(model):
        artifact_name = "model.bst"
        # Save the native booster, not the sklearn wrapper. The wrapper's
        # save_model() trips over _estimator_type with newer scikit-learn.
        model.get_booster().save_model(artifact_name)
        blob = bucket.blob("credit-scoring/artifacts/" + artifact_name)
        blob.upload_from_filename(artifact_name)

    # ---- Main flow ----
    input_file = f"gs://{bucket_name}/credit-scoring/credit_files.csv"
    df = pd.read_csv(input_file)
    df = preprocess_data(df)

    X = df.drop("CREDIT_STANDING_CODE", axis=1)
    y = df["CREDIT_STANDING_CODE"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    model = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("precision", precision)
    metrics.log_metric("recall", recall)
    metrics.log_metric("f1", f1)

    performance_metrics.log_confusion_matrix(
        ["Denied", "Approved"],
        confusion_matrix(y_test, y_pred).tolist(),
    )

    # Quality gate: only promote the model if F1 clears the threshold.
    if f1 >= f1_threshold:
        save_model_artifact(model)
        model_validation = "true"
    else:
        model_validation = "false"

    return (model_validation,)


# =============================================================================
# Component 3: Upload to Registry and Deploy to Endpoint
# =============================================================================
@component(
    packages_to_install=["google-cloud-aiplatform"]
)
def deploy_model_component(
    project_id: str,
    bucket_name: str,
) -> NamedTuple("output", [("endpoint", str)]):
    from google.cloud import aiplatform

    aiplatform.init(
        project=project_id,
        location="us-central1",
        staging_bucket=f"gs://{bucket_name}",
    )

    model = aiplatform.Model.upload(
        display_name="credit-scoring-model",
        artifact_uri=f"gs://{bucket_name}/credit-scoring/artifacts/",
        serving_container_image_uri=(
            "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"
        ),
        sync=True,
    )

    endpoint = model.deploy(
        deployed_model_display_name="credit-scoring-deployment",
        traffic_split={"0": 100},
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=1,
    )

    return (str(endpoint.resource_name),)


# =============================================================================
# Component 4: Send a Sample Prediction
# =============================================================================
@component(
    packages_to_install=["google-cloud-aiplatform"]
)
def test_prediction_component(endpoint_resource_name: str) -> str:
    from google.cloud import aiplatform

    endpoint = aiplatform.Endpoint(endpoint_resource_name)

    # One synthetic applicant in the post-preprocessing feature order. The
    # exact values are illustrative; what we are verifying here is that the
    # endpoint accepts a request and returns a prediction.
    sample = [
        1,        # CREDIT_REQUEST_ID
        5000.0,   # CREDIT_AMOUNT
        24,       # CREDIT_DURATION
        2,        # INSTALLMENT_COMMITMENT
        700,      # CREDIT_SCORE
        1500.0,   # CHECKING_BALANCE
        2000.0,   # SAVINGS_BALANCE
        1,        # EXISTING_CREDITS
        5,        # JOB_HISTORY
        35,       # AGE
        2,        # NUM_DEPENDENTS
        4,        # RESIDENCE_SINCE
        1,        # PURPOSE_CODE
        0,        # OTHER_PARTIES_CODE
        2,        # QUALIFICATION_CODE
        3,        # ASSETS_CODE
        2,        # HOUSING_CODE
        1,        # MARITAL_STATUS_CODE
        1,        # OTHER_PAYMENT_PLANS_CODE
        1,        # SEX_CODE
    ]

    response = endpoint.predict(instances=[sample])
    print("Test prediction response:", response.predictions)
    return str(response.predictions)


# =============================================================================
# Pipeline Definition
# =============================================================================
@dsl.pipeline(
    pipeline_root=PIPELINE_ROOT,
    name="credit-scoring-pipeline",
)
def pipeline(project: str = PROJECT_ID, region: str = REGION):
    # Hyperparameters
    max_depth = 5
    learning_rate = 0.2
    n_estimators = 80
    f1_threshold = 0.70  # matches the assignment's promotion bar

    validation = validate_input_ds(filename=DATA_PATH)

    with dsl.Condition(
        validation.outputs["input_validation"] == "true",
        name="data-is-valid",
    ):
        training = train_model_component(
            bucket_name=BUCKET_NAME,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            f1_threshold=f1_threshold,
        ).after(validation)

        with dsl.Condition(
            training.outputs["model_validation"] == "true",
            name="model-clears-f1",
        ):
            deployment = deploy_model_component(
                project_id=project,
                bucket_name=BUCKET_NAME,
            ).after(training)

            _ = test_prediction_component(
                endpoint_resource_name=deployment.outputs["endpoint"],
            ).after(deployment)


# =============================================================================
# Compile and Run
# =============================================================================
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="credit-scoring-pipeline.json",
    )
    print("Pipeline compiled to credit-scoring-pipeline.json")

    start_pipeline = pipeline_jobs.PipelineJob(
        display_name="credit-scoring-pipeline",
        template_path="credit-scoring-pipeline.json",
        enable_caching=False,
        location=REGION,
    )
    start_pipeline.run()
