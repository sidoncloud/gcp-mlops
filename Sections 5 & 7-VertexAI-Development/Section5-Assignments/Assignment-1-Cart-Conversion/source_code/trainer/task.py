"""
Vertex AI Custom Training entry point for the cart conversion scoring model.

Vertex AI will run this file inside the pre-built scikit-learn training container.
It reads the training CSV from GCS, trains a RandomForest classifier wrapped in a
Pipeline with a StandardScaler, and writes `model.joblib` plus `feature_order.json`
to the model directory (AIP_MODEL_DIR, which is a gs:// URI).

The resulting model.joblib can be served by the pre-built sklearn prediction
container: it expects `instances` to be a 2D list of numeric feature values in
the exact order of FEATURE_ORDER.
"""
import argparse
import json
import logging
import os
import tempfile

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("trainer")

DEVICE_MAP = {"mobile": 0, "desktop": 1, "tablet": 2}
TRAFFIC_MAP = {"organic": 0, "paid": 1, "direct": 2, "email": 3}

FEATURE_ORDER = [
    "session_duration_seconds",
    "pages_viewed",
    "items_in_cart",
    "cart_total_value",
    "unique_categories_viewed",
    "has_discount_code",
    "is_returning_customer",
    "device_code",
    "traffic_code",
    "hour_of_day",
    "day_of_week",
    "previous_purchases_count",
    "avg_time_per_page_seconds",
    "added_to_wishlist",
    "used_search",
]
TARGET_COLUMN = "converted"


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["device_code"] = df["device_type"].map(DEVICE_MAP)
    df["traffic_code"] = df["traffic_source"].map(TRAFFIC_MAP)
    if df["device_code"].isna().any() or df["traffic_code"].isna().any():
        raise ValueError("Unknown category value in device_type or traffic_source")
    return df


def load_training_data(uri: str):
    log.info("Reading training data from %s", uri)
    df = pd.read_csv(uri)
    df = encode(df)
    X = df[FEATURE_ORDER].astype(float).values
    y = df[TARGET_COLUMN].values
    log.info("Loaded %d rows, positive rate %.2f%%", len(df), 100.0 * float(y.mean()))
    return X, y


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def save_model(pipe: Pipeline, model_dir: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        local_model = os.path.join(tmp, "model.joblib")
        local_features = os.path.join(tmp, "feature_order.json")
        joblib.dump(pipe, local_model)
        with open(local_features, "w") as f:
            json.dump(FEATURE_ORDER, f)

        if model_dir.startswith("gs://"):
            from google.cloud import storage

            rest = model_dir[len("gs://") :]
            bucket_name, _, prefix = rest.partition("/")
            prefix = prefix.rstrip("/")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            bucket.blob(f"{prefix}/model.joblib").upload_from_filename(local_model)
            bucket.blob(f"{prefix}/feature_order.json").upload_from_filename(local_features)
            log.info("Uploaded model to %s", model_dir)
        else:
            os.makedirs(model_dir, exist_ok=True)
            import shutil

            shutil.copy(local_model, os.path.join(model_dir, "model.joblib"))
            shutil.copy(local_features, os.path.join(model_dir, "feature_order.json"))
            log.info("Saved model to %s", model_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data-uri", required=True, help="gs:// URI of training CSV")
    parser.add_argument("--model-dir", default=os.environ.get("AIP_MODEL_DIR", "./model_out"))
    args = parser.parse_args()

    X, y = load_training_data(args.training_data_uri)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    log.info("Fitting pipeline on %d rows", len(X_train))
    pipe.fit(X_train, y_train)

    val_preds = pipe.predict(X_val)
    val_probas = pipe.predict_proba(X_val)[:, 1]
    log.info("Validation accuracy: %.4f", accuracy_score(y_val, val_preds))
    log.info("Validation ROC AUC:  %.4f", roc_auc_score(y_val, val_probas))

    save_model(pipe, args.model_dir)
    log.info("Training complete")


if __name__ == "__main__":
    main()
