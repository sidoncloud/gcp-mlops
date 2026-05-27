"""
Product Return Prediction - Model Training Code
Trains a RandomForestClassifier on the orders dataset and saves the trained
model artifact to Google Cloud Storage.

Runs inside the prebuilt scikit-learn training container that the orchestrator
script (python-sdk-training-deployment.py) launches.

Updated: Python 3.12, google-cloud-storage>=2.0.0
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from google.cloud import storage
from joblib import dump
from sklearn.pipeline import make_pipeline

# ---- Configuration ----
BUCKET_NAME = "YOUR_BUCKET_NAME"  # Replace with your GCS bucket name
DATA_PATH = f"gs://{BUCKET_NAME}/order-return/orders_train.csv"
ARTIFACT_GCS_PATH = "order-return-rf-artifact/model.joblib"

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)


def load_data(filename: str) -> pd.DataFrame:
    """Load dataset from GCS."""
    df = pd.read_csv(filename)
    return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean, encode, and prepare features and target."""
    df = df.drop(columns=['order_id'])

    cols = ['product_category', 'payment_method']
    for col in cols:
        df[col] = df[col].astype('category')

    df_oh = df.copy()
    for col in cols:
        df_oh = one_hot_encoding(df_oh, col)

    X = df_oh.drop(columns=['returned'])
    y = df_oh['returned']
    return X, y


def one_hot_encoding(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """One-hot encode a categorical column."""
    data = pd.concat([data, pd.get_dummies(data[column], prefix=column, drop_first=True)], axis=1)
    data = data.drop([column], axis=1)
    return data


def train_model(x_train: pd.DataFrame, y_train: pd.Series):
    """Train a RandomForestClassifier pipeline."""
    model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
    pipeline = make_pipeline(model)
    pipeline.fit(x_train, y_train)
    return pipeline


def save_model_artifact(pipeline) -> None:
    """Save trained model to GCS."""
    artifact_name = 'model.joblib'
    dump(pipeline, artifact_name)
    model_artifact = bucket.blob(ARTIFACT_GCS_PATH)
    model_artifact.upload_from_filename(artifact_name)
    print(f"Model artifact uploaded to gs://{BUCKET_NAME}/{ARTIFACT_GCS_PATH}")


def main():
    df = load_data(DATA_PATH)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = train_model(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    save_model_artifact(pipeline)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC:  {auc:.4f}")


if __name__ == '__main__':
    main()
