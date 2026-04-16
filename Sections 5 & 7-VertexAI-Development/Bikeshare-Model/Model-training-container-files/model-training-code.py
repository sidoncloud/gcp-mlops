"""
Bikeshare Demand Prediction - Custom Container Training Code
Runs inside a custom Docker container on Vertex AI.

Updated: Python 3.12, google-cloud-storage>=2.0.0
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from google.cloud import storage
from joblib import dump
from sklearn.pipeline import make_pipeline

# ---- Configuration ----
BUCKET_NAME = "mlops-udemy7944"  # Replace with your GCS bucket name
DATA_PATH = f"gs://{BUCKET_NAME}/bike-share/hour.csv"
ARTIFACT_GCS_PATH = "bikeshare-artifact/model.joblib"

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)


def load_data(filename: str) -> pd.DataFrame:
    """Load dataset from GCS."""
    df = pd.read_csv(filename)
    return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean, encode, and prepare features and target."""
    df = df.rename(columns={
        'weathersit': 'weather',
        'yr': 'year',
        'mnth': 'month',
        'hr': 'hour',
        'hum': 'humidity',
        'cnt': 'count'
    })
    df = df.drop(columns=['instant', 'dteday', 'year'])

    cols = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather']
    for col in cols:
        df[col] = df[col].astype('category')

    df['count'] = np.log(df['count'])

    df_oh = df.copy()
    for col in cols:
        df_oh = one_hot_encoding(df_oh, col)

    X = df_oh.drop(columns=['atemp', 'windspeed', 'casual', 'registered', 'count'])
    y = df_oh['count']
    return X, y


def one_hot_encoding(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """One-hot encode a categorical column."""
    data = pd.concat([data, pd.get_dummies(data[column], prefix=column, drop_first=True)], axis=1)
    data = data.drop([column], axis=1)
    return data


def train_model(x_train: pd.DataFrame, y_train: pd.Series):
    """Train a RandomForestRegressor pipeline."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    pipeline = train_model(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    save_model_artifact(pipeline)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.4f}")


if __name__ == '__main__':
    main()
