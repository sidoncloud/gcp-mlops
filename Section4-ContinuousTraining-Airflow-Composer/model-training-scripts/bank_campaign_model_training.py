# !pip install imblearn
# !pip install xgboost

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump, load
from google.cloud import storage
import json
from google.cloud import bigquery
from datetime import datetime
from sklearn.pipeline import make_pipeline
storage_client = storage.Client()
bucket = storage_client.bucket("sid-ml-ops")

def load_data(path):
    return pd.read_csv(path, sep=";")

def encode_categorical(df, categorical_cols):
    le = LabelEncoder()
    df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
    return df

def preprocess_features(df):
    X = df.drop('y', axis=1)
    y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

    sc = StandardScaler()
    X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
    return X, y

def bucket_pdays(pdays):
    if pdays == 999:
        return 0
    elif pdays <= 30:
        return 1
    else:
        return 2

def apply_bucketing(df):
    df['pdays_bucketed'] = df['pdays'].apply(bucket_pdays)
    df = df.drop('pdays', axis=1)
    df = df.drop('duration', axis=1)
    return df

def train_model(model_name, x_train, y_train):
    if model_name == 'logistic':
        model = LogisticRegression(random_state=42)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(random_state=42)
    elif model_name == 'knn':
        model = KNeighborsClassifier()
    elif model_name == 'xgboost':
        model = XGBClassifier(random_state=42, use_label_encoder=False)
    else:
        raise ValueError("Invalid model name.")

    pipeline = make_pipeline(model)
    pipeline.fit(x_train, y_train)
    return pipeline

def get_classification_report(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report

def save_model_artifact(model_name, pipeline):
    artifact_name = model_name+'_model.joblib'
    dump(pipeline, artifact_name)
    # Uncomment below lines for cloud execution
    model_artifact = bucket.blob('bank_campaign_artifact/'+artifact_name)
    model_artifact.upload_from_filename(artifact_name)

def load_model_artifact(file_name):
    blob = bucket.blob("ml-artifacts/" + file_name)
    blob.download_to_filename(file_name)
    return load(file_name)

def write_metrics_to_bigquery(algo_name, training_time, model_metrics):
    client = bigquery.Client()
    table_id = "udemy-mlops.ml_ops.bank_campaign_model_metrics"
    table = bigquery.Table(table_id)

    row = {"algo_name": algo_name, "training_time": training_time.strftime('%Y-%m-%d %H:%M:%S'), "model_metrics": json.dumps(model_metrics)}
    errors = client.insert_rows_json(table, [row])

    if errors == []:
        print("Metrics inserted successfully into BigQuery.")
    else:
        print("Error inserting metrics into BigQuery:", errors)

def main():
    input_data_path = "gs://sid-ml-ops/bank_campaign_data/bank-campaign-training-data.csv"
    model_name = 'xgboost'
    df = load_data(input_data_path)
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    df = encode_categorical(df, categorical_cols)
    df = apply_bucketing(df)
    X, y = preprocess_features(df)
    
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    pipeline = train_model(model_name, X_train, y_train)
    accuracy_metrics = get_classification_report(pipeline, X_test, y_test)
    training_time = datetime.now()
    write_metrics_to_bigquery(model_name, training_time, accuracy_metrics)
    save_model_artifact(model_name, pipeline)

if __name__ == "__main__":
    main()