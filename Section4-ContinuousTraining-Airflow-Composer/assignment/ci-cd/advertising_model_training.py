import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from google.cloud import storage
from datetime import datetime
from google.cloud import bigquery
import json 

storage_client = storage.Client()
bucket = storage_client.bucket("sid-ml-ops")

def read_campaign_data(file_path):
    df = pd.read_csv(file_path)
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['YEAR'] = df['DATE'].dt.year
    df['MONTH'] = df['DATE'].dt.month
    return df

def calculate_revenue_per_month(file_path):
    df_revenue = pd.read_csv(file_path)
    df_revenue_per_month = df_revenue.groupby(['YEAR', 'MONTH'], as_index=False)['REVENUE'].sum()
    df_revenue_per_month = df_revenue_per_month.reset_index()
    return df_revenue_per_month

def calculate_spend_per_month(df):
    grouped_data = df.groupby(['YEAR', 'MONTH', 'CHANNEL'], as_index=False)['TOTAL_COST'].sum()
    grouped_data = grouped_data.sort_values(by=['YEAR', 'MONTH'])

    df_spend_per_month = pd.pivot_table(grouped_data, values='TOTAL_COST', index=['YEAR', 'MONTH'], columns='CHANNEL', aggfunc='sum')
    df_spend_per_month = df_spend_per_month.rename(columns={
        'search_engine': 'SEARCH_ENGINE',
        'social_media': 'SOCIAL_MEDIA',
        'video': 'VIDEO',
        'email': 'EMAIL'
    })
    df_spend_per_month = df_spend_per_month.reset_index()
    return df_spend_per_month

def merge_dataframes(df_revenue, df_spend):
    df_joined = df_revenue.merge(df_spend, on=["YEAR", "MONTH"])
    df_joined = df_joined.dropna()
    df_joined = df_joined.drop(["YEAR", "MONTH"], axis=1)
    return df_joined

def write_metrics_to_bigquery(algo_name, training_time, model_metrics):
    client = bigquery.Client()
    table_id = "udemy-mlops.ml_ops.advertising_roi_model_metrics"
    table = bigquery.Table(table_id)

    row = {"algo_name": algo_name, "training_time": training_time.strftime('%Y-%m-%d %H:%M:%S'), "model_metrics": json.dumps(model_metrics)}
    errors = client.insert_rows_json(table, [row])

    if errors == []:
        print("Metrics inserted successfully into BigQuery.")
    else:
        print("Error inserting metrics into BigQuery:", errors)

def train_model(df):
    numeric_features = ['SEARCH_ENGINE', 'SOCIAL_MEDIA', 'VIDEO', 'EMAIL']
    polynomial_features_degrees = 2

    numeric_transformer = Pipeline(steps=[('poly', PolynomialFeatures(degree=polynomial_features_degrees)), ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LinearRegression())])
    parameters = {}

    X = df.drop('REVENUE', axis=1)
    y = df['REVENUE']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    number_of_folds = 10
    model = GridSearchCV(pipeline, param_grid=parameters, cv=number_of_folds)

    model.fit(X_train, y_train)

    return model, X_train, y_train, X_test, y_test

def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_r2_score = model.score(X_train, y_train)
    test_r2_score = model.score(X_test, y_test)

    return train_r2_score, test_r2_score

def save_model(model):
    artifact_name = 'model.joblib'
    dump(model,artifact_name)
    model_artifact = bucket.blob('advertising_roi/artifact/'+artifact_name)
    model_artifact.upload_from_filename(artifact_name)

def main():

    campaign_file_path = "gs://sid-ml-ops/advertising_roi/campaign_spend.csv"
    df_spend = read_campaign_data(campaign_file_path)

    revenue_file_path = "gs://sid-ml-ops/advertising_roi/monthly_revenue.csv"
    df_revenue_per_month = calculate_revenue_per_month(revenue_file_path)

    df_spend_per_month = calculate_spend_per_month(df_spend)

    df_joined = merge_dataframes(df_revenue_per_month, df_spend_per_month)
    
    model, X_train, y_train, X_test, y_test = train_model(df_joined)

    train_r2_score, test_r2_score = evaluate_model(model, X_train, y_train, X_test, y_test)

    # print(train_r2_score, test_r2_score)
    model_metrics = {"r2_train":train_r2_score,"r2_test":test_r2_score}    
    training_time = datetime.now()
    model_name = "linear_regression"
    write_metrics_to_bigquery(model_name, training_time,model_metrics)
    save_model(model)

if __name__ == "__main__":
    main()