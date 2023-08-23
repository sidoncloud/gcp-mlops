from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import datetime, timedelta
import pandas as pd
from google.cloud import storage
import gcsfs, json
from google.cloud import bigquery
from google.cloud import logging
from airflow.utils.dates import days_ago
from bank_campaign_model_training import (
    encode_categorical, apply_bucketing, preprocess_features, write_metrics_to_bigquery,
    get_classification_report, save_model_artifact, train_model
)

logging_client = logging.Client()
logger = logging_client.logger('bank-campaign-training-logs')

def validate_csv():
    # Load data
    fs = gcsfs.GCSFileSystem()
    bucket_name = 'sid-ml-ops'
    folder_name = 'bank_campaign_data'
    files = fs.ls(f'gs://{bucket_name}/{folder_name}')
    df = pd.concat([pd.read_csv(f'gs://{bucket_name}/{file}') for file in files], ignore_index=True, sort=False)
    
    # Define expected columns
    expected_cols = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 
                     'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
                     'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                     'euribor3m', 'nr.employed', 'y']
    
    # Check if the loaded columns are the same as expected columns
    if list(df.columns) == expected_cols:
        return True
    else:
        logger.log_struct({
            'keyword': 'Bank Campaign-KNN Model Training',
            'description': 'This log captures the last run for Model Training',
            'training_timestamp': datetime.now().isoformat(),
            'model_output_msg': "Input Data is not valid",
            'training_status': 0
        })
        raise ValueError(f'CSV does not have expected columns. Columns in CSV are: {list(df.columns)}')

def read_last_training_metrics():
    client = bigquery.Client()
    table_id = "udemy-mlops.ml_ops.bank_campaign_model_metrics"
    query = f"""
        SELECT *
        FROM `{table_id}`
        WHERE algo_name='xgboost'
        ORDER BY training_time DESC
        LIMIT 1
    """
    result = client.query(query).result()
    latest_row = next(result)
    return json.loads(latest_row[2])

def move_trained_datasets():
    fs = gcsfs.GCSFileSystem()
    files = fs.ls(f'gs://{bucket_name}/{folder_name}')
    trained_folder_name = 'trained_input_data'
    bucket_name = 'sid-ml-ops'
    folder_name = 'bank_campaign_data'
    for file in files:
        source_path = f'gs://{bucket_name}/{folder_name}/{file}'
        target_path = f'gs://{bucket_name}/{trained_folder_name}/{file}'
        fs.mv(source_path, target_path)
        fs.rm(source_path)

def evaluate_model():
    # Load data for evaluation
    fs = gcsfs.GCSFileSystem()
    bucket_name = 'sid-ml-ops'
    folder_name = 'bank_campaign_data'
    files = fs.ls(f'gs://{bucket_name}/{folder_name}')
    df = pd.concat([pd.read_csv(f'gs://{bucket_name}/{file}') for file in files], ignore_index=True, sort=False)
    
    # Preprocess the data
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                        'day_of_week', 'poutcome']
    df = encode_categorical(df, categorical_cols)
    df = apply_bucketing(df)
    X, y = preprocess_features(df)
    
    # Train the model on new data 
    model_name = "xgboost"
    pipeline = train_model(model_name, X, y)

    # Get the current model metrics for evaluation
    model_metrics = get_classification_report(pipeline, X, y)
    precision = model_metrics['0']['precision']
    recall = model_metrics['0']['recall']

    # Get the last/existing model metrics for comparison
    last_model_metrics = read_last_training_metrics()
    last_precision = last_model_metrics['0']['precision']
    last_recall = last_model_metrics['0']['recall']

    # Define the threshold values for precision and recall
    precision_threshold = 0.8
    recall_threshold = 0.8
    
    # Save the model artifact if metrics are above the thresholds
    if (precision >= precision_threshold and recall >= recall_threshold) and (precision >= last_precision and recall >= last_recall):
        save_model_artifact(model_name, pipeline)
        write_metrics_to_bigquery("xgboost", datetime.now(), model_metrics)
        logger.log_struct({
            'keyword': 'Bank Campaign-KNN Model Training',
            'description': 'This log captures the last run for Model Training',
            'training_timestamp': datetime.now().isoformat(),
            'model_output_msg': "Model artifact saved",
            'training_status': 1
        })

        # Move files to trained_input_data folder and delete from bank_campaign_data folder
        move_trained_datasets()

    else:
        logger.log_struct({
            'keyword': 'Bank Campaign-KNN Model Training',
            'description': 'This log captures the last run for Model Training',
            'training_timestamp': datetime.now().isoformat(),
            'model_output_msg': "Model metrics do not meet the defined threshold",
            'training_status': 0
        })

# Define the default_args dictionary
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    # 'start_date': datetime(2023, 1, 1, 10, 0, 0),  # Start date at 10 AM UTC
    'retries': 1,
}

# Instantiate the DAG
dag = DAG(
    'dag_bank_campaign_model_agn2',
    default_args=default_args,
    description='A not so simple training DAG',
    schedule_interval=timedelta(days=1),  # Run daily
)

# Define the tasks/operators
validate_csv_task = PythonOperator(
    task_id='validate_csv',
    python_callable=validate_csv,
    dag=dag,
)

evaluation_task = PythonOperator(
    task_id='model_evaluation',
    python_callable=evaluate_model,
    dag=dag,
)

validate_csv_task >> evaluation_task
