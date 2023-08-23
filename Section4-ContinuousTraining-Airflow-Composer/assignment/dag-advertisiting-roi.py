# !pip3 install xgboost

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
from google.cloud import storage
import gcsfs,json
from datetime import datetime
from google.cloud import bigquery
from google.cloud import logging

from advertising_model_training import (
    read_campaign_data,calculate_revenue_per_month,calculate_spend_per_month,merge_dataframes,train_model,
    evaluate_model,save_model
)

logging_client = logging.Client()
logger = logging_client.logger('advertising-spends-ct-logs')

def validate_input_data():
    fs = gcsfs.GCSFileSystem()
    with fs.open('gs://sid-ml-ops/advertising_roi/campaign_spend.csv') as f:
        df1 = pd.read_csv(f)
    
    with fs.open('gs://sid-ml-ops/advertising_roi/monthly_revenue.csv') as f:
        df2 = pd.read_csv(f)
    
    df1_expected_cols = ['CAMPAIGN', 'CHANNEL', 'DATE', 'TOTAL_CLICKS', 'TOTAL_COST','ADS_SERVED']

    df2_expected_cols = ['YEAR', 'MONTH', 'REVENUE']

    # Check if the loaded columns are same as expected columns
    if (list(df1.columns) == df1_expected_cols) and (list(df2.columns) == df2_expected_cols):
        return True
    else:
        logger.log_struct({
            'keyword': 'advertisement_roi_training',
            'training_timestamp': datetime.now().isoformat(),
            'model_output_msg': "Input Data is not valid",
            'training_status':0
        })
        raise ValueError(f'CSV does not have expected columns')

def read_last_training_metrics():
    client = bigquery.Client()
    table_id = "udemy-mlops.ml_ops.advertising_roi_model_metrics"
    query = f"""
        SELECT *
        FROM `{table_id}`
        where algo_name='linear_regression'
        ORDER BY training_time DESC
        LIMIT 1
        """
    result = client.query(query).result()
    latest_row = next(result)
    return json.loads(latest_row[2])

def continuous_training():

    fs = gcsfs.GCSFileSystem()
    
    campaign_file_path = "gs://sid-ml-ops/advertising_roi/campaign_spend.csv"
    df_spend = read_campaign_data(campaign_file_path)

    revenue_file_path = "gs://sid-ml-ops/advertising_roi/monthly_revenue.csv"
    df_revenue_per_month = calculate_revenue_per_month(revenue_file_path)

    df_spend_per_month = calculate_spend_per_month(df_spend)

    df_joined = merge_dataframes(df_revenue_per_month,df_spend_per_month)
    
    model, X_train, y_train, X_test, y_test = train_model(df_joined)

    train_r2_score, test_r2_score = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    last_model_metrics = read_last_training_metrics()

    last_train_r2 = last_model_metrics['0']['r2_train']
    last_test_r2 = last_model_metrics['0']['r2_test']

    if (test_r2_score>=last_test_r2 and train_r2_score>=last_train_r2) and (train_r2_score>0.9 and test_r2_score>=0.9):
        save_model(model)
        logger.log_struct({
                'keyword': 'advertisement_roi_training',
                'training_timestamp': datetime.now().isoformat(),
                'test_r2_score': test_r2_score,
                'training_status':1
            })
    else :
        logger.log_struct({
                'keyword': 'advertisement_roi_training',
                'training_timestamp': datetime.now().isoformat(),
                'test_r2_score': test_r2_score,
                'training_status':0
            })
    
# Define the default_args dictionary
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'retries': 1,
}

# Instantiate the DAG
dag = DAG(
    'dag_advertisement_roi_ct',
    default_args=default_args,
    schedule_interval=None,
)

# Define the tasks/operators
ct = PythonOperator(
    task_id='continuous_training',
    python_callable=continuous_training,
    dag=dag
)