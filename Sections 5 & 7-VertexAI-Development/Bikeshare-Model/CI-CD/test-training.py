import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from google.cloud import storage
from joblib import dump
from sklearn.pipeline import make_pipeline
from model_training_code import load_data,preprocess_data,train_model,save_model_artifact

storage_client = storage.Client()
bucket = storage_client.bucket("sid-vertex-mlops")

def load_data(filename):
    df = pd.read_csv(filename)
    return df

df = load_data('gs://sid-vertex-mlops/bike-share/hour.csv')
X, y = preprocess_data(df)

def test_model_name_is_valid():
    with pytest.raises(ValueError):
        train_model("invalid_model_name", X, y)

def test_model_is_trained_correctly():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    pipeline = train_model("random_forest_regressor", X_train, y_train)
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    assert rmse < 10

def test_model_artifact_is_saved_correctly():
    assert bucket.blob('bike-share-rf-regression-artifact/model.joblib').exists()

def test_preprocess_data_returns_correct_columns():
    assert list(X.columns) == ['temp', 'humidity', 'season_2', 'season_3', 'season_4', 'month_2',
       'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8',
       'month_9', 'month_10', 'month_11', 'month_12', 'hour_1', 'hour_2',
       'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9',
       'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15',
       'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21',
       'hour_22', 'hour_23', 'holiday_1', 'weekday_1', 'weekday_2',
       'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'workingday_1',
       'weather_2', 'weather_3', 'weather_4']
    
if __name__ == '__main__':
    pytest.main()