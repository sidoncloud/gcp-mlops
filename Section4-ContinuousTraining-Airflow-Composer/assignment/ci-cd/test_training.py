import pytest
from unittest.mock import patch
from advertising_model_training import read_campaign_data, write_metrics_to_bigquery, train_model
import pandas as pd
from datetime import datetime

@pytest.fixture
def dummy_data():
    data = {
        'SEARCH_ENGINE': [100, 200, 300, 400],
        'SOCIAL_MEDIA': [50, 100, 150, 200],
        'VIDEO': [30, 60, 90, 120],
        'EMAIL': [20, 40, 60, 80],
        'REVENUE': [1000, 2000, 3000, 4000],
        'DATE': pd.date_range(start='1/1/2020', periods=4)
    }
    return pd.DataFrame(data)

@patch('pandas.read_csv')
def test_read_campaign_data(mock_read_csv, dummy_data):
    mock_read_csv.return_value = dummy_data
    df = read_campaign_data('dummy_file_path')
    assert not df.empty
    assert 'YEAR' in df.columns
    assert 'MONTH' in df.columns

@patch('advertising_model_training.bigquery.Client')  # replace 'your_module' with actual module name
def test_write_metrics_to_bigquery(mock_bigquery_Client):
    model_metrics = {"r2_train": 0.8, "r2_test": 0.7}
    write_metrics_to_bigquery("linear_regression", datetime.now(), model_metrics)
    assert mock_bigquery_Client.return_value.insert_rows_json.called

@patch('advertising_model_training.GridSearchCV')  # replace 'your_module' with actual module name
def test_train_model(mock_GridSearchCV, dummy_data):
    mock_GridSearchCV.return_value = mock_GridSearchCV
    mock_GridSearchCV.fit.return_value = None
    df = dummy_data
    model, X_train, y_train, X_test, y_test = train_model(df)
    assert mock_GridSearchCV.fit.called
