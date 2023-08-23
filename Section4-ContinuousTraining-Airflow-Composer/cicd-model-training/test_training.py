import pytest
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from joblib import load
from bank_campaign_model_training import (
    encode_categorical, preprocess_features, apply_bucketing,
    train_model, get_classification_report, load_model_artifact,
)
import pandas as pd

@pytest.fixture
def dummy_data():
    # Prepare dummy data for testing
    data = {
        'age': [30, 40, 50, 60],
        'job': ['admin.', 'technician', 'self-employed', 'management'],
        'marital': ['married', 'single', 'married', 'divorced'],
        'education': ['university.degree', 'basic.9y', 'high.school', 'basic.4y'],
        'default': ['no', 'no', 'no', 'yes'],
        'housing': ['yes', 'yes', 'no', 'no'],
        'loan': ['no', 'yes', 'no', 'yes'],
        'contact': ['cellular', 'telephone', 'cellular', 'telephone'],
        'month': ['may', 'jun', 'jul', 'aug'],
        'day_of_week': ['mon', 'tue', 'wed', 'thu'],
        'duration': [200, 300, 400, 500],
        'campaign': [1, 2, 3, 4],
        'pdays': [20, 30, 40, 999],
        'previous': [1, 2, 3, 4],
        'poutcome': ['success', 'failure', 'nonexistent', 'failure'],
        'emp.var.rate': [1.1, 2.2, 3.3, 4.4],
        'cons.price.idx': [90.1, 90.2, 90.3, 90.4],
        'cons.conf.idx': [-30.1, -30.2, -30.3, -30.4],
        'euribor3m': [1.0, 2.0, 3.0, 4.0],
        'nr.employed': [5000, 6000, 7000, 8000],
        'y': ['yes', 'no', 'yes', 'no']
    }
    return pd.DataFrame(data)

def test_preprocess_features(dummy_data):
    df = dummy_data
    df = encode_categorical(df, ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                                 'day_of_week', 'poutcome'])
    df = apply_bucketing(df)
    X, y = preprocess_features(df)
    assert X.shape == (4, 19)  
    assert y.shape == (4,)
    
def test_data_loading(dummy_data):
    df = dummy_data
    assert len(df.columns) == 21

def test_categorical_encoding(dummy_data):
    # Test categorical encoding function
    df = dummy_data
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    encoded_df = encode_categorical(df, categorical_cols)
    assert encoded_df.shape == df.shape  # Check if the shape is preserved after encoding
    
def test_get_classification_report(dummy_data):
    df = dummy_data
    df = encode_categorical(df, ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                                 'day_of_week', 'poutcome'])
    df = apply_bucketing(df)
    X, y = preprocess_features(df)
    model = load_model_artifact('xgboost_model.joblib')
    report = get_classification_report(model, X, y)
    assert isinstance(report, dict)  
    assert '0' in report.keys()  
    
def test_train_model(dummy_data):
    df = dummy_data
    df = encode_categorical(df, ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                                 'day_of_week', 'poutcome'])
    df = apply_bucketing(df)
    X, y = preprocess_features(df)
    model = train_model('xgboost', X, y)
    assert isinstance(model, Pipeline)
