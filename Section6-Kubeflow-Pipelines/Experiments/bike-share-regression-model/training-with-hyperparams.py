import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from google.cloud import storage
from sklearn.pipeline import make_pipeline

storage_client = storage.Client()
bucket = storage_client.bucket("sid-kubeflow-v1")

def load_data(filename):
    df = pd.read_csv(filename)
    return df

def preprocess_data(df):
    df = df.rename(columns={'weathersit':'weather',
                            'yr':'year',
                            'mnth':'month',
                            'hr':'hour',
                            'hum':'humidity',
                            'cnt':'count'})
    df = df.drop(columns=['instant', 'dteday', 'year'])
    cols = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather']
    for col in cols:
        df[col] = df[col].astype('category')
    df['count'] = np.log(df['count'])
    df_oh = df.copy()
    for col in cols:
        df_oh = one_hot_encoding(df_oh, col)
    X = df_oh.drop(columns=['atemp', 'windspeed', 'casual', 'registered', 'count'], axis=1)
    y = df_oh['count']
    return X, y

def one_hot_encoding(data, column):
    data = pd.concat([data, pd.get_dummies(data[column], prefix=column, drop_first=True)], axis=1)
    data = data.drop([column], axis=1)
    return data

def train_model(model_name, x_train, y_train, hyper_params):
        
    if model_name == 'random_forest':
        max_depth = hyper_params['max_depth']
        n_estimators = hyper_params['n_estimators']

        model = RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)
        pipeline = make_pipeline(model)
        pipeline.fit(x_train, y_train)
        return pipeline

    elif model_name == 'xgboost':
        max_depth = hyper_params['max_depth']
        learning_rate = hyper_params['learning_rate']
        n_estimators = hyper_params['n_estimators']

        model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators)
        pipeline = make_pipeline(model)
        pipeline.fit(x_train, y_train)
        return pipeline

    elif model_name == 'svr':
        kernel = hyper_params['kernel']
        C = hyper_params['C']
        epsilon = hyper_params['epsilon']

        model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        pipeline = make_pipeline(model)
        pipeline.fit(x_train, y_train)
        return pipeline

    else:
        raise ValueError("Invalid model_name. Choose from 'random_forest', 'xgboost', or 'svr'.")

filename = 'gs://sid-kubeflow-v1/bikeshare-model/hour.csv'
df = load_data(filename)
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

hyper_params = {'max_depth':10,'n_estimators': 200}
model_name='xgboost'

pipeline = train_model(model_name,X_train, y_train,hyper_params)

y_pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)