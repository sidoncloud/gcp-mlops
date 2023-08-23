import pandas as pd
from flask import Flask, request, jsonify
from category_encoders import HashingEncoder
import pickle
import os
from google.cloud import storage

app = Flask(__name__)
model = None

def preprocess_data(df):
    # df = df.drop(columns=['car', 'toCoupon_GEQ5min', 'direction_opp'])
    df = df.fillna(df.mode().iloc[0])
    df = df.drop_duplicates()

    df_dummy = df.copy()
    age_list = []
    age_mapping = {'below21': '<21', '21': '21-30', '26': '21-30', '31': '31-40', '36': '31-40', '41': '41-50', '46': '41-50'}
    for i in df['age']:
        age_list.append(age_mapping.get(i, '>50'))

    df_dummy['age'] = age_list
    df_dummy['passanger_destination'] = df_dummy['passanger'].astype(str) + '-' + df_dummy['destination'].astype(str)
    df_dummy['marital_hasChildren'] = df_dummy['maritalStatus'].astype(str) + '-' + df_dummy['has_children'].astype(str)
    df_dummy['temperature_weather'] = df_dummy['temperature'].astype(str) + '-' + df_dummy['weather'].astype(str)
    df_dummy = df_dummy.drop(columns=['passanger', 'destination', 'maritalStatus', 'has_children', 'temperature','weather'])
    df_dummy = df_dummy.drop(columns=['gender', 'RestaurantLessThan20'])
    
    df_le = df_dummy.replace({
        'expiration':{'2h': 0, '1d' : 1},
        'age':{'<21': 0, '21-30': 1, '31-40': 2, '41-50': 3, '>50': 4},
        'education':{'Some High School': 0, 'High School Graduate': 1, 'Some college - no degree': 2,
                     'Associates degree': 3, 'Bachelors degree': 4, 'Graduate degree (Masters or Doctorate)': 5},
        'Bar':{'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4},
        'CoffeeHouse':{'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4}, 
        'CarryAway':{'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4}, 
        'Restaurant20To50':{'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4},
        'income':{'Less than $12500':0, '$12500 - $24999':1, '$25000 - $37499':2, '$37500 - $49999':3,
                  '$50000 - $62499':4, '$62500 - $74999':5, '$75000 - $87499':6, '$87500 - $99999':7,
                  '$100000 or More':8},
        'time':{'7AM':0, '10AM':1, '2PM':2, '6PM':3, '10PM':4}
    })
    return df_le

def encode_features(x, n_components=27):
    hashing_ros_enc = HashingEncoder(cols=['passanger_destination', 'marital_hasChildren', 'occupation', 'coupon',
                                           'temperature_weather'], n_components=n_components).fit(x)
    x_encoded = hashing_ros_enc.transform(x.reset_index(drop=True))
    return x_encoded

def _load_model():
    file_path = "artifacts/xgboost_coupon_recommendation.pkl"
    model = pickle.load(open(file_path, "rb"))
    return model

def load_model():
    storage_client = storage.Client()
    bucket_name = "sid-ml-ops"
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob("ml-artifacts/xgboost_coupon_recommendation.pkl")
    blob.download_to_filename("xgboost_coupon_recommendation.pkl")
    model = pickle.load(open("xgboost_coupon_recommendation.pkl", "rb"))
    return model

def preprocess(input_json):
    try:
        df = pd.DataFrame(input_json, index=[0])
        x = preprocess_data(df)
        x_encoded = encode_features(x)
        x_encoded.fillna(0, inplace=True)
        return x_encoded
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    model = load_model()
    try : 
        input_json = request.get_json()
        df_preprocessed = preprocess(input_json)
        y_predictions = model.predict(df_preprocessed)
        response = {'predictions': y_predictions.tolist()}
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5051)))