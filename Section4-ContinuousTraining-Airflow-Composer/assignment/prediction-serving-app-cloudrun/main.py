import pandas as pd
from flask import Flask, request, jsonify
import joblib
import os,json
from google.cloud import storage
from google.cloud import logging

app = Flask(__name__)
model = None

logging_client = logging.Client()
logger = logging_client.logger('advertising-roi-prediction-serving-logs')

def load_model():
    model = joblib.load("model.joblib")
    return model

def load_model_cloud():
    storage_client = storage.Client()
    bucket_name = "sid-ml-ops"
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob("advertising_roi/artifact/model.joblib")
    blob.download_to_filename("model.joblib")
    model = joblib.load("model.joblib")
    return model

@app.route('/predict', methods=['POST'])
def predict():
    model = load_model()
    try :
        input_json = request.get_json()
        # Log input request payload . Uncomment before deploying to cloud-run
        # logger.log_struct({
        #     'keyword': 'advertisement_roi_prediction_serving',
        #     'prediction_status':1,
        #     'error_msg': input_json
        # })

        input_df = pd.DataFrame(input_json, index=[0])
        y_predictions = model.predict(input_df)
        response = {'predictions': y_predictions.tolist()}

        # Log predicted value. Uncomment before deploying to cloud-run
        # logger.log_struct({
        #     'keyword': 'advertisement_roi_prediction_serving',
        #     'prediction_status':1,
        #     'predicted_output': y_predictions
        # })

        return jsonify(response), 200
    
    except Exception as e:
        #Uncomment before deploying to cloud-run
        # logger.log_struct({
        #     'keyword': 'advertisement_roi_prediction_serving',
        #     'prediction_status': 0,
        #     'error_msg': str(e)
        # })
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5050)))
