"""
Simple Coupon Recommendation Pipeline
- Data validation component
- XGBoost model training component
- Conditional execution based on validation
Updated for kfp v2 latest patterns.
"""

from kfp import dsl, compiler
from kfp.dsl import component, Output, Metrics, OutputPath
from google.cloud.aiplatform import pipeline_jobs
import os

# --- Configuration ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
REGION = "us-central1"
BUCKET_NAME = f"{PROJECT_ID}-kubeflow-v1"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/coupon-pipeline-v1"
DATA_PATH = f"gs://{BUCKET_NAME}/coupon-recommendation/in-vehicle-coupon-recommendation.csv"


# =============================================================================
# Component 1: Validate Input Dataset
# =============================================================================
@component(
    packages_to_install=["gcsfs", "pandas", "google-cloud-storage"]
)
def validate_input_ds(
    filename: str,
    input_validation: OutputPath(str)
):
    import logging
    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Reading file: {filename}")

    df = pd.read_csv(filename)

    expected_num_cols = 26
    num_cols = len(df.columns)
    logging.info(f"Number of columns: {num_cols}")

    input_validation_value = "true"

    if num_cols != expected_num_cols:
        input_validation_value = "false"
        logging.error(f"Expected {expected_num_cols} columns, but found {num_cols}.")
    else:
        expected_col_names = [
            'destination', 'passanger', 'weather', 'temperature', 'time', 'coupon',
            'expiration', 'gender', 'age', 'maritalStatus', 'has_children',
            'education', 'occupation', 'income', 'car', 'Bar', 'CoffeeHouse',
            'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50',
            'toCoupon_GEQ5min', 'toCoupon_GEQ15min', 'toCoupon_GEQ25min',
            'direction_same', 'direction_opp', 'Y'
        ]

        if set(df.columns) != set(expected_col_names):
            input_validation_value = "false"
            missing_cols = set(expected_col_names) - set(df.columns)
            extra_cols = set(df.columns) - set(expected_col_names)
            logging.error("Column names do not match expected names.")
            if missing_cols:
                logging.error(f"Missing columns: {missing_cols}")
            if extra_cols:
                logging.error(f"Unexpected columns: {extra_cols}")

    with open(input_validation, 'w') as f:
        f.write(input_validation_value)


# =============================================================================
# Component 2: Custom Training Job - XGBoost
# =============================================================================
@component(
    packages_to_install=[
        "google-cloud-aiplatform", "gcsfs", "xgboost==2.1.1",
        "category_encoders", "imbalanced-learn", "pandas",
        "google-cloud-storage", "scikit-learn>=1.3"
    ]
)
def custom_training_job_component(
    project_id: str,
    bucket_name: str,
    max_depth: int,
    learning_rate: float,
    n_estimators: int,
    metrics: Output[Metrics],
    model_validation: OutputPath(str)
):
    import pandas as pd
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    from sklearn.model_selection import train_test_split
    from category_encoders import HashingEncoder
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    def load_data(file_path):
        df = pd.read_csv(file_path)
        return df

    def preprocess_data(df):
        df = df.drop(columns=['car', 'toCoupon_GEQ5min', 'direction_opp'])
        df = df.fillna(df.mode().iloc[0])
        df = df.drop_duplicates()

        df_dummy = df.copy()
        age_mapping = {
            'below21': '<21', '21': '21-30', '26': '21-30',
            '31': '31-40', '36': '31-40', '41': '41-50', '46': '41-50',
            '50plus': '>50'
        }
        age_list = []
        for i in df['age']:
            if i == 'below21':
                age = '<21'
            elif i in ['21', '26']:
                age = '21-30'
            elif i in ['31', '36']:
                age = '31-40'
            elif i in ['41', '46']:
                age = '41-50'
            else:
                age = '>50'
            age_list.append(age)
        df_dummy['age'] = age_list

        df_dummy['passanger_destination'] = df_dummy['passanger'].astype(str) + '-' + df_dummy['destination'].astype(str)
        df_dummy['marital_hasChildren'] = df_dummy['maritalStatus'].astype(str) + '-' + df_dummy['has_children'].astype(str)
        df_dummy['temperature_weather'] = df_dummy['temperature'].astype(str) + '-' + df_dummy['weather'].astype(str)
        df_dummy = df_dummy.drop(columns=[
            'passanger', 'destination', 'maritalStatus', 'has_children',
            'temperature', 'weather', 'Y'
        ])

        df_dummy = pd.concat([df_dummy, df['Y']], axis=1)
        df_dummy = df_dummy.drop(columns=['gender', 'RestaurantLessThan20'])

        df_le = df_dummy.replace({
            'expiration': {'2h': 0, '1d': 1},
            'age': {'<21': 0, '21-30': 1, '31-40': 2, '41-50': 3, '>50': 4},
            'education': {
                'Some High School': 0, 'High School Graduate': 1,
                'Some college - no degree': 2, 'Associates degree': 3,
                'Bachelors degree': 4, 'Graduate degree (Masters or Doctorate)': 5
            },
            'Bar': {'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4},
            'CoffeeHouse': {'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4},
            'CarryAway': {'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4},
            'Restaurant20To50': {'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4},
            'income': {
                'Less than $12500': 0, '$12500 - $24999': 1, '$25000 - $37499': 2,
                '$37500 - $49999': 3, '$50000 - $62499': 4, '$62500 - $74999': 5,
                '$75000 - $87499': 6, '$87500 - $99999': 7, '$100000 or More': 8
            },
            'time': {'7AM': 0, '10AM': 1, '2PM': 2, '6PM': 3, '10PM': 4}
        })

        # Newer pandas no longer auto-downcasts replaced values, so these
        # ordinal columns stay 'object'. Cast them to numeric explicitly,
        # otherwise XGBoost rejects the object dtypes.
        ordinal_cols = ['expiration', 'age', 'education', 'Bar', 'CoffeeHouse',
                        'CarryAway', 'Restaurant20To50', 'income', 'time']
        for col in ordinal_cols:
            df_le[col] = pd.to_numeric(df_le[col], errors='coerce')

        x = df_le.drop('Y', axis=1)
        y = df_le.Y
        return x, y

    def encode_features(x, n_components=27):
        hashing_enc = HashingEncoder(
            cols=['passanger_destination', 'marital_hasChildren', 'occupation',
                  'coupon', 'temperature_weather'],
            n_components=n_components
        ).fit(x)
        x_hashing = hashing_enc.transform(x.reset_index(drop=True))
        return x_hashing

    def oversample_data(x_train_hashing, y_train):
        sm = SMOTE(random_state=42)
        x_sm, y_sm = sm.fit_resample(x_train_hashing, y_train)
        return x_sm, y_sm

    def train_model(x_train, y_train, max_depth, learning_rate, n_estimators):
        model = XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(x_train, y_train)
        return model

    def evaluate_model(model, x_test, y_test):
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        return accuracy, precision, recall

    def save_model_artifact(model):
        artifact_name = 'model.bst'
        model.save_model(artifact_name)
        model_artifact = bucket.blob('coupon-recommendation/artifacts/' + artifact_name)
        model_artifact.upload_from_filename(artifact_name)

    # --- Main execution ---
    input_file = f"gs://{bucket_name}/coupon-recommendation/in-vehicle-coupon-recommendation.csv"
    df = load_data(input_file)
    x, y = preprocess_data(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_train.fillna(x_train.mode().iloc[0], inplace=True)
    x_test.fillna(x_train.mode().iloc[0], inplace=True)

    x_train_hashing = encode_features(x_train)
    x_test_hashing = encode_features(x_test)
    x_sm_train, y_sm_train = oversample_data(x_train_hashing, y_train)

    model = train_model(x_sm_train, y_sm_train, max_depth, learning_rate, n_estimators)
    accuracy, precision, recall = evaluate_model(model, x_test_hashing, y_test)

    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("precision", precision)
    metrics.log_metric("recall", recall)

    model_validation_value = "true"
    if accuracy > 0.5 and precision > 0.5 and recall > 0.5:
        save_model_artifact(model)
    else:
        model_validation_value = "false"

    with open(model_validation, 'w') as f:
        f.write(model_validation_value)


# =============================================================================
# Pipeline Definition
# =============================================================================
@dsl.pipeline(
    pipeline_root=PIPELINE_ROOT,
    name="coupon-model-training-pipeline",
)
def pipeline(
    project: str = PROJECT_ID,
    region: str = REGION
):
    max_depth = 5
    learning_rate = 0.2
    n_estimators = 40

    file_name = DATA_PATH
    input_validation_task = validate_input_ds(filename=file_name)

    with dsl.Condition(input_validation_task.outputs["input_validation"] == "true"):
        model_training_task = custom_training_job_component(
            project_id=project,
            bucket_name=BUCKET_NAME,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
        ).after(input_validation_task)


# =============================================================================
# Compile and Run
# =============================================================================
if __name__ == "__main__":
    # Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='coupon-pipeline.json'
    )
    print("Pipeline compiled to coupon-pipeline.json")

    # Run the pipeline
    start_pipeline = pipeline_jobs.PipelineJob(
        display_name="simple-kubeflow-pipeline",
        template_path="coupon-pipeline.json",
        enable_caching=False,
        location=REGION,
    )
    start_pipeline.run()
