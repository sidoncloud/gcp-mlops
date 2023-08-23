import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from google.cloud import storage
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score,confusion_matrix

storage_client = storage.Client()
bucket = storage_client.bucket("sid-kubeflow-v1")

def purpose_encode(x):
    if x == "Consumer Goods":
        return 1
    elif x == "Vehicle":
        return 2
    elif x == "Tuition":
        return 3
    elif x == "Business":
        return 4
    elif x == "Repairs":
        return 5
    else:
        return 0

def other_parties_encode(x):
    if x == "Guarantor":
        return 1
    elif x == "Co-Applicant":
        return 2
    else:
        return 0

def qualification_encode(x):
    if x == "unskilled":
        return 1
    elif x == "skilled":
        return 2
    elif x == "highly skilled":
        return 3
    else:
        return 0

def credit_standing_encode(x):
    if x == "good":
        return 1
    else:
        return 0

def assets_encode(x):
    if x == "Vehicle":
        return 1
    elif x == "Investments":
        return 2
    elif x == "Home":
        return 3
    else:
        return 0

def housing_encode(x):
    if x == "rent":
        return 1
    elif x == "own":
        return 2
    else:
        return 0

def marital_status_encode(x):
    if x == "Married":
        return 1
    elif x == "Single":
        return 2
    else:
        return 0

def other_payment_plans_encode(x):
    if x == "bank":
        return 1
    elif x == "stores":
        return 2
    else:
        return 0

def sex_encode(x):
    if x == "M":
        return 1
    else:
        return 0
    
def credit_score_decode(x):
    return "Approved" if x == 1 else "Denied"

def preprocess_data(df):
    df["PURPOSE_CODE"] = df["PURPOSE"].apply(purpose_encode)
    df["OTHER_PARTIES_CODE"] = df["OTHER_PARTIES"].apply(other_parties_encode)
    df["QUALIFICATION_CODE"] = df["QUALIFICATION"].apply(qualification_encode)
    df["CREDIT_STANDING_CODE"] = df["CREDIT_STANDING"].apply(credit_standing_encode)
    df["ASSETS_CODE"] = df["ASSETS"].apply(assets_encode)
    df["HOUSING_CODE"] = df["HOUSING"].apply(housing_encode)
    df["MARITAL_STATUS_CODE"] = df["MARITAL_STATUS"].apply(marital_status_encode)
    df["OTHER_PAYMENT_PLANS_CODE"] = df["OTHER_PAYMENT_PLANS"].apply(other_payment_plans_encode)
    df["SEX_CODE"] = df["SEX"].apply(sex_encode)

    columns_to_drop = ["PURPOSE", "OTHER_PARTIES", "QUALIFICATION", "CREDIT_STANDING",
                       "ASSETS", "HOUSING", "MARITAL_STATUS", "OTHER_PAYMENT_PLANS", "SEX"]
    df = df.drop(columns=columns_to_drop)

    return df

def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('CREDIT_STANDING_CODE', axis=1), 
                                                        df['CREDIT_STANDING_CODE'], test_size=0.30)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train,max_depth,learning_rate,n_estimators):    
    model = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
        use_label_encoder=False
    )    
    model.fit(X_train, y_train)
    return model

def save_model_artifact(pipeline):
    artifact_name = 'model.bst'
    pipeline.save_model(artifact_name)
    model_artifact = bucket.blob('credit-scoring/artifacts/'+artifact_name)
    model_artifact.upload_from_filename(artifact_name)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    confusion_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy,precision,recall
    
input_file = "gs://sid-kubeflow-v1/credit-scoring/credit_files.csv"
credit_df = pd.read_csv(input_file)
credit_df = preprocess_data(credit_df)

X_train, X_test, y_train, y_test = split_data(credit_df)

max_depth=5
learning_rate=0.2
n_estimators=40
pipeline = train_model(X_train, y_train,max_depth,learning_rate,n_estimators)

accuracy,precision,recall = evaluate_model(pipeline, X_test, y_test)

if accuracy>0.5 and precision>0.5 :
    save_model_artifact(pipeline)
    model_validation="true"
else :
    model_validation="false"