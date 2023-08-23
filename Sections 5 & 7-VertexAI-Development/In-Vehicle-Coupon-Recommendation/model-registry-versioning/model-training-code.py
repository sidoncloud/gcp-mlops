import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from category_encoders import HashingEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket("sid-kubeflow-v1")

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    
    df = df.drop(columns=['car', 'toCoupon_GEQ5min', 'direction_opp'])
    df = df.fillna(df.mode().iloc[0])
    df = df.drop_duplicates()

    df_dummy = df.copy()
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
    df_dummy = df_dummy.drop(columns=['passanger', 'destination', 'maritalStatus', 'has_children', 'temperature','weather', 'Y'])

    df_dummy = pd.concat([df_dummy, df['Y']], axis = 1)
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

    x = df_le.drop('Y', axis=1)
    y = df_le.Y

    return x, y

def train_model(x_train, y_train,max_depth,learning_rate,n_estimators):    
    model = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
        use_label_encoder=False
    )
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test, x_sm_train_hashing, y_sm_train):
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)
    y_pred_train = model.predict(x_sm_train_hashing)
    y_pred_train_proba = model.predict_proba(x_sm_train_hashing)

    print('accuracy (test): ' + str(accuracy_score(y_test, y_pred)))
    print('precision (test): ' + str(precision_score(y_test, y_pred)))
    print('recall (test): ' + str(recall_score(y_test, y_pred)))
    print('roc-auc (train-proba): ' + str(roc_auc_score(y_sm_train, y_pred_train_proba[:, 1])))
    print('roc-auc (test-proba): ' + str(roc_auc_score(y_test, y_pred_proba[:, 1])))

def encode_features(x, n_components=27):
    hashing_ros_enc = HashingEncoder(cols=['passanger_destination', 'marital_hasChildren', 'occupation', 'coupon',
                                           'temperature_weather'], n_components=n_components).fit(x)
    x_test_hashing = hashing_ros_enc.transform(x.reset_index(drop=True))
    return x_test_hashing

def oversample_data(x_train_hashing, y_train):
    sm = SMOTE(random_state=42)
    x_sm_train_hashing, y_sm_train = sm.fit_resample(x_train_hashing, y_train)
    return x_sm_train_hashing, y_sm_train

def get_score(model, x, y, x_test, y_test):
    model.fit(x, y)
    y_pred = model.predict_proba(x_test)[:, 1]
    score = roc_auc_score(y_test, y_pred)
    return score

def save_model_artifact(pipeline):    
    artifact_name = 'model.bst'
    pipeline.save_model(artifact_name)
    model_artifact = bucket.blob('coupon-recommendation/artifacts/'+artifact_name)
    model_artifact.upload_from_filename(artifact_name)

input_file = "gs://sid-kubeflow-v1/coupon-recommendation/in-vehicle-coupon-recommendation.csv"
df = load_data(input_file)

# Preprocess data
x, y = preprocess_data(df)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Impute categorical values
x_train.fillna(x_train.mode().iloc[0], inplace=True)
x_test.fillna(x_train.mode().iloc[0], inplace=True)

model_name = 'xgboost'
print("Training and evaluating", model_name, "model:")
x_train_hashing = encode_features(x_train)
x_test_hashing = encode_features(x_test)
x_sm_train_hashing, y_sm_train = oversample_data(x_train_hashing,y_train)

max_depth=5
learning_rate=0.2
n_estimators=40
pipeline = train_model(x_sm_train_hashing,y_sm_train,max_depth,learning_rate,n_estimators)

evaluate_model(pipeline,x_test_hashing,y_test,x_sm_train_hashing,y_sm_train)

save_model_artifact(pipeline)