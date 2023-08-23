import pytest
from main import app, preprocess

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict(client):
    input_data = {
        "destination": "No Urgent Place",
        "passanger": "Kid(s)",
        "weather": "Sunny",
        "temperature": 80,
        "time": "10AM",
        "coupon": "Bar",
        "expiration": "1d",
        "gender": "Female",
        "age": "21",
        "maritalStatus": "Unmarried partner",
        "has_children": 1,
        "education": "Some college - no degree",
        "occupation": "Unemployed",
        "income": "$37500 - $49999",
        "Bar": "never",
        "CoffeeHouse": "never",
        "CarryAway": "4~8",
        "RestaurantLessThan20": "4~8",
        "Restaurant20To50": "1~3",
        "toCoupon_GEQ15min": 1,
        "toCoupon_GEQ25min": 0,
        "direction_same": 0
    }
    response = client.post('/predict', json=input_data)
    print(response.status_code)
    print(response.json)
    assert response.status_code == 200
    assert response.json["predictions"][0] in [0, 1]

def test_predict_failure(client):
    input_data = {
        "destination": ["No Urgent Place"],
        "passanger": ["Kid(s)"],
        "weather": ["Sunny"],
        "temperature": [80],
        "time": ["10AM"],
        "coupon": ["Bar"],
        "expiration": ["1d"],
        "gender": ["Female"],
        "age": ["21"],
        "maritalStatus": ["Unmarried partner"],
        "has_children": [1],
        "education": ["Some college - no degree"],
        "occupation": ["Unemployed"],
        "income": ["$37500 - $49999"],
        "Bar": ["never"]
    }
    response = client.post('/predict', json=input_data)
    print(response.status_code)
    print(response.json)
    assert response.status_code == 400

def test_preprocess():
    input_data = {
        "destination": ["No Urgent Place"],
        "passanger": ["Kid(s)"],
        "weather": ["Sunny"],
        "temperature": [80],
        "time": ["10AM"],
        "coupon": ["Bar"],
        "expiration": ["1d"],
        "gender": ["Female"],
        "age": ["21"],
        "maritalStatus": ["Unmarried partner"],
        "has_children": [1],
        "education": ["Some college - no degree"],
        "occupation": ["Unemployed"],
        "income": ["$37500 - $49999"],
        "Bar": ["never"],
        "CoffeeHouse": ["never"],
        "CarryAway": ["4~8"],
        "RestaurantLessThan20": ["4~8"],
        "Restaurant20To50": ["1~3"],
        "toCoupon_GEQ15min": [1],
        "toCoupon_GEQ25min": [0],
        "direction_same": [0]
    }
    df = preprocess(input_data)
    assert len(df.columns) == 39