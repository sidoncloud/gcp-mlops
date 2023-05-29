import pytest
from flask import Flask

from main import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_main_endpoint(client):
    response = client.get('/')
    assert response.status_code == 200
    data = response.get_json()
    assert 'data' in data
    assert isinstance(data['data'], int)
