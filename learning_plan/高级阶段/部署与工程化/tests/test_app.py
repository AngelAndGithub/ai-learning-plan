
import pytest
from app import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AI Model API"}

def test_predict():
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0, 4.0]})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()
