from fastapi.testclient import TestClient
from src.serving.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict():
    r = client.post("/predict", json={"text": "I love this!"})
    assert r.status_code == 200
    body = r.json()
    assert "label" in body and body["label"] in {"positive", "neutral", "negative"}
    assert 0.0 <= body.get("score", 0.0) <= 1.0
