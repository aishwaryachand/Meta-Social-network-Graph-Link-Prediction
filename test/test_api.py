from fastapi.testclient import TestClient
from src.serving.api import app

client = TestClient(app)

def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()

def test_score_endpoint():
    r = client.post("/score", json={"pairs": [[1, 2], [2, 3]]})
    assert r.status_code == 200
    assert "scores" in r.json()
