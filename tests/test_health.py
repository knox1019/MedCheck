from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_has_required_fields():
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "env" in data
    assert "version" in data
