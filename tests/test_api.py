"""Integration tests for the FastAPI service."""

from fastapi.testclient import TestClient

from src.retinocare.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_non_image_content_type():
    response = client.post(
        "/predict", files={"file": ("notes.txt", b"just some text", "text/plain")}
    )
    assert response.status_code == 400
    assert "PNG" in response.json()["detail"]


def test_predict_rejects_corrupted_image_bytes():
    response = client.post(
        "/predict", files={"file": ("fake.png", b"not-a-real-png-file", "image/png")}
    )
    assert response.status_code == 400
    assert "Could not read image" in response.json()["detail"]