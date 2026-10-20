from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_root_endpoint() -> None:
    """Check that the API root endpoint is available."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_predict_endpoint() -> None:
    """Check that the prediction endpoint returns a valid prediction."""
    patient = {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }

    response = client.post(
        "/predict",
        json=patient,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert result["prediction_label"] in [
        "no heart disease",
        "heart disease",
    ]
    assert 0.0 <= result["confidence"] <= 1.0

def test_metrics_endpoint() -> None:
    """Check that Prometheus metrics are exposed."""
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "api_requests_total" in response.text
    assert "api_request_duration_seconds" in response.text