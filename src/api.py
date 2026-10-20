import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_model.pkl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
)

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
)


class PatientData(BaseModel):
    """Input features required for heart disease prediction."""

    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float | None = None
    thal: float | None = None


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Log API requests and collect request metrics."""
    start_time = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start_time
    endpoint = request.url.path

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=response.status_code,
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=endpoint,
    ).observe(duration)

    logger.info(
        "%s %s | status=%s | duration=%.4fs",
        request.method,
        endpoint,
        response.status_code,
        duration,
    )

    return response


@app.get("/")
def root() -> dict[str, str]:
    """Return API status information."""
    return {
        "message": "Heart Disease Prediction API",
        "status": "running",
    }


@app.post("/predict")
def predict(
    patient: PatientData,
) -> dict[str, int | float | str]:
    """Predict heart disease and return the model confidence."""
    patient_data = pd.DataFrame(
        [patient.model_dump()]
    )

    prediction = int(model.predict(patient_data)[0])
    probabilities = model.predict_proba(patient_data)[0]
    confidence = float(probabilities[prediction])

    prediction_label = (
        "heart disease"
        if prediction == 1
        else "no heart disease"
    )

    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "confidence": confidence,
    }


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics for API monitoring."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )