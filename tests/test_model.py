from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_model.pkl"


def create_sample_patient() -> pd.DataFrame:
    """Create one valid raw patient record for model testing."""
    return pd.DataFrame(
        [
            {
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
        ]
    )


def test_saved_model_loads_as_pipeline() -> None:
    """Check that the packaged model can be loaded as a Pipeline."""
    model = joblib.load(MODEL_PATH)

    assert isinstance(model, Pipeline)


def test_saved_model_makes_prediction() -> None:
    """Check that the packaged model predicts from raw patient data."""
    model = joblib.load(MODEL_PATH)
    patient = create_sample_patient()

    prediction = model.predict(patient)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_saved_model_returns_probability() -> None:
    """Check that the packaged model returns a valid probability."""
    model = joblib.load(MODEL_PATH)
    patient = create_sample_patient()

    probability = model.predict_proba(patient)[0, 1]

    assert 0.0 <= probability <= 1.0