import pandas as pd
from sklearn.compose import ColumnTransformer

from src.preprocess import create_preprocessor


def test_create_preprocessor_returns_column_transformer() -> None:
    """Check that the preprocessing factory returns a ColumnTransformer."""
    preprocessor = create_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)


def test_preprocessor_transforms_sample_data() -> None:
    """Check that the preprocessor can fit and transform raw feature data."""
    sample_data = pd.DataFrame(
        {
            "age": [63, 45],
            "sex": [1, 0],
            "cp": [1, 2],
            "trestbps": [145, 130],
            "chol": [233, 250],
            "fbs": [1, 0],
            "restecg": [2, 0],
            "thalach": [150, 170],
            "exang": [0, 0],
            "oldpeak": [2.3, 1.2],
            "slope": [3, 2],
            "ca": [0, None],
            "thal": [6, 3],
        }
    )

    preprocessor = create_preprocessor()

    transformed_data = preprocessor.fit_transform(sample_data)

    assert transformed_data.shape[0] == 2