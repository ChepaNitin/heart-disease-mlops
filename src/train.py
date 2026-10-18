from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import create_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_model.pkl"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

RANDOM_STATE = 42
EXPERIMENT_NAME = "Heart Disease Classification"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the raw dataset and prepare features and binary target."""
    data = pd.read_csv(DATA_PATH)

    X = data.drop(columns=["num"])
    y = (data["num"] > 0).astype(int)

    return X, y


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate a trained model on the test set."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def save_evaluation_plots(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> list[Path]:
    """Create and save evaluation plots for a trained model."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.lower().replace(" ", "_")

    confusion_matrix_path = (
        ARTIFACT_DIR / f"{safe_name}_confusion_matrix.png"
    )
    roc_curve_path = (
        ARTIFACT_DIR / f"{safe_name}_roc_curve.png"
    )

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    plt.close()

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )
    plt.title(f"{model_name} - ROC Curve")
    plt.tight_layout()
    plt.savefig(roc_curve_path)
    plt.close()

    return [
        confusion_matrix_path,
        roc_curve_path,
    ]


def train_models() -> None:
    """Tune, evaluate, track, and save candidate classification models."""
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model_configs = {
        "Logistic Regression": {
            "classifier": LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            "parameters": {
                "classifier__C": [0.1, 1.0, 10.0],
            },
        },
        "Random Forest": {
            "classifier": RandomForestClassifier(
                random_state=RANDOM_STATE,
            ),
            "parameters": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5],
            },
        },
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results: list[dict[str, object]] = []

    for model_name, config in model_configs.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("classifier", config["classifier"]),
            ]
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=config["parameters"],
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        metrics = evaluate_model(
            best_model,
            X_test,
            y_test,
        )

        with mlflow.start_run(
            run_name=f"{model_name} - Tuned"
        ):
            mlflow.log_param(
                "model_name",
                model_name,
            )
            mlflow.log_param(
                "random_state",
                RANDOM_STATE,
            )
            mlflow.log_param(
                "cv_folds",
                5,
            )

            for parameter_name, value in (
                grid_search.best_params_.items()
            ):
                clean_name = parameter_name.replace(
                    "classifier__",
                    "",
                )

                mlflow.log_param(
                    clean_name,
                    value,
                )

            mlflow.log_metric(
                "cv_roc_auc",
                grid_search.best_score_,
            )

            for metric_name, value in metrics.items():
                mlflow.log_metric(
                    metric_name,
                    value,
                )

            plot_paths = save_evaluation_plots(
                best_model,
                X_test,
                y_test,
                model_name,
            )

            for plot_path in plot_paths:
                mlflow.log_artifact(
                    str(plot_path),
                    artifact_path="evaluation_plots",
                )

            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

            results.append(
                {
                    "model_name": model_name,
                    "cv_roc_auc": grid_search.best_score_,
                    "test_roc_auc": metrics["roc_auc"],
                    "model": best_model,
                }
            )

        print(f"\n{model_name}")
        print("-" * len(model_name))
        print(
            f"Best parameters: "
            f"{grid_search.best_params_}"
        )
        print(
            f"Best 5-Fold CV ROC-AUC: "
            f"{grid_search.best_score_:.4f}"
        )

        for metric_name, value in metrics.items():
            print(
                f"{metric_name}: "
                f"{value:.4f}"
            )

    best_result = max(
        results,
        key=lambda result: result["cv_roc_auc"],
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        best_result["model"],
        MODEL_PATH,
    )

    print("\nFinal Model Selection")
    print("---------------------")
    print(
        f"Selected model: "
        f"{best_result['model_name']}"
    )
    print(
        f"Selection CV ROC-AUC: "
        f"{best_result['cv_roc_auc']:.4f}"
    )
    print(
        f"Final model saved to: "
        f"{MODEL_PATH}"
    )


def main() -> None:
    train_models()


if __name__ == "__main__":
    main()