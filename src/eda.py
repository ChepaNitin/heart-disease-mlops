from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "heart_disease_clean.csv"
)
EDA_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "eda"

CONTINUOUS_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]


def load_data() -> pd.DataFrame:
    """Load the raw Heart Disease dataset."""
    return pd.read_csv(RAW_DATA_PATH)


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Create the binary target and prepare a cleaned EDA dataset."""
    cleaned_data = data.copy()

    cleaned_data["target"] = (
        cleaned_data["num"] > 0
    ).astype(int)

    cleaned_data["ca"] = cleaned_data["ca"].fillna(
        cleaned_data["ca"].median()
    )
    cleaned_data["thal"] = cleaned_data["thal"].fillna(
        cleaned_data["thal"].median()
    )

    cleaned_data = cleaned_data.drop(columns=["num"])

    return cleaned_data


def save_class_balance_plot(data: pd.DataFrame) -> Path:
    """Create and save the target class-balance plot."""
    output_path = EDA_ARTIFACT_DIR / "class_balance.png"

    plt.figure(figsize=(6, 4))
    sns.countplot(
        data=data,
        x="target",
    )
    plt.title("Heart Disease Class Balance")
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def save_histograms(data: pd.DataFrame) -> Path:
    """Create and save histograms for continuous features."""
    output_path = EDA_ARTIFACT_DIR / "feature_histograms.png"

    data[CONTINUOUS_FEATURES].hist(
        figsize=(12, 8),
        bins=20,
    )

    plt.suptitle(
        "Distribution of Continuous Features",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def save_correlation_heatmap(data: pd.DataFrame) -> Path:
    """Create and save a feature correlation heatmap."""
    output_path = EDA_ARTIFACT_DIR / "correlation_heatmap.png"

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        data.corr(numeric_only=True),
        cmap="coolwarm",
        center=0,
    )

    plt.title("Heart Disease Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def run_eda() -> None:
    """Run reproducible data preparation and EDA artifact generation."""
    data = load_data()

    print("\nEDA Summary")
    print("-----------")
    print(f"Raw shape: {data.shape}")
    print(f"Duplicate rows: {data.duplicated().sum()}")

    missing_values = data.isna().sum()
    missing_values = missing_values[missing_values > 0]

    print("\nMissing values:")
    if missing_values.empty:
        print("None")
    else:
        print(missing_values.to_string())

    cleaned_data = prepare_data(data)

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    EDA_ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_data.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    class_balance_path = save_class_balance_plot(
        cleaned_data
    )
    histogram_path = save_histograms(
        cleaned_data
    )
    heatmap_path = save_correlation_heatmap(
        cleaned_data
    )

    print(
        f"\nProcessed data saved to: "
        f"{PROCESSED_DATA_PATH}"
    )
    print("EDA artifacts saved:")
    print(f"- {class_balance_path}")
    print(f"- {histogram_path}")
    print(f"- {heatmap_path}")


def main() -> None:
    run_eda()


if __name__ == "__main__":
    main()