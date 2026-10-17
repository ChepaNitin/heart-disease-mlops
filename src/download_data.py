from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


DATASET_ID = 45
OUTPUT_PATH = Path("data/raw/heart_disease.csv")


def download_data() -> None:
    """Download the UCI Heart Disease dataset and save the raw data locally."""
    heart_disease = fetch_ucirepo(id=DATASET_ID)

    features = heart_disease.data.features
    target = heart_disease.data.targets

    data = pd.concat([features, target], axis=1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset saved to {OUTPUT_PATH}")
    print(f"Shape: {data.shape}")


def main() -> None:
    download_data()


if __name__ == "__main__":
    main()