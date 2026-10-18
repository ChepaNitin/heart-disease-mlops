from src.download_data import download_data
from src.eda import run_eda
from src.train import train_models


def main() -> None:
    """Run the complete Heart Disease ML workflow."""
    print("=" * 60)
    print("Heart Disease MLOps Pipeline")
    print("=" * 60)

    print("\n[1/3] Downloading dataset...")
    download_data()

    print("\n[2/3] Running data preparation and EDA...")
    run_eda()

    print("\n[3/3] Training and evaluating models...")
    train_models()

    print("\n" + "=" * 60)
    print("Pipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()