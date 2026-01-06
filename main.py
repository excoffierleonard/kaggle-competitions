"""AutoGluon pipeline for Kaggle Playground S6E1."""

import zipfile

from dotenv import load_dotenv

load_dotenv()

import kaggle
from pandas import DataFrame
from autogluon.tabular import TabularDataset, TabularPredictor

COMPETITION = "playground-series-s6e1"
TARGET = "exam_score"

PRESET = "high_v150"
TIME_LIMIT = None

SUBMIT = False


def download_data() -> None:
    """Download competition data from Kaggle."""
    kaggle.api.competition_download_files(COMPETITION, path="data")
    with zipfile.ZipFile(f"data/{COMPETITION}.zip", "r") as z:
        z.extractall("data")


def load_data() -> tuple[DataFrame, DataFrame, DataFrame]:
    """Load train, test, and submission DataFrames."""
    train = TabularDataset("data/train.csv").drop(columns=["id"])
    test = TabularDataset("data/test.csv").drop(columns=["id"])
    sub = TabularDataset("data/sample_submission.csv")
    return train, test, sub


def train_model(train: DataFrame) -> TabularPredictor:
    """Train the AutoGluon predictor."""
    predictor = TabularPredictor(
        label=TARGET,
        problem_type="regression",
        eval_metric="rmse",
    )
    predictor.fit(
        train,
        presets=PRESET,
        time_limit=TIME_LIMIT,
    )

    return predictor


def submit_to_kaggle():
    """Submit the generated submission file to Kaggle."""
    kaggle.api.competition_submit(
        "data/submission.csv", "AutoGluon Pipeline", COMPETITION
    )


def main() -> None:
    """Main entry point."""
    download_data()
    train, test, sub = load_data()

    # Train model
    predictor = train_model(train)

    # Evaluate
    print(predictor.leaderboard())
    print(predictor.feature_importance(train))

    # Predict & clip to training range
    preds = predictor.predict(test)
    sub[TARGET] = preds.clip(train[TARGET].min(), train[TARGET].max())
    sub.to_csv("data/submission.csv", index=False)

    # Submit
    if SUBMIT:
        submit_to_kaggle()

    print("Done!")


if __name__ == "__main__":
    main()
