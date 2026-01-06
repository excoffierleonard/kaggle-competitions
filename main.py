"""AutoGluon pipeline for Kaggle Playground S6E1."""

import os
import time
import zipfile

from dotenv import load_dotenv

load_dotenv()

import kaggle
from pandas import DataFrame
from autogluon.tabular import TabularDataset, TabularPredictor

COMPETITION = "playground-series-s6e1"
TARGET = "exam_score"

PRESETS = os.getenv("PRESETS", "") or "high_v150"
TIME_LIMIT = (
    int(v) if (v := os.getenv("TIME_LIMIT", "")).isdigit() and int(v) > 0 else None
)

SUBMIT = os.getenv("SUBMIT", "").lower() == "true"


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
        presets=PRESETS,
        time_limit=TIME_LIMIT,
    )

    return predictor


def submit_to_kaggle() -> None:
    """Submit the generated submission file to Kaggle."""
    kaggle.api.competition_submit(
        "data/submission.csv", "AutoGluon Pipeline", COMPETITION
    )


def main() -> None:
    """Main entry point."""
    start_time = time.time()

    # Settings
    print(f"PRESETS: {PRESETS}")
    print(f"TIME_LIMIT: {TIME_LIMIT}")
    print(f"SUBMIT: {SUBMIT}\n")

    # Download & load data
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

    elapsed = time.time() - start_time
    print(f"Done! ({elapsed:.2f}s)")


if __name__ == "__main__":
    main()
