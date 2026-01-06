"""AutoGluon pipeline for Kaggle Playground S6E1."""

import zipfile

import numpy as np
from pandas import DataFrame
from autogluon.tabular import TabularDataset, TabularPredictor
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "playground-series-s6e1"
TARGET = "exam_score"

USE_ENGINEERED_FEATURES = False
FAST_MODE = True

PRESET = "high_v150"
TIME_LIMIT = 600

SUBMIT = False


def engineer_features(df: DataFrame) -> DataFrame:
    """Create interaction, polynomial, and ratio features.

    AutoGluon automatically handles:
    - Categorical encoding (maps to integers)
    - Missing values
    - Datetime expansion

    This function adds what AutoGluon does NOT create:
    - Numerical interactions
    - Polynomial features
    - Ratio features
    """
    df = df.copy()

    # Interaction features (numerical * numerical)
    df["study_attend"] = df["study_hours"] * df["class_attendance"]
    df["study_sleep"] = df["study_hours"] * df["sleep_hours"]

    # Polynomial features
    df["study_hours_sq"] = df["study_hours"] ** 2
    df["attendance_sq"] = df["class_attendance"] ** 2
    df["study_attend_sqrt"] = np.sqrt(df["study_attend"])

    # Ratio features
    df["study_per_sleep"] = df["study_hours"] / (df["sleep_hours"] + 0.1)
    df["efficiency"] = df["study_attend"] / (df["study_hours"] + 0.1)

    return df


def download_data() -> None:
    """Download competition data from Kaggle."""
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(COMPETITION, path="data")
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

    if FAST_MODE:
        predictor.fit(
            train,
            hyperparameters={"CAT": {}},
            time_limit=60,
        )
    else:
        predictor.fit(
            train,
            presets=PRESET,
            time_limit=TIME_LIMIT,
        )
    return predictor


def submit_to_kaggle():
    """Submit the generated submission file to Kaggle."""
    api = KaggleApi()
    api.authenticate()
    api.competition_submit("data/submission.csv", "AutoGluon Pipeline", COMPETITION)


def main() -> None:
    """Main entry point."""
    download_data()
    train, test, sub = load_data()

    if USE_ENGINEERED_FEATURES:
        train = engineer_features(train)
        test = engineer_features(test)

    predictor = train_model(train)

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
