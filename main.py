"""AutoGluon pipeline for Kaggle Playground S6E1."""

import zipfile
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "playground-series-s6e1"
TARGET = "exam_score"

# Toggle feature engineering on/off for benchmarking
USE_ENGINEERED_FEATURES = False


def engineer_features(df):
    """Create interaction and derived features.

    AutoGluon automatically handles:
    - Categorical encoding (maps to integers)
    - Missing values
    - Datetime expansion

    AutoGluon does NOT create:
    - Numerical interactions (we add these)
    - Polynomial features (we add these)
    - Ratio features (we add these)
    """
    df = df.copy()

    # Ordinal encodings for quality score calculation
    # (AutoGluon handles original categoricals, these are for interactions)
    sleep_map = {"poor": 1, "average": 2, "good": 3}
    facility_map = {"low": 1, "medium": 2, "high": 3}
    df["sleep_ord"] = df["sleep_quality"].map(sleep_map)
    df["facility_ord"] = df["facility_rating"].map(facility_map)
    df["quality_score"] = df["sleep_ord"] + df["facility_ord"]

    # Interaction features (numerical * numerical)
    df["study_attend"] = df["study_hours"] * df["class_attendance"]
    df["study_sleep"] = df["study_hours"] * df["sleep_hours"]
    df["study_quality"] = df["study_hours"] * df["quality_score"]
    df["attend_quality"] = df["class_attendance"] * df["quality_score"]

    # Polynomial features
    df["study_hours_sq"] = df["study_hours"] ** 2
    df["attendance_sq"] = df["class_attendance"] ** 2
    df["study_attend_sqrt"] = np.sqrt(df["study_attend"])

    # Ratio features
    df["study_per_sleep"] = df["study_hours"] / (df["sleep_hours"] + 0.1)
    df["efficiency"] = df["study_attend"] / (df["study_hours"] + 0.1)

    return df


# Download data
api = KaggleApi()
api.authenticate()
api.competition_download_files(COMPETITION, path="data")
with zipfile.ZipFile(f"data/{COMPETITION}.zip", "r") as z:
    z.extractall("data")

# Load data
train = TabularDataset("data/train.csv").drop(columns=["id"])
test = TabularDataset("data/test.csv").drop(columns=["id"])
sub = TabularDataset("data/sample_submission.csv")

# Apply feature engineering based on toggle
if USE_ENGINEERED_FEATURES:
    train = engineer_features(train)
    test = engineer_features(test)

# Train
predictor = TabularPredictor(
    label=TARGET, problem_type="regression", eval_metric="rmse"
).fit(
    train,
    # presets="extreme",
    time_limit=600,
)

print(predictor.leaderboard())
print(predictor.feature_importance(train))

# Predict & clip to training range
preds = predictor.predict(test)
sub[TARGET] = preds.clip(train[TARGET].min(), train[TARGET].max())
sub.to_csv("data/submission.csv", index=False)

# Submit
# api.competition_submit("data/submission.csv", "Prediction Pipipeline", COMPETITION)
print("Done!")
