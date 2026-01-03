"""AutoGluon pipeline for Kaggle Playground S6E1."""

import zipfile
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "playground-series-s6e1"
TARGET = "exam_score"


def engineer_features(df):
    """Create interaction and derived features."""
    df = df.copy()

    # Ordinal encodings for categorical features with clear ordering
    sleep_map = {"poor": 1, "average": 2, "good": 3}
    method_map = {
        "self-study": 1,
        "online videos": 2,
        "group study": 3,
        "mixed": 4,
        "coaching": 5,
    }
    facility_map = {"low": 1, "medium": 2, "high": 3}

    df["sleep_ord"] = df["sleep_quality"].map(sleep_map)
    df["method_ord"] = df["study_method"].map(method_map)
    df["facility_ord"] = df["facility_rating"].map(facility_map)

    # Combined quality score (sum of ordinal features)
    df["quality_score"] = df["sleep_ord"] + df["method_ord"] + df["facility_ord"]

    # Key interaction features (highest correlations with target)
    df["study_attend"] = df["study_hours"] * df["class_attendance"]
    df["study_quality"] = df["study_hours"] * df["quality_score"]
    df["study_attend_quality"] = df["study_attend"] * df["quality_score"]

    # Additional useful interactions
    df["study_sleep"] = df["study_hours"] * df["sleep_hours"]
    df["attend_quality"] = df["class_attendance"] * df["quality_score"]

    # Polynomial features for top predictors
    df["study_hours_sq"] = df["study_hours"] ** 2
    df["study_attend_sqrt"] = np.sqrt(df["study_attend"])

    # Ratio features
    df["study_per_sleep"] = df["study_hours"] / (df["sleep_hours"] + 0.1)

    # Boolean flags for high-performing combinations
    df["good_sleep"] = (df["sleep_quality"] == "good").astype(int)
    df["uses_coaching"] = (df["study_method"] == "coaching").astype(int)
    df["high_facility"] = (df["facility_rating"] == "high").astype(int)
    df["optimal_combo"] = df["good_sleep"] * df["uses_coaching"] * df["high_facility"]

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

# Apply feature engineering
train = engineer_features(train)
test = engineer_features(test)

# Train
predictor = TabularPredictor(label=TARGET, eval_metric="rmse").fit(
    train,
    presets="extreme",
    time_limit=3600,
)

# Predict & clip to training range
preds = predictor.predict(test)
sub[TARGET] = preds.clip(train[TARGET].min(), train[TARGET].max())
sub.to_csv("data/submission.csv", index=False)

# Submit
api.competition_submit("data/submission.csv", "Prediction Pipipeline", COMPETITION)
print("Done!")
