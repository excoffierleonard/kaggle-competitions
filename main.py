"""AutoGluon pipeline for Kaggle Playground S6E1."""

import zipfile

import numpy as np
from pandas import DataFrame
from autogluon.common.features.types import R_FLOAT, R_INT
from autogluon.features.generators import (
    AbstractFeatureGenerator,
    AutoMLPipelineFeatureGenerator,
)
from autogluon.tabular import TabularDataset, TabularPredictor
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "playground-series-s6e1"
TARGET = "exam_score"
PRESET = "medium"
TIME_LIMIT = 600

# Toggle feature engineering on/off for benchmarking
USE_ENGINEERED_FEATURES = True


class InteractionFeatureGenerator(AbstractFeatureGenerator):
    """Custom feature generator for interaction, polynomial, and ratio features.

    AutoGluon automatically handles:
    - Categorical encoding (maps to integers)
    - Missing values
    - Datetime expansion

    This generator adds what AutoGluon does NOT create:
    - Numerical interactions
    - Polynomial features
    - Ratio features
    """

    def _fit_transform(self, X: DataFrame, **kwargs) -> tuple[DataFrame, dict]:
        X_out = self._transform(X)
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        X = X.copy()

        # Interaction features (numerical * numerical)
        X["study_attend"] = X["study_hours"] * X["class_attendance"]
        X["study_sleep"] = X["study_hours"] * X["sleep_hours"]

        # Polynomial features
        X["study_hours_sq"] = X["study_hours"] ** 2
        X["attendance_sq"] = X["class_attendance"] ** 2
        X["study_attend_sqrt"] = np.sqrt(X["study_attend"])

        # Ratio features
        X["study_per_sleep"] = X["study_hours"] / (X["sleep_hours"] + 0.1)
        X["efficiency"] = X["study_attend"] / (X["study_hours"] + 0.1)

        return X

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return {"valid_raw_types": [R_FLOAT, R_INT]}


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


def create_feature_generator() -> AutoMLPipelineFeatureGenerator | None:
    """Create the feature generator based on toggle."""
    if not USE_ENGINEERED_FEATURES:
        return None
    return AutoMLPipelineFeatureGenerator(
        pre_generators=[InteractionFeatureGenerator(verbosity=0)]
    )


def train_model(train: DataFrame, feature_generator) -> TabularPredictor:
    """Train the AutoGluon predictor."""
    predictor = TabularPredictor(
        label=TARGET,
        preset=PRESET,
        problem_type="regression",
        eval_metric="rmse",
    )
    predictor.fit(
        train,
        feature_generator=feature_generator,
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

    feature_generator = create_feature_generator()
    print(f"Feature engineering: {'ON' if USE_ENGINEERED_FEATURES else 'OFF'}")

    predictor = train_model(train, feature_generator)

    print(predictor.leaderboard())
    print(predictor.feature_importance(train))

    # Predict & clip to training range
    preds = predictor.predict(test)
    sub[TARGET] = preds.clip(train[TARGET].min(), train[TARGET].max())
    sub.to_csv("data/submission.csv", index=False)

    # Submit
    # submit_to_kaggle()

    print("Done!")


if __name__ == "__main__":
    main()
