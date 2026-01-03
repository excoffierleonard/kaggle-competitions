"""AutoGluon pipeline for Kaggle Playground S6E1."""

import zipfile
from autogluon.tabular import TabularDataset, TabularPredictor
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "playground-series-s6e1"
TARGET = "exam_score"

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
