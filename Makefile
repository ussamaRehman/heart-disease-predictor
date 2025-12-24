.PHONY: setup fmt lint type test check data check-data preprocess split pipeline train-baseline

# Defaults for inference
INPUT ?= data/processed/test.csv
OUT ?= reports/predictions_baseline.csv
THRESH ?= 0.5

RAW ?= data/raw/uci_heart_disease.csv
PROCESSED ?= data/processed/heart.csv
TRAIN ?= data/processed/train.csv
VAL ?= data/processed/val.csv
TEST ?= data/processed/test.csv


setup:
	uv sync --dev

data: data/raw/uci_heart_disease.csv

preprocess: $(PROCESSED)

split: $(TRAIN) $(VAL) $(TEST)

models/baseline_logreg.joblib: data/processed/train.csv data/processed/val.csv
	PYTHONPATH=src uv run python -m mlproj.models.train_baseline
