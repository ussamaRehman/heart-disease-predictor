.PHONY: setup fmt lint type test check data check-data preprocess split pipeline train-baseline predict-baseline

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

models/baseline_logreg.joblib: $(TRAIN) $(VAL) $(TEST)
	PYTHONPATH=src uv run python -m mlproj.models.train_baseline

predict-baseline: preprocess split models/baseline_logreg.joblib
	PYTHONPATH=src uv run python -m mlproj.inference.predict_baseline --input $(INPUT) --out $(OUT) --threshold $(THRESH)
