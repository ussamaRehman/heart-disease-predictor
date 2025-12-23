.PHONY: setup fmt lint type test check data check-data preprocess split pipeline train-baseline predict-baseline

# Defaults for inference
INPUT ?= data/processed/test.csv
OUT ?= reports/predictions_baseline.csv
THRESH ?= 0.5


setup:
	uv sync --dev

data: data/raw/uci_heart_disease.csv

preprocess: data/processed/heart.csv

split: data/processed/train.csv data/processed/val.csv data/processed/test.csv

train-baseline: models/baseline_logreg.joblib

predict-baseline: $(INPUT) models/baseline_logreg.joblib
	PYTHONPATH=src uv run python -m mlproj.inference.predict_baseline --input $(INPUT) --out $(OUT) --threshold $(THRESH)
