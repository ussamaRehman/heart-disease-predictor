.PHONY: setup fmt lint type test check data check-data preprocess split pipeline train-baseline ml predict-baseline clean-preds

# Defaults for inference
INPUT ?= data/processed/test.csv
OUT ?= reports/predictions_baseline.csv
THRESH ?= 0.5

setup:
	uv sync --dev

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

type:
	uv run pyright

test:
	PYTHONPATH=src uv run pytest

check: fmt lint type test

data: data/raw/uci_heart_disease.csv

data/raw/uci_heart_disease.csv:
	uv run python scripts/download_uci_heart_disease.py

check-data: data/raw/uci_heart_disease.csv
	uv run python scripts/check_data.py

preprocess: data/processed/heart.csv

data/processed/heart.csv: data/raw/uci_heart_disease.csv
	PYTHONPATH=src uv run python -m mlproj.data.make_dataset

split: data/processed/train.csv data/processed/val.csv data/processed/test.csv

data/processed/train.csv data/processed/val.csv data/processed/test.csv: data/processed/heart.csv
	PYTHONPATH=src uv run python -m mlproj.data.split

pipeline: data check-data preprocess split check

# IMPORTANT: model depends ONLY on train/val (not test)
models/baseline_logreg.joblib: data/processed/train.csv data/processed/val.csv
	PYTHONPATH=src uv run python -m mlproj.models.train_baseline

train-baseline: models/baseline_logreg.joblib

ml: pipeline train-baseline

# Inference is cached via OUT
predict-baseline: $(OUT)

$(OUT): $(INPUT) models/baseline_logreg.joblib
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.inference.predict_baseline --input $(INPUT) --out $@ --threshold $(THRESH)

clean-preds:
	rm -f reports/predictions_*.csv
