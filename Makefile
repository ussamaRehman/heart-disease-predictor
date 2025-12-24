.PHONY: setup fmt lint type test check data check-data preprocess split pipeline train-baseline ml predict-baseline clean-preds eval-baseline

# Defaults for inference
INPUT ?= data/processed/test.csv
OUT ?= reports/predictions_baseline.csv
THRESH ?= 0.5

EVAL_INPUT ?= data/processed/test.csv
EVAL_PREDS ?= reports/predictions_baseline.csv
EVAL_OUT ?= reports/latest_eval.json

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

eval-baseline: $(EVAL_OUT)

$(EVAL_OUT): $(EVAL_INPUT) $(EVAL_PREDS)
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.evaluation.eval_predictions --input $(EVAL_INPUT) --preds $(EVAL_PREDS) --out $@

