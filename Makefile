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

models/baseline_logreg.joblib: $(TRAIN) $(VAL) $(TEST)
	PYTHONPATH=src uv run python -m mlproj.models.train_baseline



predict-baseline: $(OUT)

$(OUT): $(INPUT) models/baseline_logreg.joblib
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.inference.predict_baseline --input $(INPUT) --out $@ --threshold $(THRESH)


.PHONY: clean clean-preds clean-runs clean-data

clean-preds:
	rm -f reports/predictions_*.csv

clean-runs:
	rm -rf reports/runs models/runs
	@rm -f reports/latest_baseline_metrics.* reports/latest_*.json reports/latest_*.md models/latest_baseline_logreg.joblib || true

clean-data:
	@rm -f data/raw/uci_heart_disease.csv || true
	@rm -f data/processed/*.csv || true

clean: clean-preds clean-runs clean-data
