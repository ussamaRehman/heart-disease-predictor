.PHONY: setup fmt lint type test check data check-data preprocess split pipeline train-baseline ml predict-baseline clean-preds eval-baseline sweep-thresholds predict-baseline-best eval-baseline-best predict-baseline-valtuned eval-baseline-valtuned predict-val val-sweep val-best-threshold predict-baseline-valtuned-auto eval-baseline-valtuned-auto

# Defaults for inference
INPUT ?= data/processed/test.csv
OUT ?= reports/predictions_baseline.csv
THRESH ?= 0.5


EVAL_INPUT ?= data/processed/test.csv
EVAL_PREDS ?= reports/predictions_baseline.csv
EVAL_OUT ?= reports/latest_eval.json

SWEEP_INPUT ?= $(EVAL_INPUT)
SWEEP_PREDS ?= $(EVAL_PREDS)
SWEEP_OUT ?= reports/threshold_sweep.csv
TMIN ?= 0.05
TMAX ?= 0.95
TSTEP ?= 0.05


BEST_INPUT ?= data/processed/test.csv
BEST_THRESH ?= 0.70
BEST_PREDS ?= reports/predictions_baseline_t$(BEST_THRESH).csv
BEST_EVAL_OUT ?= reports/eval_thresh_$(BEST_THRESH).json

VAL_TUNED_THRESH ?= 0.35
VAL_TUNED_INPUT ?= data/processed/test.csv
VAL_TUNED_PREDS ?= reports/predictions_baseline_valtuned_t$(VAL_TUNED_THRESH).csv
VAL_TUNED_EVAL_OUT ?= reports/eval_valtuned_t$(VAL_TUNED_THRESH).json
VAL_SWEEP_INPUT ?= data/processed/val.csv
VAL_SWEEP_PREDS ?= reports/predictions_val.csv
VAL_SWEEP_OUT ?= reports/val_threshold_sweep.csv
VAL_BEST_METRIC ?= f1
VAL_BEST_THRESH_FILE ?= reports/val_best_threshold.txt
VALTUNED_AUTO_INPUT ?= data/processed/test.csv
VALTUNED_AUTO_PREDS ?= reports/predictions_baseline_valtuned_auto.csv
VALTUNED_AUTO_EVAL ?= reports/eval_valtuned_auto.json



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

sweep-thresholds: $(SWEEP_OUT)

$(SWEEP_OUT): $(SWEEP_INPUT) $(SWEEP_PREDS)
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.evaluation.sweep_thresholds --input $(SWEEP_INPUT) --preds $(SWEEP_PREDS) --out $@ --t-min $(TMIN) --t-max $(TMAX) --t-step $(TSTEP)

predict-baseline-best: $(BEST_PREDS)

$(BEST_PREDS): $(BEST_INPUT) models/baseline_logreg.joblib
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.inference.predict_baseline --input $(BEST_INPUT) --out $@ --threshold $(BEST_THRESH)

eval-baseline-best: $(BEST_EVAL_OUT)

$(BEST_EVAL_OUT): $(BEST_INPUT) $(BEST_PREDS)
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.evaluation.eval_predictions --input $(BEST_INPUT) --preds $(BEST_PREDS) --out $@

predict-baseline-valtuned: $(VAL_TUNED_PREDS)

$(VAL_TUNED_PREDS): $(VAL_TUNED_INPUT) models/baseline_logreg.joblib
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.inference.predict_baseline --input $(VAL_TUNED_INPUT) --out $@ --threshold $(VAL_TUNED_THRESH)

eval-baseline-valtuned: $(VAL_TUNED_EVAL_OUT)

$(VAL_TUNED_EVAL_OUT): $(VAL_TUNED_INPUT) $(VAL_TUNED_PREDS)
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.evaluation.eval_predictions --input $(VAL_TUNED_INPUT) --preds $(VAL_TUNED_PREDS) --out $@

# --- val-tuned auto (from val sweep) ---
predict-val: $(VAL_SWEEP_PREDS)

$(VAL_SWEEP_PREDS): $(VAL_SWEEP_INPUT) models/baseline_logreg.joblib
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.inference.predict_baseline --input $(VAL_SWEEP_INPUT) --out $@ --threshold 0.5

val-sweep: $(VAL_SWEEP_OUT)

$(VAL_SWEEP_OUT): $(VAL_SWEEP_INPUT) $(VAL_SWEEP_PREDS)
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.evaluation.sweep_thresholds --input $(VAL_SWEEP_INPUT) --preds $(VAL_SWEEP_PREDS) --out $@ --t-min $(TMIN) --t-max $(TMAX) --t-step $(TSTEP)

val-best-threshold: $(VAL_BEST_THRESH_FILE)

$(VAL_BEST_THRESH_FILE): $(VAL_SWEEP_OUT)
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.evaluation.pick_best_threshold --csv $(VAL_SWEEP_OUT) --metric $(VAL_BEST_METRIC) > $@

predict-baseline-valtuned-auto: $(VALTUNED_AUTO_PREDS)

$(VALTUNED_AUTO_PREDS): $(VALTUNED_AUTO_INPUT) models/baseline_logreg.joblib $(VAL_BEST_THRESH_FILE)
	mkdir -p $(dir $@)
	THRESH=$$(cat $(VAL_BEST_THRESH_FILE)); PYTHONPATH=src uv run python -m mlproj.inference.predict_baseline --input $(VALTUNED_AUTO_INPUT) --out $@ --threshold $$THRESH

eval-baseline-valtuned-auto: $(VALTUNED_AUTO_EVAL)

$(VALTUNED_AUTO_EVAL): $(VALTUNED_AUTO_INPUT) $(VALTUNED_AUTO_PREDS)
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.evaluation.eval_predictions --input $(VALTUNED_AUTO_INPUT) --preds $(VALTUNED_AUTO_PREDS) --out $@
# --- end val-tuned auto ---

