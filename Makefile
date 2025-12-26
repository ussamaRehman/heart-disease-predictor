FINAL_REPORT ?= reports/final_report.md
.PHONY: setup fmt lint type test check data check-data preprocess split pipeline train-baseline ml predict-baseline clean-preds eval-baseline sweep-thresholds predict-baseline-best eval-baseline-best predict-baseline-valtuned eval-baseline-valtuned predict-val val-sweep val-best-threshold predict-baseline-valtuned-auto eval-baseline-valtuned-auto val-tune-baseline clean-val-tune val-tune-baseline-clean val-tune-report val-tune-baseline-report train-rf predict-rf eval-rf sweep-thresholds-rf rf-predict-val rf-val-sweep rf-val-best-threshold predict-rf-valtuned-auto eval-rf-valtuned-auto rf-val-tune-report clean-rf-val-tune val-tune-rf-report compare-models model-compare-report compare-models-report final-report final-report-print

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
VAL_TUNE_REPORT ?= reports/val_tuning_report.md
COMPARE_REPORT ?= reports/model_comparison.md
RF_MODEL_OUT ?= models/rf.joblib
RF_REPORT_OUT ?= reports/rf_metrics.md
RF_INPUT ?= data/processed/test.csv
RF_PREDS ?= reports/predictions_rf.csv
RF_THRESH ?= 0.5
RF_EVAL_OUT ?= reports/eval_rf.json
RF_SWEEP_OUT ?= reports/rf_threshold_sweep.csv
RF_VAL_SWEEP_INPUT ?= data/processed/val.csv
RF_VAL_SWEEP_PREDS ?= reports/predictions_rf_val.csv
RF_VAL_SWEEP_OUT ?= reports/rf_val_threshold_sweep.csv
RF_VAL_BEST_THRESH_FILE ?= reports/rf_val_best_threshold.txt
RF_VALTUNED_AUTO_INPUT ?= data/processed/test.csv
RF_VALTUNED_AUTO_PREDS ?= reports/predictions_rf_valtuned_auto.csv
RF_VALTUNED_AUTO_EVAL ?= reports/eval_rf_valtuned_auto.json
RF_VAL_TUNE_REPORT ?= reports/rf_val_tuning_report.md

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

val-tune-baseline: predict-val val-sweep val-best-threshold predict-baseline-valtuned-auto eval-baseline-valtuned-auto

clean-val-tune:
	rm -f reports/predictions_val.csv reports/val_threshold_sweep.csv reports/val_best_threshold.txt reports/predictions_baseline_valtuned_auto.csv reports/eval_valtuned_auto.json

val-tune-baseline-clean: clean-val-tune val-tune-baseline

val-tune-report: $(VAL_TUNE_REPORT)

$(VAL_TUNE_REPORT): $(VAL_SWEEP_OUT) $(VAL_BEST_THRESH_FILE) $(VALTUNED_AUTO_EVAL)
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.evaluation.write_val_tuning_report --sweep-csv $(VAL_SWEEP_OUT) --metric $(VAL_BEST_METRIC) --threshold-file $(VAL_BEST_THRESH_FILE) --eval-json $(VALTUNED_AUTO_EVAL) --out $@

val-tune-baseline-report: val-tune-baseline-clean val-tune-report


# --- RF TARGETS START ---
train-rf: $(RF_MODEL_OUT)

$(RF_MODEL_OUT): data/processed/train.csv data/processed/val.csv data/processed/test.csv
	mkdir -p $(dir $@)
	mkdir -p reports/
	PYTHONPATH=src uv run python -m mlproj.models.train_rf --model-out $@ --report-out $(RF_REPORT_OUT)

predict-rf: $(RF_PREDS)

$(RF_PREDS): $(RF_INPUT) $(RF_MODEL_OUT)
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.inference.predict_rf --input $(RF_INPUT) --out $@ --model $(RF_MODEL_OUT) --threshold $(RF_THRESH)

eval-rf: $(RF_EVAL_OUT)

$(RF_EVAL_OUT): $(RF_INPUT) $(RF_PREDS)
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.evaluation.eval_predictions --input $(RF_INPUT) --preds $(RF_PREDS) --out $@

sweep-thresholds-rf: $(RF_SWEEP_OUT)

$(RF_SWEEP_OUT): $(RF_INPUT) $(RF_PREDS)
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.evaluation.sweep_thresholds --input $(RF_INPUT) --preds $(RF_PREDS) --out $@ --t-min $(TMIN) --t-max $(TMAX) --t-step $(TSTEP)
# --- RF TARGETS END ---

rf-predict-val: $(RF_VAL_SWEEP_PREDS)

$(RF_VAL_SWEEP_PREDS): $(RF_VAL_SWEEP_INPUT) $(RF_MODEL_OUT)
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.inference.predict_rf --input $(RF_VAL_SWEEP_INPUT) --out $@ --model $(RF_MODEL_OUT) --threshold 0.5

rf-val-sweep: $(RF_VAL_SWEEP_OUT)

$(RF_VAL_SWEEP_OUT): $(RF_VAL_SWEEP_INPUT) $(RF_VAL_SWEEP_PREDS)
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.evaluation.sweep_thresholds --input $(RF_VAL_SWEEP_INPUT) --preds $(RF_VAL_SWEEP_PREDS) --out $@ --t-min $(TMIN) --t-max $(TMAX) --t-step $(TSTEP)

rf-val-best-threshold: $(RF_VAL_BEST_THRESH_FILE)

$(RF_VAL_BEST_THRESH_FILE): $(RF_VAL_SWEEP_OUT)
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.evaluation.pick_best_threshold --csv $(RF_VAL_SWEEP_OUT) --metric $(VAL_BEST_METRIC) > $@

predict-rf-valtuned-auto: $(RF_VALTUNED_AUTO_PREDS)

$(RF_VALTUNED_AUTO_PREDS): $(RF_VALTUNED_AUTO_INPUT) $(RF_MODEL_OUT) $(RF_VAL_BEST_THRESH_FILE)
	mkdir -p $(dir $@)
	THRESH=$$(cat $(RF_VAL_BEST_THRESH_FILE)); PYTHONPATH=src uv run python -m mlproj.inference.predict_rf --input $(RF_VALTUNED_AUTO_INPUT) --out $@ --model $(RF_MODEL_OUT) --threshold $$THRESH

eval-rf-valtuned-auto: $(RF_VALTUNED_AUTO_EVAL)

$(RF_VALTUNED_AUTO_EVAL): $(RF_VALTUNED_AUTO_INPUT) $(RF_VALTUNED_AUTO_PREDS)
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.evaluation.eval_predictions --input $(RF_VALTUNED_AUTO_INPUT) --preds $(RF_VALTUNED_AUTO_PREDS) --out $@

rf-val-tune-report: $(RF_VAL_TUNE_REPORT)

$(RF_VAL_TUNE_REPORT): $(RF_VAL_SWEEP_OUT) $(RF_VAL_BEST_THRESH_FILE) $(RF_VALTUNED_AUTO_EVAL)
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.evaluation.write_val_tuning_report --sweep-csv $(RF_VAL_SWEEP_OUT) --metric $(VAL_BEST_METRIC) --threshold-file $(RF_VAL_BEST_THRESH_FILE) --eval-json $(RF_VALTUNED_AUTO_EVAL) --out $@

clean-rf-val-tune:
	rm -f $(RF_VAL_SWEEP_PREDS) $(RF_VAL_SWEEP_OUT) $(RF_VAL_BEST_THRESH_FILE) $(RF_VALTUNED_AUTO_PREDS) $(RF_VALTUNED_AUTO_EVAL) $(RF_VAL_TUNE_REPORT)

val-tune-rf-report: clean-rf-val-tune rf-predict-val rf-val-sweep rf-val-best-threshold predict-rf-valtuned-auto eval-rf-valtuned-auto rf-val-tune-report


compare-models-report: compare-models
	@echo
	@echo '---- $(COMPARE_REPORT) ----'
	@sed -n '1,200p' $(COMPARE_REPORT)

# --- model-compare targets ---
model-compare-report: $(COMPARE_REPORT)

$(COMPARE_REPORT): reports/eval_valtuned_auto.json reports/val_best_threshold.txt reports/eval_rf_valtuned_auto.json reports/rf_val_best_threshold.txt
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.evaluation.compare_models --metric $(VAL_BEST_METRIC) --baseline-eval reports/eval_valtuned_auto.json --baseline-threshold-file reports/val_best_threshold.txt --rf-eval reports/eval_rf_valtuned_auto.json --rf-threshold-file reports/rf_val_best_threshold.txt --out $@

compare-models: val-tune-baseline-report val-tune-rf-report model-compare-report

compare-models-report: compare-models
	@echo
	@echo "---- $(COMPARE_REPORT) ----"
	@sed -n "1,200p" $(COMPARE_REPORT)
# --- end model-compare targets ---


# --- final-report targets ---
final-report: $(FINAL_REPORT)

$(FINAL_REPORT): reports/val_tuning_report.md reports/rf_val_tuning_report.md $(COMPARE_REPORT)
	mkdir -p $(dir $@)
	PYTHONPATH=src uv run python -m mlproj.evaluation.write_final_report --baseline-report reports/val_tuning_report.md --rf-report reports/rf_val_tuning_report.md --compare-report $(COMPARE_REPORT) --out $@

final-report-print: compare-models-report final-report
	@echo
	@echo "---- $(FINAL_REPORT) ----"
	@sed -n "1,220p" $(FINAL_REPORT)
# --- end final-report targets ---

