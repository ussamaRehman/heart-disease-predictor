# Heart Disease Predictor Model

[![CI](https://github.com/ussamaRehman/heart-disease-predictor/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/ussamaRehman/heart-disease-predictor/actions/workflows/ci.yml)

A production-minded ML project scaffold (env + quality gates + CI) for a heart disease risk prediction model.
Modeling + data pipeline will be added next.

## Quickstart

### 1) Setup

    uv sync --dev

### 2) Run checks (format, lint, types, tests)

    make check

## Repo structure (current)
- `src/` — project code (will contain data/features/models pipeline)
- `tests/` — tests (currently a smoke test)
- `data/` — raw/processed/external (ignored by git; folders kept via `.gitkeep`)
- `models/` — saved artifacts (ignored by git; folder kept via `.gitkeep`)
- `reports/` — figures + documentation outputs (ignored by git; folder kept via `.gitkeep`)
^- `notebooks/` — exploration notebooks (folder kept via `.gitkeep`)
- `docs/` — notes/decisions (folder kept via `.gitkeep`)

## Tooling
- Python: pinned via `.python-version` (pyenv)
- Dependency management: `uv` + `pyproject.toml` + `uv.lock`
- Quality gates: Ruff (format + lint), Pyright (types), Pytest (tests)
- Local gates: pre-commit
- CI: GitHub Actions (`.github/workflows/ci.yml`)

## Notes
This is a portfolio project scaffold. It is not a medical device and is not intended for diagnostic use.


## Run end-to-end (single command)

```bash
make ml
```

### Outputs
- Model artifact: `models/baseline_logreg.joblib`
- Metrics report: `reports/baseline_metrics.md`
- Processed data + splits: `data/processed/heart.csv`, `data/processed/train.csv`, `data/processed/val.csv`, `data/processed/test.csv`

## Docs
- Decision Log: `docs/decisions.md`
- Model Card: `reports/model_card.md`

## Run inference (baseline)

    # default (uses test split)
    make predict-baseline

    # customize input/output/threshold
    make predict-baseline INPUT=data/processed/val.csv OUT=reports/predictions_val.csv THRESH=0.7

### Output
- Predictions CSV (gitignored): `reports/predictions_*.csv`

### Notes
- Inference is cached by Make via `OUT`. If it says "Nothing to be done", the output is already up-to-date.
- To force regeneration: `make clean-preds` (or delete the OUT file).

## Evaluate predictions (baseline)

    # evaluate default baseline predictions vs labeled test split
    make eval-baseline

    # customize paths
    make eval-baseline EVAL_INPUT=data/processed/val.csv EVAL_PREDS=reports/predictions_val.csv EVAL_OUT=reports/val_eval.json

### Output
- Metrics JSON (gitignored): `reports/*eval*.json`

## Sweep thresholds (baseline)

    # generate threshold_sweep.csv for baseline predictions
    make sweep-thresholds

    # customize input/preds + sweep range
    make sweep-thresholds SWEEP_INPUT=data/processed/val.csv SWEEP_PREDS=reports/predictions_val.csv SWEEP_OUT=reports/val_threshold_sweep.csv TMIN=0.1 TMAX=0.9 TSTEP=0.05

### Output
- Threshold sweep CSV (gitignored): `reports/threshold_sweep.csv`

## Best threshold (baseline)

After running `make sweep-thresholds`, the best F1 on the current test split was at threshold **0.70**.

    # generate predictions using BEST_THRESH (default: 0.70)
    make predict-baseline-best

    # evaluate those predictions (writes JSON)
    make eval-baseline-best

    cat reports/eval_thresh_0.70.json

### Notes
- In a real project, you should tune the threshold on **val** and only report final numbers on **test**.



## Val-tuned threshold (baseline)

End-to-end (recommended): tune the decision threshold on **val**, then apply it on **test**.

    # full rerun + write a markdown report (recommended)
    make val-tune-baseline-report

    sed -n "1,160p" reports/val_tuning_report.md

### Notes
- This keeps the workflow correct: tune on **val**, report final metrics on **test**.
## Val-tuned threshold (baseline)

End-to-end (recommended): tune the decision threshold on **val**, then apply it on **test**.

    # full rerun + write a markdown report (recommended)
    make val-tune-baseline-report

    sed -n "1,160p" reports/val_tuning_report.md

### Notes
- This keeps the workflow correct: tune on **val**, report final metrics on **test**.

## Random Forest (RF)

Train + evaluate RF on the current splits.

    make train-rf
    make predict-rf RF_INPUT=data/processed/test.csv
    make eval-rf RF_INPUT=data/processed/test.csv
    cat reports/eval_rf.json

### Val-tuned threshold (RF)

End-to-end (recommended): sweep thresholds on **val**, pick the best metric, apply once on **test**, and write a report.

    # full rerun + markdown report (recommended)
    make val-tune-rf-report VAL_BEST_METRIC=f1

    sed -n "1,200p" reports/rf_val_tuning_report.md

### Notes
- Threshold tuning can change the precision/recall tradeoff a lot (e.g., optimizing recall usually lowers precision).

## Model comparison (baseline vs RF)

Compare **val-tuned** baseline logistic regression vs Random Forest, using the same optimization metric (picked on **val**), and report the results on **test**.

    # runs: val-tune-baseline-report + val-tune-rf-report + model-compare-report
    make compare-models VAL_BEST_METRIC=f1

    sed -n "1,200p" reports/model_comparison.md

### Notes
- The winner depends on the metric you optimize (e.g., optimizing recall usually lowers precision).
- Thresholds are tuned on **val**; the comparison table reports metrics on **test**.
