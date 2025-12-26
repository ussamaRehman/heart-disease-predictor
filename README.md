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

    # uses caching (great for fresh clone or repeat runs)
    make val-tune-baseline

    # force a full rerun (cleans generated files first)
    make val-tune-baseline-clean

    cat reports/val_best_threshold.txt
    cat reports/eval_valtuned_auto.json

### Notes
- This keeps the workflow correct: tune on **val**, report final metrics on **test**.
## Val-tuned threshold (baseline)

End-to-end (recommended): tune the decision threshold on **val**, then apply it on **test**.

    # uses caching (great for fresh clone or repeat runs)
    make val-tune-baseline

    # force a full rerun (cleans generated files first)
    make val-tune-baseline-clean

    cat reports/val_best_threshold.txt
    cat reports/eval_valtuned_auto.json

### Notes
- This keeps the workflow correct: tune on **val**, report final metrics on **test**.
