# Model Card — Heart Disease Predictor

## Model Details
- **Task:** Binary classification (heart disease risk)
- **Candidate models:** Logistic regression (baseline), RandomForest, HistGradientBoosting
- **Best model (by val-tuned F1):** HistGradientBoosting
- **Framework:** scikit-learn
- **Version:** v0.1.0

## Intended Use
- Portfolio / educational project
- Demonstrates an end-to-end ML workflow and reproducible evaluation

## Data
- **Dataset:** UCI Heart Disease (id=45) via `ucimlrepo`
- **Target:** `target = (num > 0)` where `num` is the original UCI label
- **Splits:** Stratified train/val/test (15% test, 15% val)

## Metrics Snapshot (test set at val-tuned threshold)
Best model metrics from `reports/model_comparison.md`:

| Model | Threshold | accuracy | precision | recall | f1 | roc_auc |
|---|---:|:---:|:---:|:---:|:---:|:---:|
| hist_gradient_boosting | 0.950 | 0.891 | 0.944 | 0.810 | 0.872 | 0.945 |

## Ethical / Safety Notes
- Not for diagnostic use
- False negatives can be harmful in healthcare contexts

## Limitations
- Small dataset size; metrics can vary with splits and seeds
- Potential dataset bias (single-source clinical data)
- Feature availability mismatch in real-world screening

## Reproducibility
- Python: `.python-version`
- Dependencies: `pyproject.toml` + `uv.lock`
- Checks: `make check` (local) / `make check-ci` (CI-safe)
