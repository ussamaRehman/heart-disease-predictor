# Model Card — Heart Disease Predictor Model (Placeholder)

## Model Details
- **Model name:** Heart Disease Predictor Model
- **Model type:** Binary classification (heart disease risk)
- **Framework:** scikit-learn (planned)
- **Version:** v0.0 (scaffold only)

## Intended Use
- Portfolio / educational project.
- Demonstrates an end-to-end ML workflow and production-minded repo structure.

## Data
- Dataset: TBD (planned: UCI Cleveland Heart Disease dataset or equivalent).
- Notes: Will document exact source + preprocessing once dataset is added.

## Metrics (to be filled)
- Accuracy:
- F1:
- ROC AUC:
- Recall (positive class):

## Ethical / Safety Notes
- Not for diagnostic use.
- False negatives can be harmful in healthcare contexts; evaluation will emphasize recall and calibration.

## Limitations (to be filled)
- Small dataset size
- Potential dataset bias
- Feature availability mismatch (screening vs diagnostic features)

## Reproducibility
- Python: `.python-version`
- Dependencies: `pyproject.toml` + `uv.lock`
- Checks: `make check`
