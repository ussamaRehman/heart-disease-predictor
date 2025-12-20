# Decision Log

This file tracks key technical decisions for the Heart Disease Predictor Model.

## 2025-12-20 — Repository scaffolding choices
- Python pinned via pyenv (`.python-version`)
- Dependency management: uv (`pyproject.toml` + `uv.lock`)
- Quality gates: Ruff (format + lint), Pyright (types), Pytest (tests)
- Local gating: pre-commit
- CI: GitHub Actions (ruff/pyright/pytest)

## Next decisions (to be logged)
- Dataset source (UCI vs Kaggle) + exact download method
- Feature set policy (exclude invasive/leaky features like `ca`, `thal` if modeling “screening” scenario)
- Train/val/test strategy (CV vs holdout) and metric priorities (recall/AUC)
- Thresholding + calibration approach
