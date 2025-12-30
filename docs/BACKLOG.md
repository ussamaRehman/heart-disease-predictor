# Backlog

## Nice-to-haves (non-blocking)
- Add Makefile help target + DEFAULT_GOAL help
- Add CLI console scripts in `pyproject.toml` for `predict_*` entrypoints
- Add a small pipeline diagram under `docs/`
- Add a short “interpretability note” section to `reports/model_card.md`
- Add a quick "data provenance" note to `docs/results_snapshot.md`
- Add a "demo GIF" or terminal cast in `docs/`
- Add a compact "metrics glossary" section in README
- Add a simple `make demo` target that prints key reports
- Add a `make clean` target for reports and models

## Out of scope for now
- Hyperparameter tuning beyond the current baselines
- Model explainability tooling (SHAP/feature importance dashboards)
- Deployment or API serving layer
- Expanded dataset sourcing or multi-site validation
- Automated data drift monitoring
- CI matrix testing across multiple Python versions
- GPU acceleration or deep learning models
- End-user UI / web app
