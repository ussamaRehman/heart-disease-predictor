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

## Docs
- Decision Log: `docs/decisions.md`
- Model Card: `reports/model_card.md`

