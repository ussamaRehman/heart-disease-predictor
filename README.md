# Heart Disease Predictor (reproducible ML pipeline)

![CI](../../actions/workflows/ci.yml/badge.svg?branch=master)


A small, production-style ML project that predicts heart disease using the UCI Heart Disease dataset.
Includes a reproducible pipeline, threshold tuning on a validation set, model comparison, PR curve summaries,
and a final aggregated report.

## Quickstart

### 1) Install dependencies
This repo uses `uv` for fast, pinned installs.

- Install `uv` (once): https://docs.astral.sh/uv/
- Then run:

    uv sync --dev

### 2) Run the end-to-end value pipeline
This runs checks + trains models + generates reports:

    make report-e2e VAL_BEST_METRIC=f1

## Demo / proof

For a quick, read-only demo path without committing generated reports:

- `docs/results_snapshot.md`
- `reports/model_card.md`
- Latest Release (download the `reports_bundle_<tag>.zip` asset): `../../releases/latest`

## What this project produces

All outputs are written to `reports/`. The main entry point is:

- `reports/final_report.md` (aggregates everything)

Also included:
- `reports/model_comparison.md` (baseline vs RF vs HGB at each model’s val-tuned threshold)
- `reports/*val_tuning_report.md` (chosen threshold on val + resulting test metrics)
- `reports/pr_curve_*.md` and `reports/pr_curve_*.csv` (Precision–Recall summaries + curve data)
- `reports/*.json` (machine-readable metrics)
- `reports/predictions_*.csv` (predictions for val/test runs)

## How evaluation works (high level)

- The model outputs probabilities.
- We select the best classification threshold on the **validation set** based on `VAL_BEST_METRIC` (default: `f1`).
- We then report metrics on the **test set** at that chosen threshold.
- PR curves are generated to summarize the precision/recall tradeoff.

## Make targets (useful ones)

- `make check` — format/lint/typecheck/tests
- `make data` — download dataset
- `make preprocess` — build processed dataset
- `make split` — train/val/test split
- `make train-baseline` / `make train-rf` / `make train-hgb` — train models + write metric reports
- `make final-report-print VAL_BEST_METRIC=f1` — generate final report
- `make pr-curves-print VAL_BEST_METRIC=f1` — generate PR summaries
- `make report-e2e VAL_BEST_METRIC=f1` — **one-command end-to-end “value step”**

## CI

GitHub Actions runs the same value step:

- `make report-e2e VAL_BEST_METRIC=f1`

It also uploads the `reports/` outputs as an artifact (even if the job fails),
so you can download `final_report.md` and related files directly from the workflow run.

## Notes / limitations

- This is a small dataset; metrics can vary with random seeds/splits.
- This repo emphasizes **clean workflow + reproducibility** over state-of-the-art modeling.

## License

MIT
