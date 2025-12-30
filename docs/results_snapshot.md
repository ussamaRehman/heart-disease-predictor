# Results Snapshot

This is a compact, committed summary of the best model results for quick portfolio review.
Full details are in `reports/model_comparison.md` and `reports/final_report.md`.

## Best model (val-tuned F1)

| Model | Threshold | accuracy | precision | recall | f1 | roc_auc |
|---|---:|:---:|:---:|:---:|:---:|:---:|
| hist_gradient_boosting | 0.950 | 0.891 | 0.944 | 0.810 | 0.872 | 0.945 |

Notes:
- Threshold is selected on the validation set; metrics above are on the test set.
- `VAL_BEST_METRIC` is set to `f1` in the default pipeline.
