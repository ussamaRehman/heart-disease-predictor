from __future__ import annotations

from mlproj.evaluation.compare_models_3 import ModelRow, render_compare_models_3


def test_compare_models_3_picks_winner_by_metric() -> None:
    rows = [
        ModelRow("baseline_logreg", 0.35, {"f1": 0.81, "roc_auc": 0.93, "accuracy": 0.80}),
        ModelRow("random_forest", 0.20, {"f1": 0.75, "roc_auc": 0.94, "accuracy": 0.70}),
        ModelRow("hist_gradient_boosting", 0.95, {"f1": 0.87, "roc_auc": 0.94, "accuracy": 0.89}),
    ]
    md = render_compare_models_3("f1", rows)
    assert "**Winner (by `f1` on test):** `hist_gradient_boosting`" in md
    assert "| hist_gradient_boosting | `0.950`" in md
