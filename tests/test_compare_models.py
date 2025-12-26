from mlproj.evaluation.compare_models import render_model_comparison_report


def test_render_model_comparison_report_contains_winner_and_table() -> None:
    md = render_model_comparison_report(
        metric="f1",
        baseline_thr=0.35,
        baseline_metrics={
            "accuracy": 0.80,
            "precision": 0.73,
            "recall": 0.90,
            "f1": 0.81,
            "roc_auc": 0.93,
        },
        rf_thr=0.20,
        rf_metrics={
            "accuracy": 0.89,
            "precision": 0.86,
            "recall": 0.90,
            "f1": 0.88,
            "roc_auc": 0.94,
        },
    )
    assert "# Model comparison" in md
    assert "Winner" in md
    assert "| Model | Threshold |" in md
    assert "baseline_logreg" in md
    assert "random_forest" in md
