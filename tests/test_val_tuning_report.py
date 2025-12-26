from mlproj.evaluation.write_val_tuning_report import render_val_tuning_report


def test_render_val_tuning_report_contains_metric_and_threshold() -> None:
    md = render_val_tuning_report(
        metric="f1",
        best_threshold=0.35,
        val_best={"threshold": 0.35, "f1": 0.82, "precision": 0.85, "recall": 0.81},
        test_metrics={"f1": 0.80, "roc_auc": 0.93},
    )
    assert "`f1`" in md
    assert "`0.350`" in md
    assert "## Best row on val" in md
    assert "## Test metrics" in md
