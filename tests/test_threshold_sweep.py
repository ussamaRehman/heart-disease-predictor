import pandas as pd

from mlproj.evaluation.sweep_thresholds import sweep_thresholds


def test_sweep_thresholds_outputs_expected_columns() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    proba = pd.Series([0.1, 0.4, 0.6, 0.9])

    df = sweep_thresholds(y_true=y_true, proba=proba, t_min=0.5, t_max=0.5, t_step=0.1)

    assert len(df) == 1
    assert list(df.columns) == [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "tp",
        "fp",
        "tn",
        "fn",
    ]

    row = df.iloc[0]
    assert row["threshold"] == 0.5
    assert 0.0 <= row["roc_auc"] <= 1.0
