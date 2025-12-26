import pandas as pd

from mlproj.evaluation.pr_curve import compute_pr_curve


def test_compute_pr_curve_outputs_valid_ap_and_columns() -> None:
    y_true = pd.Series([0, 0, 1, 1], dtype=int)
    proba = pd.Series([0.1, 0.4, 0.6, 0.9], dtype=float)

    df, ap = compute_pr_curve(y_true, proba)

    assert 0.0 <= ap <= 1.0
    assert list(df.columns) == ["precision", "recall", "threshold"]
    assert len(df) >= 2
