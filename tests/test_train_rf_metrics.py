import pandas as pd

from mlproj.models.train_rf import _metrics


def test_metrics_returns_expected_keys() -> None:
    y_true = pd.Series([0, 1, 0, 1])
    y_prob = [0.1, 0.9, 0.2, 0.8]
    m = _metrics(y_true, y_prob, threshold=0.5)

    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "tp", "fp", "tn", "fn"]:
        assert k in m
