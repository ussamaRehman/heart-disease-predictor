from __future__ import annotations

import pandas as pd

from mlproj.data.make_dataset import _ensure_target, _median_impute_numeric
from mlproj.data.split import stratified_split


def test_ensure_target_drops_num_and_creates_binary_target() -> None:
    df = pd.DataFrame(
        {
            "age": [50, 60, 40, 70],
            "ca": [0.0, None, 2.0, None],
            "thal": [3.0, 6.0, None, 7.0],
            "num": [0, 2, 1, 0],
        }
    )

    out = _ensure_target(df)

    assert "target" in out.columns
    assert "num" not in out.columns  # should be dropped after target is created
    assert set(out["target"].unique()) <= {0, 1}


def test_median_impute_numeric_removes_nans() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, None, 3.0],
            "b": [None, 2.0, 3.0],
            "target": [0, 1, 0],
        }
    )

    out = _median_impute_numeric(df, exclude=["target"])
    assert int(out.isna().sum().sum()) == 0


def test_stratified_split_keeps_target_distribution_reasonably_close() -> None:
    # 100 rows, balanced target
    df = pd.DataFrame({"x": range(100), "target": [0] * 50 + [1] * 50})

    train, val, test = stratified_split(df, test_size=0.2, val_size=0.1, random_state=42)

    assert len(train) + len(val) + len(test) == 100

    # Compare positive rate across splits (should be close)
    def pos_rate(x: pd.DataFrame) -> float:
        return float(x["target"].mean())

    pr_train, pr_val, pr_test = pos_rate(train), pos_rate(val), pos_rate(test)
    assert abs(pr_train - 0.5) <= 0.1
    assert abs(pr_val - 0.5) <= 0.2
    assert abs(pr_test - 0.5) <= 0.2
