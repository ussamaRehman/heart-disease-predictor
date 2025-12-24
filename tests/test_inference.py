from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlproj.inference.predict_baseline import prepare_features


def test_prepare_features_drops_target_reorders_and_ignores_extras(tmp_path: Path) -> None:
    train = pd.DataFrame({"a": [0, 1, 0, 1], "b": [0, 0, 1, 1]})
    y = pd.Series([0, 0, 1, 1])

    model = LogisticRegression().fit(train, y)
    joblib.dump(model, tmp_path / "m.joblib")

    inp = pd.DataFrame(
        {
            "b": [1, 0],
            "a": [0, 1],
            "target": [123, 456],  # should be dropped
            "extra": [9, 9],  # should be ignored
        }
    )

    x = prepare_features(model, inp)

    assert list(x.columns) == ["a", "b"]
    assert len(x) == 2
