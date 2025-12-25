import pandas as pd

from mlproj.evaluation.pick_best_threshold import pick_best_threshold


def test_pick_best_threshold_tie_breaker_prefers_lower_threshold() -> None:
    df = pd.DataFrame(
        {
            "threshold": [0.30, 0.35, 0.40],
            "f1": [0.80, 0.82, 0.82],  # tie between 0.35 and 0.40
        }
    )
    thr = pick_best_threshold(df, metric="f1")
    assert thr == 0.35
