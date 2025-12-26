import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mlproj.inference.predict_rf import main


def test_predict_rf_writes_prob_and_pred(tmp_path, monkeypatch) -> None:
    # tiny toy model
    x = pd.DataFrame({"a": [0, 1, 0, 1], "b": [0, 0, 1, 1]})
    y = pd.Series([0, 1, 0, 1])

    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(x, y)

    model_path = tmp_path / "rf.joblib"
    joblib.dump(clf, model_path)

    inp = tmp_path / "in.csv"
    x.to_csv(inp, index=False)

    out = tmp_path / "preds.csv"

    monkeypatch.setattr(
        "sys.argv",
        [
            "predict_rf",
            "--input",
            str(inp),
            "--out",
            str(out),
            "--model",
            str(model_path),
            "--threshold",
            "0.5",
        ],
    )

    main()

    df = pd.read_csv(out)
    assert list(df.columns) == ["prob", "pred"]
    assert len(df) == len(x)
