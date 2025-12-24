from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


def test_predict_baseline_cli_writes_csv():
    project_root = Path(__file__).resolve().parents[1]
    src = project_root / "src"

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        model_path = td_path / "toy_model.joblib"
        inp_path = td_path / "input.csv"
        out_path = td_path / "out.csv"

        # Train a tiny model with known feature names
        X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 0.0, 1.0, 0.0]})
        y = pd.Series([0, 0, 1, 1])
        model = LogisticRegression(max_iter=200).fit(X, y)
        joblib.dump(model, model_path)

        # Input CSV can have extra columns; script should select/align features
        X_in = pd.DataFrame({"a": [0.5, 2.5, 1.5], "b": [1.0, 0.0, 1.0], "target": [0, 1, 0]})
        X_in.to_csv(inp_path, index=False)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(src)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlproj.inference.predict_baseline",
                "--model",
                str(model_path),
                "--input",
                str(inp_path),
                "--out",
                str(out_path),
                "--threshold",
                "0.5",
            ],
            check=True,
            env=env,
            cwd=str(project_root),
        )

        assert out_path.exists()

        out = pd.read_csv(out_path)
        assert list(out.columns) == ["row_id", "proba_disease", "pred"]
        assert len(out) == 3
        assert out["proba_disease"].between(0.0, 1.0).all()
        assert set(out["pred"].unique()).issubset({0, 1})
