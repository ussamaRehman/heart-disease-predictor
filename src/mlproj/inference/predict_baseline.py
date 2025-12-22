"""
Baseline inference script.

Loads the committed baseline model:
- models/baseline_logreg.joblib

Reads a CSV, drops target/num if present, predicts:
- probability of disease (class=1)
- predicted label with threshold (default 0.5)

Writes outputs to:
- reports/predictions_baseline.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def load_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop label-ish columns if present
    for col in ["target", "num"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Keep only expected order
    return df[FEATURES]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/test.csv")
    parser.add_argument("--model", type=str, default="models/baseline_logreg.joblib")
    parser.add_argument("--out", type=str, default="reports/predictions_baseline.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `make train-baseline` or `make ml` first."
        )

    x = load_input(Path(args.input))

    model = joblib.load(model_path)

    proba = model.predict_proba(x)[:, 1]
    pred = (proba >= args.threshold).astype(int)

    out = pd.DataFrame({"proba_disease": proba, "pred": pred})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Loaded model: {model_path}")
    print(f"Input: {args.input} | rows={len(x)}")
    print(f"Wrote predictions: {out_path}")


if __name__ == "__main__":
    main()
