from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import joblib
import pandas as pd


def _expected_features(model: object) -> list[str] | None:
    feats = getattr(model, "feature_names_in_", None)
    if feats is None:
        return None
    return [str(x) for x in cast(Sequence[object], feats)]


def prepare_features(model: object, df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    # Real inference inputs usually won't have target, but allow it if present.
    if "target" in x.columns:
        x = x.drop(columns=["target"])

    feats = _expected_features(model)
    if feats is None:
        # Fallback: best effort. (Still OK for our project inputs.)
        return x

    missing = [c for c in feats if c not in x.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Reorder + ignore extras
    return x[feats]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baseline model inference on a CSV.")
    ap.add_argument("--model", default="models/baseline_logreg.joblib")
    ap.add_argument("--input", default="data/processed/test.csv")
    ap.add_argument("--out", default="reports/predictions_baseline.csv")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    model_path = Path(args.model)
    model = joblib.load(model_path)

    df = pd.read_csv(args.input)
    x = prepare_features(model, df)

    proba = model.predict_proba(x)[:, 1]
    pred = (proba >= args.threshold).astype(int)

    out = pd.DataFrame(
        {
            "row_id": x.index.astype(int),
            "proba_disease": proba,
            "pred": pred,
        }
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Loaded model: {model_path}")
    print(f"Input: {args.input} | rows={len(x)}")
    print(f"Wrote predictions: {out_path}")


if __name__ == "__main__":
    main()
