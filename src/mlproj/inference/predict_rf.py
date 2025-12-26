from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with features (may include target column)")
    ap.add_argument("--out", required=True, help="Output predictions CSV")
    ap.add_argument("--model", default="models/rf.joblib", help="Path to RF model joblib")
    ap.add_argument(
        "--threshold", type=float, default=0.5, help="Decision threshold (default: 0.5)"
    )
    args = ap.parse_args()

    df = pd.read_csv(Path(args.input))
    x = df.drop(columns=["target"]) if "target" in df.columns else df

    clf = joblib.load(Path(args.model))
    prob = [float(p[1]) for p in clf.predict_proba(x)]
    pred = [1 if p >= args.threshold else 0 for p in prob]

    out_df = pd.DataFrame({"prob": prob, "pred": pred})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(Path(args.out), index=False)

    print(f"Loaded model: {args.model}")
    print(f"Input: {args.input} | rows={len(df)}")
    print(f"Wrote predictions: {args.out}")


if __name__ == "__main__":
    main()
