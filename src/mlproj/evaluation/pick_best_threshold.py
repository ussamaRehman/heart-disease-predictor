from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def pick_best_threshold(df: pd.DataFrame, metric: str = "f1") -> float:
    required = {"threshold", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in sweep CSV: {sorted(missing)}")

    # tie-breaker: higher metric wins; if tie, smaller threshold wins
    best = df.sort_values([metric, "threshold"], ascending=[False, True]).iloc[0]
    return float(best["threshold"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to threshold sweep CSV")
    parser.add_argument("--metric", default="f1", help="Metric column to maximize (default: f1)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    thr = pick_best_threshold(df, metric=args.metric)
    # keep consistent formatting
    print(f"{thr:.3f}")


if __name__ == "__main__":
    main()
