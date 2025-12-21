"""
Create a processed dataset from the raw UCI Heart Disease CSV.

Reads:
  data/raw/uci_heart_disease.csv

Writes:
  data/processed/heart.csv

Processing rules:
- Ensures `target` is int 0/1.
- Imputes missing values in numeric columns (median) for `ca` and `thal` (and any others if present).
- Saves a clean CSV for modeling.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH_DEFAULT = "data/raw/uci_heart_disease.csv"
OUT_PATH_DEFAULT = "data/processed/heart.csv"


def _ensure_target(df: pd.DataFrame) -> pd.DataFrame:
    if "target" not in df.columns:
        raise ValueError("Expected column `target` not found. Run `make data` first.")
    # Coerce to numeric, then to 0/1 int
    t = pd.to_numeric(df["target"], errors="coerce")
    if t.isna().any():
        raise ValueError("`target` contains non-numeric or missing values.")
    df = df.copy()
    df["target"] = (t > 0).astype(int)
    return df


def _median_impute_numeric(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    exclude = exclude or []
    df = df.copy()

    numeric_cols = [
        c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    for col in numeric_cols:
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
    return df


def main(raw_path: str = RAW_PATH_DEFAULT, out_path: str = OUT_PATH_DEFAULT) -> None:
    raw = Path(raw_path)
    if not raw.exists():
        raise SystemExit(f"Missing raw dataset: {raw}. Run `make data` first.")

    df = pd.read_csv(raw)

    df = _ensure_target(df)

    # Impute numerics (excluding target)
    df = _median_impute_numeric(df, exclude=["target"])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Saved processed dataset: {out} | shape={df.shape}")
    print("Missing values (should be 0):")
    miss = df.isna().sum().sum()
    print(miss)
    print("Target distribution:")
    print(df["target"].value_counts())


if __name__ == "__main__":
    main()
