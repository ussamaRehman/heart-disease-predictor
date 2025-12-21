"""
Create a processed dataset from the raw UCI Heart Disease CSV.

Reads:
  data/raw/uci_heart_disease.csv

Writes:
  data/processed/heart.csv

Processing rules:
- Ensures `target` is int 0/1.
- Median-imputes missing values in numeric columns (excluding target).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH_DEFAULT = "data/raw/uci_heart_disease.csv"
OUT_PATH_DEFAULT = "data/processed/heart.csv"


def _ensure_target(df: pd.DataFrame) -> pd.DataFrame:
    if "target" not in df.columns:
        raise ValueError("Expected column `target` not found. Run `make data` first.")

    # Ensure a clean numeric series
    target_series: pd.Series = pd.to_numeric(df["target"], errors="coerce")
    if pd.isna(target_series).any():
        raise ValueError("`target` contains non-numeric or missing values.")

    target_int: pd.Series = target_series.astype("int64")
    out = df.copy()
    out["target"] = (target_int > 0).astype(int)
    if "num" in out.columns:
        out = out.drop(columns=["num"])
    return out


def _median_impute_numeric(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    exclude = exclude or []
    out = df.copy()

    for col in out.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            s: pd.Series = out[col]
            if pd.isna(s).any():
                out[col] = s.fillna(s.median())
    return out


def main(raw_path: str = RAW_PATH_DEFAULT, out_path: str = OUT_PATH_DEFAULT) -> None:
    raw = Path(raw_path)
    if not raw.exists():
        raise SystemExit(f"Missing raw dataset: {raw}. Run `make data` first.")

    df = pd.read_csv(raw)

    df = _ensure_target(df)
    df = _median_impute_numeric(df, exclude=["target"])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Saved processed dataset: {out} | shape={df.shape}")
    print("Missing values (should be 0):", int(df.isna().sum().sum()))
    print("Target distribution:")
    print(df["target"].value_counts())


if __name__ == "__main__":
    main()
