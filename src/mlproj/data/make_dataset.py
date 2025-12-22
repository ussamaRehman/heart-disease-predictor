from __future__ import annotations

from pathlib import Path

import pandas as pd


def _ensure_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have a binary `target` column.

    Rules:
    - If `target` exists, keep it (cast to int).
    - Else if `num` exists, create target = (num > 0) and drop `num`.
    - Else raise.
    """
    out = df.copy()

    if "target" in out.columns:
        out["target"] = pd.to_numeric(out["target"], errors="coerce").fillna(0).astype(int)
        if "num" in out.columns:
            out = out.drop(columns=["num"])
        return out

    if "num" in out.columns:
        num_int = pd.to_numeric(out["num"], errors="coerce").fillna(0).astype(int)
        out["target"] = (num_int > 0).astype(int)
        out = out.drop(columns=["num"])
        return out

    raise ValueError("Expected `target` or `num` column not found.")


def _median_impute_numeric(df: pd.DataFrame, *, exclude: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].median())
    return out


def main() -> None:
    raw = Path("data/raw/uci_heart_disease.csv")
    out_path = "data/processed/heart.csv"

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
