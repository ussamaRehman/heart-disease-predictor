from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train/val/test with stratification on the `target` column.

    test_size and val_size are fractions of the FULL dataset.
    """
    if "target" not in df.columns:
        raise ValueError("Expected column 'target' in dataframe.")

    if not (0 < test_size < 1) or not (0 < val_size < 1) or (test_size + val_size) >= 1:
        raise ValueError("test_size and val_size must be in (0,1) and sum to < 1.")

    # Explicit Series for stratify (helps type checking)
    target = cast(pd.Series, df.loc[:, "target"])

    # 1) test split
    train_val_any, test_any = train_test_split(
        df,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )
    train_val = cast(pd.DataFrame, train_val_any)
    test = cast(pd.DataFrame, test_any)

    # 2) val split from remaining portion
    val_fraction_of_train_val = val_size / (1.0 - test_size)
    train_any, val_any = train_test_split(
        train_val,
        test_size=val_fraction_of_train_val,
        stratify=cast(pd.Series, train_val.loc[:, "target"]),
        random_state=random_state,
    )
    train = cast(pd.DataFrame, train_any)
    val = cast(pd.DataFrame, val_any)

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def main() -> None:
    in_path = Path("data/processed/heart.csv")
    train_path = Path("data/processed/train.csv")
    val_path = Path("data/processed/val.csv")
    test_path = Path("data/processed/test.csv")

    df = pd.read_csv(in_path)

    train, val, test = stratified_split(df, test_size=0.15, val_size=0.15, random_state=42)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    def _dist(x: pd.DataFrame) -> str:
        vc = x["target"].astype(int).value_counts(normalize=True).sort_index()
        parts: list[str] = []
        for k, v in vc.items():
            parts.append(f"{int(cast(int, k))}:{float(v):.3f}")
        return ", ".join(parts)

    print("Saved splits:")
    print(f"  train: {train_path} | n={len(train)} | target dist: {_dist(train)}")
    print(f"  val:   {val_path}   | n={len(val)} | target dist: {_dist(val)}")
    print(f"  test:  {test_path}  | n={len(test)} | target dist: {_dist(test)}")


if __name__ == "__main__":
    main()
