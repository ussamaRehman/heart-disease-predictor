"""
Create stratified train/val/test splits from the processed dataset.

Reads:
  data/processed/heart.csv

Writes:
  data/processed/train.csv
  data/processed/val.csv
  data/processed/test.csv

Defaults:
  train=0.70, val=0.15, test=0.15 (stratified on target)
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split

IN_PATH_DEFAULT = "data/processed/heart.csv"
OUT_DIR_DEFAULT = "data/processed"


def main(
    in_path: str = IN_PATH_DEFAULT,
    out_dir: str = OUT_DIR_DEFAULT,
    seed: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> None:
    p = Path(in_path)
    if not p.exists():
        raise SystemExit(f"Missing processed dataset: {p}. Run `make preprocess` first.")

    df = pd.read_csv(p)
    if "target" not in df.columns:
        raise SystemExit("Expected column `target` not found in processed dataset.")

    y: pd.Series = df["target"]

    # 1) Split off test
    train_val_any, test_any = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    train_val = cast(pd.DataFrame, train_val_any)
    test = cast(pd.DataFrame, test_any)

    # 2) Split remaining into train/val
    val_rel = val_size / (1.0 - test_size)
    y_tv: pd.Series = train_val["target"]

    train_any, val_any = train_test_split(
        train_val,
        test_size=val_rel,
        random_state=seed,
        stratify=y_tv,
    )
    train = cast(pd.DataFrame, train_any)
    val = cast(pd.DataFrame, val_any)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "train.csv"
    val_path = out / "val.csv"
    test_path = out / "test.csv"

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    def _dist(x: pd.DataFrame) -> str:
        vc = x["target"].astype(int).value_counts(normalize=True).sort_index()
        parts: list[str] = []
        for k, v in vc.items():
            k_int = int(cast(int, k))
            parts.append(f"{k_int}:{float(v):.3f}")
        return ", ".join(parts)

    print("Saved splits:")
    print(f"  train: {train_path} | n={len(train)} | target dist: {_dist(train)}")
    print(f"  val:   {val_path}   | n={len(val)} | target dist: {_dist(val)}")
    print(f"  test:  {test_path}  | n={len(test)} | target dist: {_dist(test)}")


if __name__ == "__main__":
    main()
