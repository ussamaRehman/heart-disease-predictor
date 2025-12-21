"""
Basic sanity checks for the downloaded dataset.

Reads:  data/raw/uci_heart_disease.csv
Prints: shape, dtypes, missingness summary, and target distribution.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main(path: str = "data/raw/uci_heart_disease.csv") -> None:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Missing file: {p}. Run `make data` first.")

    df = pd.read_csv(p)
    print(f"Loaded: {p} | shape={df.shape}")

    # Target checks
    if "target" in df.columns:
        vc = df["target"].value_counts(dropna=False)
        print("\nTarget distribution (target):")
        print(vc)

    # Missingness
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    print("\nMissing values per column:")
    print(miss if len(miss) else "None")

    print("\nDtypes:")
    print(df.dtypes)

    print("\nHead:")
    print(df.head(3))


if __name__ == "__main__":
    main()
