"""
Download the UCI Heart Disease dataset (id=45) via ucimlrepo and save as CSV.

Output:
  data/raw/uci_heart_disease.csv  (gitignored)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


def main(out_path: str = "data/raw/uci_heart_disease.csv") -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = fetch_ucirepo(id=45)

    X = ds.data.features  # pandas DataFrame
    y = ds.data.targets  # pandas DataFrame (contains 'num' typically)

    df = X.copy()
    # Join targets (keeps original multiclass target if present)
    for col in y.columns:
        df[col] = y[col]

    # Standard binary target used in most papers:
    # num == 0 => no disease, num in {1,2,3,4} => disease
    if "num" in df.columns:
        df["target"] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype("Int64")

    df.to_csv(out, index=False)

    print(f"Saved: {out} | shape={df.shape} | columns={len(df.columns)}")
    if "target" in df.columns:
        print(df["target"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
