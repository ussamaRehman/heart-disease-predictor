from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


def _align_input_and_preds(
    input_df: pd.DataFrame, preds_df: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    if "target" not in input_df.columns:
        raise ValueError('input must contain target column "target"')

    if "proba_disease" in preds_df.columns:
        proba = preds_df["proba_disease"]
    elif "proba" in preds_df.columns:
        proba = preds_df["proba"]
    else:
        raise ValueError('preds must contain probability column "proba_disease" (or "proba")')

    if len(input_df) != len(preds_df):
        raise ValueError(
            f"input and preds must have same number of rows (got {len(input_df)} vs {len(preds_df)})"
        )

    y_true = input_df["target"].astype(int)
    return y_true, proba.astype(float)


def compute_pr_curve(y_true: pd.Series, proba: pd.Series) -> tuple[pd.DataFrame, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    thr = list(thresholds) + [float("nan")]  # pad to same length as precision/recall
    df = pd.DataFrame({"precision": precision, "recall": recall, "threshold": thr})
    ap = float(average_precision_score(y_true, proba))
    return df, ap


def render_pr_summary(ap: float) -> str:
    return "# Precision–Recall (PR) summary\n\n" + f"- **Average Precision (AP):** `{ap:.4f}`\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    input_df = pd.read_csv(args.input)
    preds_df = pd.read_csv(args.preds)

    y_true, proba = _align_input_and_preds(input_df, preds_df)
    curve_df, ap_score = compute_pr_curve(y_true, proba)

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    curve_df.to_csv(out_csv, index=False)
    out_md.write_text(render_pr_summary(ap_score), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
