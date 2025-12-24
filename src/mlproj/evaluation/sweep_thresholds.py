import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _align_input_and_preds(
    input_df: pd.DataFrame, preds_df: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    if "target" not in input_df.columns:
        raise ValueError("input must contain ground truth column 'target'")

    if "proba_disease" in preds_df.columns:
        proba = preds_df["proba_disease"]
    elif "proba" in preds_df.columns:
        proba = preds_df["proba"]
    else:
        raise ValueError("preds must contain probability column 'proba_disease' (or 'proba')")

    y_true = input_df["target"]

    if len(y_true) != len(proba):
        raise ValueError(f"Row mismatch: input rows={len(y_true)} preds rows={len(proba)}")

    return y_true, proba


def sweep_thresholds(
    y_true: pd.Series, proba: pd.Series, t_min: float, t_max: float, t_step: float
) -> pd.DataFrame:
    thresholds = np.arange(t_min, t_max + 1e-12, t_step)

    # AUC does not depend on threshold (uses probabilities)
    roc_auc = float(roc_auc_score(y_true, proba))

    rows: list[dict[str, float | int]] = []
    for t in thresholds:
        pred = (proba >= float(t)).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

        rows.append(
            {
                "threshold": float(t),
                "accuracy": float(accuracy_score(y_true, pred)),
                "precision": float(precision_score(y_true, pred, zero_division=cast(Any, 0))),
                "recall": float(recall_score(y_true, pred, zero_division=cast(Any, 0))),
                "f1": float(f1_score(y_true, pred, zero_division=cast(Any, 0))),
                "roc_auc": roc_auc,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "tp",
            "fp",
            "tn",
            "fn",
        ],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep classification thresholds.")
    ap.add_argument("--input", required=True, help="CSV with ground truth target.")
    ap.add_argument("--preds", required=True, help="Predictions CSV from inference.")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument("--t-min", type=float, default=0.05)
    ap.add_argument("--t-max", type=float, default=0.95)
    ap.add_argument("--t-step", type=float, default=0.05)
    args = ap.parse_args()

    input_path = Path(args.input)
    preds_path = Path(args.preds)
    out_path = Path(args.out)

    input_df = pd.read_csv(input_path)
    preds_df = pd.read_csv(preds_path)

    y_true, proba = _align_input_and_preds(input_df, preds_df)
    df = sweep_thresholds(
        y_true=y_true, proba=proba, t_min=args.t_min, t_max=args.t_max, t_step=args.t_step
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    best_f1 = df.sort_values(["f1", "threshold"], ascending=[False, True]).iloc[0]
    best_recall = df.sort_values(["recall", "threshold"], ascending=[False, True]).iloc[0]

    print(f"Wrote: {out_path} | rows={len(df)}")
    print(
        f"Best F1: threshold={best_f1['threshold']:.3f} f1={best_f1['f1']:.4f} "
        f"precision={best_f1['precision']:.4f} recall={best_f1['recall']:.4f}"
    )
    print(
        f"Best Recall: threshold={best_recall['threshold']:.3f} recall={best_recall['recall']:.4f} "
        f"precision={best_recall['precision']:.4f} f1={best_recall['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
