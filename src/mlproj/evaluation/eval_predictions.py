from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=cast(Any, 0))),
        "recall": float(recall_score(y_true, y_pred, zero_division=cast(Any, 0))),
        "f1": float(f1_score(y_true, y_pred, zero_division=cast(Any, 0))),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


def load_and_align(
    input_path: Path, preds_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    df_true = pd.read_csv(input_path)
    if "target" not in df_true.columns:
        raise ValueError(f"Input must contain a 'target' column: {input_path}")

    df_true = df_true.reset_index(drop=True).copy()
    df_true["row_id"] = np.arange(len(df_true), dtype=int)

    df_pred = pd.read_csv(preds_path)
    if "pred" not in df_pred.columns:
        raise ValueError(f"Predictions must contain a 'pred' column: {preds_path}")

    if "row_id" in df_pred.columns:
        merged = df_true.merge(df_pred, on="row_id", how="inner", validate="one_to_one")
    else:
        if len(df_pred) != len(df_true):
            raise ValueError(
                f"Row mismatch: input rows={len(df_true)} preds rows={len(df_pred)} and no row_id to align."
            )
        merged = pd.concat([df_true[["row_id", "target"]], df_pred], axis=1)

    y_true = merged["target"].astype(int).to_numpy()
    y_pred = merged["pred"].astype(int).to_numpy()
    y_proba = merged["proba_disease"].to_numpy() if "proba_disease" in merged.columns else None
    return y_true, y_pred, y_proba, len(merged)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate predictions CSV against a labeled input CSV."
    )
    ap.add_argument("--input", required=True, help="CSV with ground truth 'target' column.")
    ap.add_argument("--preds", required=True, help="Predictions CSV from predict_baseline.")
    ap.add_argument("--out", default="reports/latest_eval_baseline.json", help="Output JSON path.")
    args = ap.parse_args()

    input_path = Path(args.input)
    preds_path = Path(args.preds)
    out_path = Path(args.out)

    y_true, y_pred, y_proba, n_rows = load_and_align(input_path, preds_path)
    metrics = compute_metrics(y_true, y_pred, y_proba)

    payload: dict[str, Any] = {
        "input": str(input_path),
        "preds": str(preds_path),
        "n_rows": n_rows,
        "metrics": metrics,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path} | metrics={metrics}")


if __name__ == "__main__":
    main()
