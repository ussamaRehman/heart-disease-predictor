from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ZERO_DIV: Any = 0


def _load_xy(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if "target" not in df.columns:
        raise ValueError("Expected a target column in processed dataset")
    x = df.drop(columns=["target"])
    y = df["target"]
    return x, y


def _metrics(y_true: pd.Series, y_prob: list[float], threshold: float = 0.5) -> dict[str, Any]:
    y_pred = [1 if p >= threshold else 0 for p in y_prob]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=ZERO_DIV)),
        "recall": float(recall_score(y_true, y_pred, zero_division=ZERO_DIV)),
        "f1": float(f1_score(y_true, y_pred, zero_division=ZERO_DIV)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    return out


def _render_report_md(val_metrics: dict[str, Any], test_metrics: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# RandomForest baseline")
    lines.append("")
    lines.append("## VAL metrics")
    lines.append("")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "tp", "fp", "tn", "fn"]:
        lines.append(f"- **{k}**: {val_metrics[k]}")
    lines.append("")
    lines.append("## TEST metrics")
    lines.append("")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "tp", "fp", "tn", "fn"]:
        lines.append(f"- **{k}**: {test_metrics[k]}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--val", default="data/processed/val.csv")
    ap.add_argument("--test", default="data/processed/test.csv")
    ap.add_argument("--model-out", default="models/rf_clf.joblib")
    ap.add_argument("--report-out", default="reports/rf_metrics.md")
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--max-depth", type=int, default=0, help="0 means None")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)
    test_path = Path(args.test)

    x_train, y_train = _load_xy(train_path)
    x_val, y_val = _load_xy(val_path)
    x_test, y_test = _load_xy(test_path)

    max_depth = None if args.max_depth == 0 else args.max_depth

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        random_state=args.random_state,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)

    val_prob = [float(p[1]) for p in clf.predict_proba(x_val)]
    test_prob = [float(p[1]) for p in clf.predict_proba(x_test)]

    val_metrics = _metrics(y_val, val_prob, threshold=0.5)
    test_metrics = _metrics(y_test, test_prob, threshold=0.5)

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, Path(args.model_out))
    Path(args.report_out).write_text(_render_report_md(val_metrics, test_metrics), encoding="utf-8")

    print("Training complete.")
    print(f"Saved model: {args.model_out}")
    print(f"Wrote report: {args.report_out}")
    print(f"VAL metrics: {val_metrics}")
    print(f"TEST metrics: {test_metrics}")


if __name__ == "__main__":
    main()
