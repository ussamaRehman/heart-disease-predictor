"""
Train a baseline classifier and evaluate on val/test.

Reads:
  data/processed/train.csv
  data/processed/val.csv
  data/processed/test.csv

Writes:
  models/baseline_logreg.joblib
  reports/baseline_metrics.md
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_meta() -> dict[str, str]:
    # Best-effort: script should still run outside git
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    try:
        subprocess.check_call(["git", "diff", "--quiet"])
        dirty = "false"
    except Exception:
        dirty = "true"
    return {"git_sha": sha, "git_dirty": dirty}


def load_split(name: str) -> pd.DataFrame:
    p = DATA_DIR / f"{name}.csv"
    if not p.exists():
        raise SystemExit(f"Missing split file: {p}. Run `make pipeline` first.")
    return pd.read_csv(p)


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    # All features are numeric in this dataset; keep it explicit anyway.
    numeric_features = list(X.columns)

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=2000, random_state=42)

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe


def eval_split(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, name: str) -> dict[str, float]:
    y_pred = pipe.predict(X)

    # Probability-based metrics
    proba = pipe.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division="warn")),
        "recall": float(recall_score(y, y_pred, zero_division="warn")),
        "f1": float(f1_score(y, y_pred, zero_division="warn")),
        "roc_auc": float(auc),
    }


def main() -> None:
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    if "target" not in train.columns:
        raise SystemExit("Expected `target` column not found in train split.")

    X_train = train.drop(columns=["target"])
    y_train = cast(pd.Series, train["target"].astype(int))

    X_val = val.drop(columns=["target"])
    y_val = cast(pd.Series, val["target"].astype(int))

    X_test = test.drop(columns=["target"])
    y_test = cast(pd.Series, test["target"].astype(int))

    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    val_metrics = eval_split(pipe, X_val, y_val, "val")
    test_metrics = eval_split(pipe, X_test, y_test, "test")

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "baseline_logreg.joblib"
    joblib.dump(pipe, model_path)

    # Richer text outputs for humans
    val_pred = pipe.predict(X_val)
    test_pred = pipe.predict(X_test)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "baseline_metrics.md"

    report = []
    report.append("# Baseline Metrics — Logistic Regression\n")
    report.append(f"Model artifact: `{model_path}`\n")
    report.append("## Validation\n")
    report.append(f"- Accuracy: {val_metrics['accuracy']:.3f}\n")
    report.append(f"- Precision: {val_metrics['precision']:.3f}\n")
    report.append(f"- Recall: {val_metrics['recall']:.3f}\n")
    report.append(f"- F1: {val_metrics['f1']:.3f}\n")
    report.append(f"- ROC AUC: {val_metrics['roc_auc']:.3f}\n")
    report.append("\nConfusion matrix (val):\n")
    report.append(f"```\n{confusion_matrix(y_val, val_pred)}\n```\n")
    report.append("\nClassification report (val):\n")
    report.append(f"```\n{classification_report(y_val, val_pred, zero_division='warn')}\n```\n")

    report.append("\n## Test\n")
    report.append(f"- Accuracy: {test_metrics['accuracy']:.3f}\n")
    report.append(f"- Precision: {test_metrics['precision']:.3f}\n")
    report.append(f"- Recall: {test_metrics['recall']:.3f}\n")
    report.append(f"- F1: {test_metrics['f1']:.3f}\n")
    report.append(f"- ROC AUC: {test_metrics['roc_auc']:.3f}\n")
    report.append("\nConfusion matrix (test):\n")
    report.append(f"```\n{confusion_matrix(y_test, test_pred)}\n```\n")
    report.append("\nClassification report (test):\n")
    report.append(f"```\n{classification_report(y_test, test_pred, zero_division='warn')}\n```\n")

    report_path.write_text("".join(report), encoding="utf-8")

    # --- Run-scoped outputs + metadata ---

    git = _git_meta()

    run_ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    run_id = f"{run_ts}-{git['git_sha']}"

    report_dir = Path("reports/runs") / run_id

    model_dir = Path("models/runs") / run_id

    report_dir.mkdir(parents=True, exist_ok=True)

    model_dir.mkdir(parents=True, exist_ok=True)

    metrics_json_path = report_dir / "metrics.json"

    latest_report_md = Path("reports/latest_baseline_metrics.md")

    latest_report_json = Path("reports/latest_baseline_metrics.json")

    latest_model = Path("models/latest_baseline_logreg.joblib")

    dataset_hashes: dict[str, str] = {}

    for _name, _path in {
        "processed_heart_csv_sha256": Path("data/processed/heart.csv"),
        "train_csv_sha256": Path("data/processed/train.csv"),
        "val_csv_sha256": Path("data/processed/val.csv"),
        "test_csv_sha256": Path("data/processed/test.csv"),
    }.items():
        try:
            dataset_hashes[_name] = _sha256_file(_path)

        except Exception:
            dataset_hashes[_name] = "unavailable"

    payload = {
        "run_id": run_id,
        **git,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "dataset_hashes": dataset_hashes,
    }
    metrics_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Write latest copies for convenience
    latest_report_md.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_report_json.write_text(metrics_json_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Also keep a latest model copy (gitignored)
    try:
        latest_model.write_bytes(model_path.read_bytes())
    except Exception:
        pass

    print("Training complete.")
    print(f"Saved model: {model_path}")
    print(f"Wrote report: {report_path}")
    print("VAL metrics:", val_metrics)
    print("TEST metrics:", test_metrics)


if __name__ == "__main__":
    main()
