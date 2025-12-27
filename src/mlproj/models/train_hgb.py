from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def _load_split(split: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(f"data/processed/{split}.csv")
    if "target" not in df.columns:
        raise ValueError('Expected a target column named "target" in processed splits.')
    y = df["target"].astype(int).to_numpy()
    X = df.drop(columns=["target"])
    return X, y


def _classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),  # pyright: ignore[reportArgumentType]
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),  # pyright: ignore[reportArgumentType]
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),  # pyright: ignore[reportArgumentType]
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def train_hgb(*, model_out: Path, report_out: Path, random_state: int = 42) -> None:
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    X_test, y_test = _load_split("test")

    model = HistGradientBoostingClassifier(
        random_state=random_state,
        learning_rate=0.05,
        max_depth=None,
        max_iter=400,
    )
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    val_metrics = _classification_metrics(y_val, val_probs, threshold=0.5)
    test_metrics = _classification_metrics(y_test, test_probs, threshold=0.5)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_out)

    report_out.write_text(
        "# HistGradientBoosting metrics\n\n"
        f"VAL metrics: {val_metrics}\n"
        f"TEST metrics: {test_metrics}\n",
        encoding="utf-8",
    )

    (report_out.parent / "hgb_eval_default050.json").write_text(
        json.dumps({"val": val_metrics, "test": test_metrics}, indent=2),
        encoding="utf-8",
    )

    print("Training complete.")
    print(f"Saved model: {model_out}")
    print(f"Wrote report: {report_out}")
    print(f"VAL metrics: {val_metrics}")
    print(f"TEST metrics: {test_metrics}")


def main() -> None:
    train_hgb(model_out=Path("models/hgb.joblib"), report_out=Path("reports/hgb_metrics.md"))


if __name__ == "__main__":
    main()
