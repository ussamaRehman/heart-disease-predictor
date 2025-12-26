"""Write a simple markdown report for val-tuned threshold selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mlproj.evaluation.pick_best_threshold import pick_best_threshold


def _best_row(df: pd.DataFrame, metric: str) -> dict[str, Any]:
    thr = pick_best_threshold(df, metric=metric)
    row = df.loc[df["threshold"] == thr].iloc[0]
    out: dict[str, Any] = {}
    for k, v in row.items():
        out[str(k)] = v
    return out


def render_val_tuning_report(
    *, metric: str, best_threshold: float, val_best: dict[str, Any], test_metrics: dict[str, Any]
) -> str:
    lines = []
    lines.append("# Val-tuned threshold report (baseline)")
    lines.append("")
    lines.append(f"**Optimized metric (on val):** `{metric}`")
    lines.append(f"**Chosen threshold:** `{best_threshold:.3f}`")
    lines.append("")
    lines.append("## Best row on val")
    lines.append("")
    keys = ["threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "tp", "fp", "tn", "fn"]
    for k in keys:
        if k in val_best:
            lines.append(f"- **{k}**: {val_best[k]}")
    lines.append("")
    lines.append("## Test metrics at chosen threshold")
    lines.append("")
    order = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for k in order:
        if k in test_metrics:
            lines.append(f"- **{k}**: {test_metrics[k]}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-csv", required=True)
    ap.add_argument("--metric", default="f1")
    ap.add_argument("--threshold-file", required=True)
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(Path(args.sweep_csv))
    best_thr = float(Path(args.threshold_file).read_text(encoding="utf-8").strip())
    val_best = _best_row(df, args.metric)

    payload = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    test_metrics = payload.get("metrics", {})

    md = render_val_tuning_report(
        metric=args.metric, best_threshold=best_thr, val_best=val_best, test_metrics=test_metrics
    )
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
