from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelRow:
    name: str
    threshold: float
    metrics: dict[str, float]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_metrics(eval_path: Path) -> dict[str, float]:
    data = _read_json(eval_path)
    m = data.get("metrics", data)  # support {"metrics": {...}} or {...}
    out: dict[str, float] = {}
    for k, v in m.items():
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
    return out


def _read_threshold(path: Path) -> float:
    return float(path.read_text(encoding="utf-8").strip())


def _pick_winner(metric: str, rows: list[ModelRow]) -> str:
    def key(r: ModelRow) -> tuple[float, float, float]:
        return (
            r.metrics.get(metric, float("-inf")),
            r.metrics.get("roc_auc", float("-inf")),
            r.metrics.get("accuracy", float("-inf")),
        )

    return max(rows, key=key).name


def render_compare_models_3(metric: str, rows: list[ModelRow]) -> str:
    winner = _pick_winner(metric, rows)

    parts: list[str] = []
    parts.append("# Model comparison (val-tuned thresholds)\n\n")
    parts.append(f"**Optimized metric (picked on val):** `{metric}`\n")
    parts.append(f"**Winner (by `{metric}` on test):** `{winner}`\n\n")

    parts.append("## Test metrics at each model\x27s val-tuned threshold\n\n")
    parts.append("| Model | Threshold | accuracy | precision | recall | f1 | roc_auc |\n")
    parts.append("|---|---:|:---:|:---:|:---:|:---:|:---:|\n")

    for r in rows:
        m = r.metrics
        parts.append(
            "| {name} | `{thr:.3f}` | {acc:.3f} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {auc:.3f} |\n".format(
                name=r.name,
                thr=r.threshold,
                acc=m.get("accuracy", float("nan")),
                prec=m.get("precision", float("nan")),
                rec=m.get("recall", float("nan")),
                f1=m.get("f1", float("nan")),
                auc=m.get("roc_auc", float("nan")),
            )
        )

    parts.append("\n### Notes\n")
    parts.append("- Thresholds are tuned on **val**; this table reports metrics on **test**.\n")
    parts.append(
        "- Different models can prefer very different thresholds; this shifts the precision/recall tradeoff.\n"
    )
    return "".join(parts).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True)

    ap.add_argument("--baseline-eval", required=True)
    ap.add_argument("--baseline-threshold-file", required=True)

    ap.add_argument("--rf-eval", required=True)
    ap.add_argument("--rf-threshold-file", required=True)

    ap.add_argument("--hgb-eval", required=True)
    ap.add_argument("--hgb-threshold-file", required=True)

    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [
        ModelRow(
            name="baseline_logreg",
            threshold=_read_threshold(Path(args.baseline_threshold_file)),
            metrics=_read_metrics(Path(args.baseline_eval)),
        ),
        ModelRow(
            name="random_forest",
            threshold=_read_threshold(Path(args.rf_threshold_file)),
            metrics=_read_metrics(Path(args.rf_eval)),
        ),
        ModelRow(
            name="hist_gradient_boosting",
            threshold=_read_threshold(Path(args.hgb_threshold_file)),
            metrics=_read_metrics(Path(args.hgb_eval)),
        ),
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_compare_models_3(args.metric, rows), encoding="utf-8")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
