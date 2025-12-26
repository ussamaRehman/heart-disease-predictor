from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ORDER = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _load_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Expected key 'metrics' to be a dict in {path}")
    return {str(k): v for k, v in metrics.items()}


def _load_threshold(path: Path) -> float:
    return float(path.read_text(encoding="utf-8").strip())


def _fmt(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{float(v):.3f}"
    return str(v)


def render_model_comparison_report(
    *,
    metric: str,
    baseline_thr: float,
    baseline_metrics: dict[str, Any],
    rf_thr: float,
    rf_metrics: dict[str, Any],
) -> str:
    def metric_value(m: dict[str, Any]) -> float:
        v = m.get(metric)
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    rows = [
        ("baseline_logreg", baseline_thr, baseline_metrics),
        ("random_forest", rf_thr, rf_metrics),
    ]
    winner = max(rows, key=lambda r: metric_value(r[2]))[0]

    lines: list[str] = []
    lines.append("# Model comparison (val-tuned thresholds)")
    lines.append("")
    lines.append(f"**Optimized metric (picked on val):** `{metric}`")
    lines.append(f"**Winner (by `{metric}` on test):** `{winner}`")
    lines.append("")
    lines.append("## Test metrics at each model's val-tuned threshold")
    lines.append("")
    lines.append("| Model | Threshold | " + " | ".join(ORDER) + " |")
    lines.append("|---|---:|" + "|".join([":---:" for _ in ORDER]) + "|")

    for name, thr, m in rows:
        vals = [m.get(k, "") for k in ORDER]
        lines.append("| " + " | ".join([name, f"`{thr:.3f}`"] + [_fmt(x) for x in vals]) + " |")

    lines.append("")
    lines.append("### Notes")
    lines.append("- Thresholds are tuned on **val**; this table reports metrics on **test**.")
    lines.append(
        "- Different models can prefer very different thresholds; this shifts the precision/recall tradeoff."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="f1")
    ap.add_argument("--baseline-eval", required=True)
    ap.add_argument("--baseline-threshold-file", required=True)
    ap.add_argument("--rf-eval", required=True)
    ap.add_argument("--rf-threshold-file", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    baseline_metrics = _load_metrics(Path(args.baseline_eval))
    rf_metrics = _load_metrics(Path(args.rf_eval))

    baseline_thr = _load_threshold(Path(args.baseline_threshold_file))
    rf_thr = _load_threshold(Path(args.rf_threshold_file))

    md = render_model_comparison_report(
        metric=args.metric,
        baseline_thr=baseline_thr,
        baseline_metrics=baseline_metrics,
        rf_thr=rf_thr,
        rf_metrics=rf_metrics,
    )
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
