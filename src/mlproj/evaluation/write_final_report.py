from __future__ import annotations

import argparse
import re
from pathlib import Path


def _parse_metric_and_winner(compare_md: str) -> tuple[str, str]:
    metric_m = re.search(r"\*\*Optimized metric \(picked on val\):\*\* `([^`]+)`", compare_md)
    winner_m = re.search(r"\*\*Winner \(by `[^`]+` on test\):\*\* `([^`]+)`", compare_md)

    if not metric_m or not winner_m:
        raise ValueError("Could not parse metric/winner from comparison markdown")

    return metric_m.group(1), winner_m.group(1)


def render_final_report(
    *,
    metric: str,
    winner: str,
    compare_md: str,
    baseline_md: str,
    rf_md: str,
    pr_baseline_md: str | None = None,
    pr_rf_md: str | None = None,
) -> str:
    parts: list[str] = []

    parts.append("# Final report\n\n")
    parts.append("This file aggregates the project outputs into one place.\n\n")
    parts.append(f"**Optimized metric (picked on val):** `{metric}`\n")
    parts.append(f"**Winner (by `{metric}` on test):** `{winner}`\n\n")

    parts.append("## Model comparison report\n\n")
    parts.append(compare_md.strip() + "\n\n")

    parts.append("## Baseline val-tuning report\n\n")
    parts.append(baseline_md.strip() + "\n\n")

    parts.append("## Random Forest val-tuning report\n\n")
    parts.append(rf_md.strip() + "\n\n")

    if pr_baseline_md or pr_rf_md:
        parts.append("## Precision–Recall (PR) summaries\n\n")
        if pr_baseline_md:
            parts.append("### Baseline PR summary\n\n")
            parts.append(pr_baseline_md.strip() + "\n\n")
        if pr_rf_md:
            parts.append("### Random Forest PR summary\n\n")
            parts.append(pr_rf_md.strip() + "\n\n")

    return "".join(parts).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-report", required=True)
    ap.add_argument("--rf-report", required=True)
    ap.add_argument("--compare-report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pr-baseline-md", required=False)
    ap.add_argument("--pr-rf-md", required=False)
    args = ap.parse_args()

    baseline_md = Path(args.baseline_report).read_text(encoding="utf-8")
    rf_md = Path(args.rf_report).read_text(encoding="utf-8")
    compare_md = Path(args.compare_report).read_text(encoding="utf-8")

    metric, winner = _parse_metric_and_winner(compare_md)

    pr_baseline_md = (
        Path(args.pr_baseline_md).read_text(encoding="utf-8") if args.pr_baseline_md else None
    )
    pr_rf_md = Path(args.pr_rf_md).read_text(encoding="utf-8") if args.pr_rf_md else None

    md = render_final_report(
        metric=metric,
        winner=winner,
        compare_md=compare_md,
        baseline_md=baseline_md,
        rf_md=rf_md,
        pr_baseline_md=pr_baseline_md,
        pr_rf_md=pr_rf_md,
    )

    Path(args.out).write_text(md, encoding="utf-8")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
