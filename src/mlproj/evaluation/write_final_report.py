from __future__ import annotations

import argparse
import re
from pathlib import Path


def _parse_metric_and_winner(compare_md: str) -> tuple[str, str]:
    metric = "f1"
    winner = "baseline_logreg"

    for line in compare_md.splitlines():
        if "Optimized metric" in line:
            m = re.findall(r"`([^`]+)`", line)
            if m:
                metric = m[0]
        if line.strip().startswith("**Winner"):
            m = re.findall(r"`([^`]+)`", line)
            # Winner line has two backticked values: metric + winner; we want the last one.
            if m:
                winner = m[-1]

    return metric, winner


def render_final_report(
    *, metric: str, winner: str, compare_md: str, baseline_md: str, rf_md: str
) -> str:
    return (
        "# Final report\n\n"
        "This file aggregates the project outputs into one place.\n\n"
        f"**Optimized metric (picked on val):** `{metric}`\n"
        f"**Winner (by `{metric}` on test):** `{winner}`\n\n"
        "## Model comparison report\n\n"
        f"{compare_md.strip()}\n\n"
        "## Baseline val-tuning report\n\n"
        f"{baseline_md.strip()}\n\n"
        "## Random Forest val-tuning report\n\n"
        f"{rf_md.strip()}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-report", required=True)
    ap.add_argument("--rf-report", required=True)
    ap.add_argument("--compare-report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    baseline_md = Path(args.baseline_report).read_text(encoding="utf-8")
    rf_md = Path(args.rf_report).read_text(encoding="utf-8")
    compare_md = Path(args.compare_report).read_text(encoding="utf-8")

    metric, winner = _parse_metric_and_winner(compare_md)

    md = render_final_report(
        metric=metric,
        winner=winner,
        compare_md=compare_md,
        baseline_md=baseline_md,
        rf_md=rf_md,
    )
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
