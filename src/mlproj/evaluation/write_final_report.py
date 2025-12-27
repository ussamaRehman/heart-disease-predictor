from __future__ import annotations

import argparse
import re
from pathlib import Path


def _parse_metric_and_winner(compare_md: str) -> tuple[str, str]:
    metric_m = re.search(r"\*\*Optimized metric \(picked on val\):\*\* `([^`]+)`", compare_md)
    winner_m = re.search(r"\*\*Winner \(by `[^`]+` on test\):\*\* `([^`]+)`", compare_md)
    if not metric_m or not winner_m:
        raise SystemExit("❌ Could not parse metric/winner from compare report")
    return metric_m.group(1), winner_m.group(1)


def _strip_pr_heading(md: str) -> str:
    md = re.sub(
        r"(?s)\A # Precision–Recall \$begin:math:text$PR\$end:math:text$ summary\s*\n+", "", md
    )
    return md.lstrip("\n")


def _strip_pr_h1(md: str) -> str:
    """Strip the PR curve file H1 so it can be embedded under our own headings."""
    lines = md.splitlines()
    if lines and lines[0].lstrip().startswith("# Precision–Recall (PR) summary"):
        lines = lines[1:]
        if lines and lines[0].strip() == "":
            lines = lines[1:]
    return "\n".join(lines).strip("\n")


def render_final_report(
    metric: str,
    winner: str,
    compare_md: str,
    baseline_md: str,
    rf_md: str,
    hgb_md: str | None = None,
    pr_baseline_md: str | None = None,
    pr_rf_md: str | None = None,
    pr_hgb_md: str | None = None,
) -> str:
    parts: list[str] = []
    parts.append("# Final report\n\nThis file aggregates the project outputs into one place.\n")
    parts.append(
        f"**Optimized metric (picked on val):** `{metric}`\n**Winner (by `{metric}` on test):** `{winner}`\n"
    )

    parts.append("## Model comparison report\n\n" + compare_md.strip() + "\n")
    parts.append("## Baseline val-tuning report\n\n" + baseline_md.strip() + "\n")
    parts.append("## Random Forest val-tuning report\n\n" + rf_md.strip() + "\n")

    if hgb_md is not None:
        parts.append("## HistGradientBoosting val-tuning report\n\n" + hgb_md.strip() + "\n")

    # PR summaries (strip the PR file's own H1 so we embed cleanly)
    if any(x is not None for x in (pr_baseline_md, pr_rf_md, pr_hgb_md)):
        pr_parts: list[str] = []
        pr_parts.append("## Precision–Recall (PR) summaries\n")
        if pr_baseline_md is not None:
            pr_parts.append("### Baseline PR summary\n\n" + _strip_pr_h1(pr_baseline_md) + "\n")
        if pr_rf_md is not None:
            pr_parts.append("### Random Forest PR summary\n\n" + _strip_pr_h1(pr_rf_md) + "\n")
        if pr_hgb_md is not None:
            pr_parts.append(
                "### HistGradientBoosting PR summary\n\n" + _strip_pr_h1(pr_hgb_md) + "\n"
            )
        parts.append("\n".join(pr_parts).strip() + "\n")

    md = "\n\n".join(parts).strip() + "\n"

    # Normalize headings: remove accidental leading whitespace before markdown headings
    md = re.sub(r"(?m)^[ \t]+(?=#+\s)", "", md)
    # Safety: if any PR H1 survived, drop it
    md = re.sub(r"(?m)^# Precision–Recall \(PR\) summary\s*\n", "", md)
    # Collapse excessive blank lines
    md = re.sub(r"\n{3,}", "\n\n", md)

    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-report", required=True)
    ap.add_argument("--rf-report", required=True)
    ap.add_argument("--compare-report", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--hgb-report", required=False)
    ap.add_argument("--pr-baseline-md", required=False)
    ap.add_argument("--pr-rf-md", required=False)
    ap.add_argument("--pr-hgb-md", required=False)
    args = ap.parse_args()

    compare_md = Path(args.compare_report).read_text(encoding="utf-8")
    metric, winner = _parse_metric_and_winner(compare_md)

    baseline_md = Path(args.baseline_report).read_text(encoding="utf-8")
    rf_md = Path(args.rf_report).read_text(encoding="utf-8")

    hgb_md = Path(args.hgb_report).read_text(encoding="utf-8") if args.hgb_report else None
    pr_baseline_md = (
        Path(args.pr_baseline_md).read_text(encoding="utf-8") if args.pr_baseline_md else None
    )
    pr_rf_md = Path(args.pr_rf_md).read_text(encoding="utf-8") if args.pr_rf_md else None
    pr_hgb_md = Path(args.pr_hgb_md).read_text(encoding="utf-8") if args.pr_hgb_md else None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_final_report(
            metric=metric,
            winner=winner,
            compare_md=compare_md,
            baseline_md=baseline_md,
            rf_md=rf_md,
            hgb_md=hgb_md,
            pr_baseline_md=pr_baseline_md,
            pr_rf_md=pr_rf_md,
            pr_hgb_md=pr_hgb_md,
        ),
        encoding="utf-8",
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
