from __future__ import annotations

from mlproj.evaluation.write_final_report import _parse_metric_and_winner, render_final_report


def test_parse_metric_and_winner_robust() -> None:
    compare_md = """# Model comparison (val-tuned thresholds)

**Optimized metric (picked on val):** `f1`
**Winner (by `f1` on test):** `baseline_logreg`

| Model | Threshold | accuracy |
|---|---:|:---:|
| baseline_logreg | `0.350` | 0.804 |
| random_forest | `0.200` | 0.696 |
"""
    metric, winner = _parse_metric_and_winner(compare_md)
    assert metric == "f1"
    assert winner == "baseline_logreg"


def test_render_final_report_includes_pr_summaries_when_given() -> None:
    compare_md = """# Model comparison (val-tuned thresholds)

**Optimized metric (picked on val):** `f1`
**Winner (by `f1` on test):** `baseline_logreg`
"""

    baseline_md = "# Baseline report\n- ok\n"
    rf_md = "# RF report\n- ok\n"
    pr_baseline_md = "# PR baseline\n- AP: 0.91\n"
    pr_rf_md = "# PR rf\n- AP: 0.89\n"

    md = render_final_report(
        metric="f1",
        winner="baseline_logreg",
        compare_md=compare_md,
        baseline_md=baseline_md,
        rf_md=rf_md,
        pr_baseline_md=pr_baseline_md,
        pr_rf_md=pr_rf_md,
    )

    assert "**Winner (by `f1` on test):** `baseline_logreg`" in md
    assert "## Precision–Recall (PR) summaries" in md
    assert "### Baseline PR summary" in md
    assert "### Random Forest PR summary" in md
