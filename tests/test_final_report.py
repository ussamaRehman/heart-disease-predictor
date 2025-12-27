from __future__ import annotations

from mlproj.evaluation.write_final_report import _parse_metric_and_winner, render_final_report


def test_parse_metric_and_winner_robust() -> None:
    compare_md = """# Model comparison (val-tuned thresholds)

**Optimized metric (picked on val):** `f1`
**Winner (by `f1` on test):** `hist_gradient_boosting`
"""
    metric, winner = _parse_metric_and_winner(compare_md)
    assert metric == "f1"
    assert winner == "hist_gradient_boosting"


def test_render_final_report_includes_pr_summaries_when_given_and_strips_pr_h1() -> None:
    compare_md = """# Model comparison (val-tuned thresholds)

**Optimized metric (picked on val):** `f1`
**Winner (by `f1` on test):** `hist_gradient_boosting`
"""

    baseline_md = "# Baseline report\n- ok\n"
    rf_md = "# RF report\n- ok\n"
    hgb_md = "# HGB report\n- ok\n"

    pr_baseline_md = "# Precision–Recall (PR) summary\n\n- **Average Precision (AP):** `0.91`\n"
    pr_rf_md = "# Precision–Recall (PR) summary\n\n- **Average Precision (AP):** `0.89`\n"
    pr_hgb_md = "# Precision–Recall (PR) summary\n\n- **Average Precision (AP):** `0.94`\n"

    md = render_final_report(
        metric="f1",
        winner="hist_gradient_boosting",
        compare_md=compare_md,
        baseline_md=baseline_md,
        rf_md=rf_md,
        hgb_md=hgb_md,
        pr_baseline_md=pr_baseline_md,
        pr_rf_md=pr_rf_md,
        pr_hgb_md=pr_hgb_md,
    )

    assert "## Precision–Recall (PR) summaries" in md
    assert "### Baseline PR summary" in md
    assert "### Random Forest PR summary" in md
    assert "### HistGradientBoosting PR summary" in md

    # The embedded PR sections must NOT include the original H1
    assert "# Precision–Recall (PR) summary" not in md
