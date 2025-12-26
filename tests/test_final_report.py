from mlproj.evaluation.write_final_report import render_final_report


def test_render_final_report_contains_sections_and_header() -> None:
    md = render_final_report(
        metric="f1",
        winner="baseline_logreg",
        compare_md="# Model comparison\n\nWinner: baseline",
        baseline_md="# Baseline report",
        rf_md="# RF report",
    )
    assert "# Final report" in md
    assert "## Model comparison report" in md
    assert "## Baseline val-tuning report" in md
    assert "## Random Forest val-tuning report" in md
    assert "`f1`" in md
    assert "`baseline_logreg`" in md
