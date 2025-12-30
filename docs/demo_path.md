# 60-Second Demo Path

Use this checklist for a quick portfolio walkthrough:

1) README
   - Project summary and quickstart.
2) Results snapshot
   - `docs/results_snapshot.md` (committed metrics table).
3) Model card
   - `reports/model_card.md` (intended use, data, metrics).
4) Release proof bundle
   - Latest Release asset: `reports_bundle_<tag>.zip` from `../../releases/latest`.
5) CI workflow value step
   - `make report-e2e VAL_BEST_METRIC=f1` in `.github/workflows/ci.yml`.
