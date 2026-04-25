# Test Automation Summary

## Generated Tests

### CLI E2E Tests
- [x] `tests/test_cli_e2e.py` - `main.py` end-to-end run exports manifest, metrics, and checkpoint artifacts
- [x] `tests/test_cli_e2e.py` - `python -m inference` loads a registered canonical specialist and exports prediction quantiles
- [x] `tests/test_cli_e2e.py` - `python -m inference --plot` generates `fan_chart_*.png` and `calibration_*.png`
- [x] `tests/test_cli_e2e.py` - `scripts/verify_cp_walkforward.py` full mode launches training and validates the emitted manifest
- [x] `tests/test_cli_e2e.py` - controlled positive `main.py --save-canonical` path registers a specialist and a real `inference` run consumes it
- [x] `tests/test_cli_e2e.py` - minimal `tune_hyperparams.py` CLI orchestration persists an Optuna study and result CSVs
- [x] `tests/test_cli_e2e.py` - `scripts/verify_cp_walkforward.py --report-only` reports CP status from a real manifest

## Coverage
- Existing feature workflows covered: 7
- Framework used: `pytest`
- Scope: real CLI subprocess execution in isolated temporary workdirs

## Notes
- No `_bmad/bmm/config.yaml` was present in this repo, so this summary uses the fallback path `docs/tests/test-summary.md`
- The CLI suite is marked `slow` because it launches real subprocess workflows
- `verify_cp_walkforward.py` is covered through a real negative-path manifest report on a tiny fixture run
- `inference --plot` is now covered with real PNG artifact generation in an isolated output directory
- The `--save-canonical` positive path uses a controlled wrapper to make the portfolio gate deterministic while preserving real artifact generation and registry consumption
- The minimal `tune_hyperparams.py` E2E uses a wrapper to stabilize the expensive training subprocess while still exercising the real CLI orchestration path

## Next Steps
- Run `python -m pytest -q tests/test_cli_e2e.py -m slow`
- Decide whether any of these CLI E2E checks should graduate into CI or remain opt-in
- If wanted, replace the wrapped `tune_hyperparams.py` E2E with a fully unwrapped tiny-study path once a stable low-cost training configuration exists
