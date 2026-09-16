# Codex project handoff

Before changing this repository, read `PROJECT_CONTEXT.md` and `README.md` completely.
`PROJECT_CONTEXT.md` records the scientific intent, methodological decisions, current
execution semantics, and the latest reference results.

Important invariants:

- Per-node metrics are primary; global averages are supplementary.
- The main hypothesis concerns one integrated model forecasting every node jointly and
  its accuracy/resource trade-off against eight dedicated predictors.
- Never use the test partition for hyperparameter or epoch selection.
- Never silently fall back from optimized configurations to built-in defaults.
- `--optimize` is part of `lora-benchmark`, not a separate program.
- Do not restore the removed TCN model.
- Preserve chronological splitting, train-only scalers, complete strictly-hourly
  windows, and no imputation across missing values or time gaps.

Run at least `python -m compileall -q src tests` after Python changes. The local virtual
environment may not contain pytest; do not claim that the pytest suite passed unless it
was actually available and executed.
