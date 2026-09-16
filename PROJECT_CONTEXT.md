# Project context and handoff

Last updated: 2026-09-16 (America/Sao_Paulo).

## Scientific objective

This project benchmarks RSSI forecasting for eight LoRaWAN nodes. The central question
is whether a single integrated multi-node model can forecast all nodes simultaneously
with competitive accuracy while using fewer deployed model instances than one model per
node. Therefore, accuracy must be interpreted together with total parameters, weight
storage, training cost, inference cost for all nodes, and number of predictors.

Per-node MAE/RMSE is the primary result. A global average can hide weak nodes and is only
supplementary. Forecast horizons are H=1, 6, 12, and 24 hours.

## Data protocol that must be preserved

- Dataset: `data/combined_hourly_data.csv` (`vineyard-2021`).
- Chronological split: 70% train, 15% validation, 15% test.
- Scalers are fitted using training observations only.
- No imputation is performed.
- A window is valid only when every input and target is observed and timestamps are
  strictly consecutive at one-hour intervals.
- Hyperparameters are selected on validation only. Test remains untouched until the
  final benchmark.
- The task is direct multi-horizon forecasting from observed history. Reports include
  global, terminal-horizon, per-lead-time, and per-node metrics.
- Default seed is 42 and default history is 24 hours.

The peer-reviewed reference paper provided by the user is
`1-s2.0-S1389128624000902-main.pdf`. It is methodological context, not a requirement to
reproduce the paper's algorithms exactly.

## Current models

- Persistence.
- AR and ARIMA: eight dedicated univariate predictors; reported training time is the
  aggregate required to serve all nodes.
- Joint VAR and Joint VARX: one statistical predictor for all nodes.
- SingleNode LSTM: eight dedicated neural predictors.
- MultiNode Seq2Seq with temporal attention: one integrated predictor.
- NLinear: one integrated linear predictor.
- Physical/adaptive STGNN: one integrated graph predictor.

TCN was intentionally removed at the user's request and must not be restored.

## CLI and configuration semantics

There is one public command: `lora-benchmark`.

- `--optimize` performs validation-only selection independently for each requested
  horizon, runs the final benchmark, and persists every winning model configuration in
  `.lora_benchmark/last_optimized_configs.json`.
- Without `--optimize`, the CLI automatically reuses the latest saved optimized
  configuration for each horizon.
- `--use-defaults` is the only explicit way to ignore saved winners.
- Missing or dataset/history-incompatible saved configurations must cause an actionable
  error; never silently use defaults.
- The cache identifies the dataset by SHA-256, so it remains valid after moving the
  repository to another machine.
- Each benchmark protocol records `hyperparameter_source` as `fresh_optimization`,
  `last_saved_optimization`, or `built_in_defaults`.
- `--search-epochs` applies to each neural candidate during selection. `--epochs`
  applies to the final fit. Statistical models do not use epochs.

The editable installation command for an offline environment with dependencies already
installed is:

```bash
pip install -e . --no-deps --no-build-isolation
```

## Latest reference experiment

The newest complete experiment is `benchmark_results/optimize_32_64/`:

- Horizons: 1, 6, 12, 24.
- Optimization: 12 trials per neural model, 32 search epochs per candidate.
- Final training: 64 epochs.
- Seed: 42.
- History: 24 hours.
- Protocol source: `fresh_optimization`.
- Search reports, protocols, per-node CSV/PNG files, predictions, and consolidated
  results are all retained in that directory.

Global MAE in dBm from this run (lower is better):

| Model | H=1 | H=6 | H=12 | H=24 |
|---|---:|---:|---:|---:|
| AR | 0.6769 | 0.9585 | 1.1530 | 1.3117 |
| ARIMA | 0.7044 | 1.0301 | 1.2795 | 1.4971 |
| Joint VAR | 0.7273 | 1.1195 | 1.3456 | 1.5083 |
| Joint VARX | 0.8423 | 1.5455 | 2.1913 | 2.4110 |
| SingleNode LSTM | 0.7466 | 1.0502 | 1.3066 | 1.4709 |
| MultiNode Seq2Seq | 0.8193 | 1.0851 | 1.3055 | 1.5758 |
| NLinear | 0.8741 | 1.1439 | 1.3160 | 1.4939 |
| PhysicalAdaptive STGNN | 0.6977 | 1.0086 | 1.3271 | 1.4998 |
| Persistence | 0.9246 | 1.1366 | 1.4114 | 1.5942 |

Interpretation at this checkpoint:

- AR remains the strongest global-MAE baseline across all four horizons.
- STGNN is highly competitive at H=1 and H=6 while serving every node with one model.
- At H=12, Seq2Seq nearly matches the dedicated SingleNode LSTM.
- At H=24, SingleNode LSTM is the strongest neural model; NLinear and STGNN are close to
  ARIMA and use one integrated predictor.
- Joint VARX deteriorates substantially at longer horizons and needs methodological
  review before any positive claim.
- Claims must still be checked node by node in `metrics_per_node_H*.csv`; global MAE
  alone is insufficient.

The exact selected hyperparameters live in `benchmark_protocol_H*.json` and the current
portable cache. Do not copy them manually into source defaults.

## Known lessons and pitfalls

- A prior apparent regression was caused by comparing an optimized run with a run that
  silently used built-in defaults. This motivated persistent optimized configurations.
- More final epochs did not uniformly improve test metrics. Do not select the epoch
  budget by repeatedly inspecting test performance; tune stopping behavior on validation.
- A large mismatch between search epochs and final epochs can favor fast-learning
  candidates rather than candidates that generalize best after long training.
- Always use a new output directory for a distinct experiment to avoid overwriting
  evidence. Keep the protocol JSON beside each result.
- Early/best validation weights are restored by the neural trainers, and training
  summaries record requested and best epochs. Verify these fields when results look
  anomalous.

## Useful commands

Reuse the current winners without re-running optimization:

```bash
lora-benchmark \
  --horizons 1 6 12 24 \
  --epochs 64 \
  --seed 42 \
  --output-dir benchmark_results/reuse_latest_seed42
```

Run a new independent optimization per horizon:

```bash
lora-benchmark \
  --horizons 1 6 12 24 \
  --optimize \
  --trials 12 \
  --search-epochs 32 \
  --epochs 64 \
  --seed 42 \
  --output-dir benchmark_results/new_optimization_seed42
```

Use built-in defaults only for an intentional control experiment:

```bash
lora-benchmark --horizons 1 6 12 24 --use-defaults
```

## Safe next steps

- Compare per-node improvements and failures for integrated versus dedicated models.
- Add repeated seeds and uncertainty intervals before making strong scientific claims.
- Define a validation-only policy for epoch budget/early stopping.
- Preserve the resource comparison for serving all eight nodes; latency alone is not a
  sufficient deployment metric.
