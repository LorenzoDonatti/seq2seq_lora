# Project context and handoff

Updated: 2026-09-17. Protocol version: 4.

## Scientific objective and invariants

Compare dedicated per-node, parameter-shared, and integrated cross-node RSSI
forecasting in two deployments: eight vineyard-2021 nodes and nine UVA urban sensors
received by gateway A. Per-node accuracy is primary; global means supplement it.
Resource costs cover serving every node in each experiment.

Preserve chronological 70/15/15 splitting, train-only scalers, no imputation,
strictly hourly complete observed input/target windows, validation-only
hyperparameter/epoch selection. Future weather is neither used nor required.
Do not restore TCN. Do not silently fall back from optimized configurations.

## User-authorized revision

The user requested paired ARX/ARIMAX, VARX/VARIMAX, dedicated/integrated Seq2Seq,
NLinear and DLinear, plus the multi-node graph contribution. AR, persistence and
plain VAR were removed.
The user explicitly requested retaining the Seq2Seq name and replacing the direct
projection architecture with a literature-consistent recurrent encoder-decoder.

Active models:
- ARX/ARIMAX: one historical-weather predictor per node.
- VARX/VARIMAX: joint predictors using all RSSI and historical-weather lags.
- SingleNode_Seq2Seq: one recurrent encoder-decoder per node.
- MultiNode_Seq2Seq: LSTM encoder, LSTMCell autoregressive decoder, additive attention
  at every lead, last-observation residual. No teacher forcing in training or inference.
- NLinear/DLinear: original channel-independent shared-weight formulations.
  RSSI-only; these share parameters but do not mix channels. DLinear is a real
  decomposition model, no longer an alias for NLinear.
- PhysicalAdaptive_STGNN: existing physical/adaptive graph architecture.

SharedNode_LSTM is a useful control, but Seq2Seq/STGNN comparisons still confound
architecture and cross-node input. Do not claim this fully isolates all effects.

## Training and execution semantics

Public command: lora-benchmark. --optimize is a flag, not a separate program.
Default --epochs 64, --patience 10; omitted --search-epochs equals --epochs.
All neural training and checkpoint selection use macro physical-unit MAE.
Best validation weights are restored; summaries record executed and best epochs.
--device auto chooses CUDA if available; explicit cuda errors if unavailable.
Statistical models stay on CPU. No runtime failure triggers a CPU fallback.
CUDA RNG, cuBLAS workspace and deterministic torch algorithms are configured.

The v4 config cache separates winners by dataset hash, topology hash, history and
horizon. Without --optimize, compatible saved winners are reused;
--use-defaults explicitly requests internal defaults.
Result directories with completed results are protected from overwriting.

## Reports

Per-node CSV/PNG, global/terminal/per-lead metrics, predictions NPZ with origin
timestamps and node names, source/dataset hashes, environment/device metadata.
RSSI is dBm, error is dB; legacy JSON keys mae_dbm/rmse_dbm retain their names.
Latencies: batches 1 and 32, all nodes, median 30 runs/5 warmups, synchronized CUDA,
including host/device transfers. Numeric parameter storage excludes framework,
buffers and activations; fit time excludes hyperparameter search.

## Historical evidence and limitations

Older v1 results used a channel-mixing NLinear, a direct-head Seq2Seq, and statistical
handling that compressed discontinuous origins. They cannot be represented as results
of the current models and are not retained as current evidence.

Existing test data has been inspected repeatedly: current experiments remain
exploratory. Confirmatory evidence still needs multiple temporal blocks, repeated
seeds, dependence-aware intervals and preferably a second deployment.
Hourly mean RSSI prediction does not establish per-packet ADR, energy or PDR gains.
Removing simple baselines was the user's scope decision, not evidence against them.

## Environment and validation

The current sandbox `.venv` uses Python 3.12.3 and CPU PyTorch 2.14.0. A prior external
environment detected an NVIDIA GeForce RTX 5060 Ti and CUDA-enabled PyTorch; GPU runs
must record their actual environment in the generated protocol.
Use .venv/bin/python (the shell's generic python may not be configured). The default
history is 24 hours. The confirmatory benchmark is one-step-ahead (H=1); longer
horizons remain available only for exploratory analysis.
Run pytest and compileall after Python changes; never claim unexecuted tests passed.

Validation executed in this revision: 34 pytest tests passed and 2 CUDA tests were
skipped because CUDA is unavailable in the sandbox. Compileall and git diff checks
passed. The UVA one-epoch CPU smoke is an execution check, not scientific evidence.

The completed v3 vineyard run is preserved at
`benchmark_results/experiment_1_vineyard/v3_h1_gpu_seed42`. It used history 13 and
patience 64 and therefore remains exploratory rather than the frozen confirmatory run.

The external UVA dataset is stored under `dataset/` and standardized to
`data/experiment_2_uva_gatewayA_hourly.csv`. Experiment 2 uses gateway A and sensors
01-09; sensor10 and gateways B/C were excluded by coverage criteria fixed before
modeling. See `dataset/FORECASTING_VALIDATION.md`. With history 24 and H=1 there are
7,145/1,369/1,015 valid train/validation/test windows. A one-epoch CPU smoke completed
for all paired models and graph ablations at
`benchmark_results/experiment_2_uva_gatewayA/smoke_cpu`; it is not scientific evidence.

## Run commands

```bash
.venv/bin/python -m src.cli.run_benchmark --horizon 1 \
  --optimize --trials 12 --epochs 64 --search-epochs 64 --patience 10 \
  --history 24 --device auto --graph-ablations \
  --output-dir benchmark_results/experiment_1_vineyard/confirmatory_seed42
```

Always choose a fresh output directory. Do not tune based on observed test results.
