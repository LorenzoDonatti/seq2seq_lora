# Project context and handoff

Updated: 2026-09-17.

## Scientific objective and invariants

Compare dedicated per-node, parameter-shared, and integrated cross-node RSSI
forecasting in two deployments: eight vineyard-2021 nodes and nine UVA urban sensors
received by gateway A. Per-node accuracy is primary; global means supplement it.
Resource costs cover serving every node in each experiment.

Preserve chronological 70/15/15 splitting, train-only scalers, no imputation,
strictly hourly complete observed input/target windows, validation-only
hyperparameter/epoch selection. Future weather is neither used nor required.
Do not restore TCN. Do not silently fall back from optimized configurations.

## Current model set

Active models:
- ARIMAX: one Box--Jenkins historical-weather predictor per node, with a separate
  training-AICc-selected order and residual diagnostics for every link. ADF/KPSS
  restricts admissible differencing; transformed-series ACF/PACF generates a bounded
  shortlist containing the suggested order, immediate neighbours and simple forms.
- VARX: a joint predictor using all RSSI and historical-weather lags.
  Training BIC compares dense lag sets 1..p up to 24 hours and sparse seasonal sets
  {1,6,12,24} and {1,2,3,6,12,24}; stability remains mandatory.
- SingleNode_Seq2Seq: one recurrent encoder-decoder per node.
- MultiNode_Seq2Seq: LSTM encoder, LSTMCell autoregressive decoder, additive attention
  at every lead, last-observation residual. No teacher forcing in training or inference.
- NLinear/DLinear: original channel-independent shared-weight formulations.
  RSSI-only; these share parameters but do not mix channels. DLinear is a real
  decomposition model, no longer an alias for NLinear.
- PhysicalAdaptive_STGNN: hybrid physical/adaptive graph architecture, accompanied by
  fixed-configuration no-graph, physical-only and adaptive-only ablations.

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

The config cache separates winners by dataset hash, topology hash and history.
Without --optimize, compatible saved winners are reused;
--use-defaults explicitly requests internal defaults.
Result directories with completed results are protected from overwriting.

## Reports

Per-node CSV/PNG, global/terminal/per-lead metrics, predictions NPZ with origin
timestamps and node names, source/dataset hashes, environment/device metadata.
RSSI is dBm and errors use the `mae_db`/`rmse_db` keys in dB.
Latency was removed from the scientific comparison because it is too dependent on
hardware/runtime details for the paper's consolidation claim. Numeric parameter
storage excludes framework, buffers and activations; fit time excludes search.
The primary operational measure is the number of model instances required to serve
every link, supplemented by parameters and serialized weight storage.

Statistical identification is training-only. ADF/KPSS are recorded in levels and
first differences. The Johansen trace rank is checked with 1, 2 and 3 lagged
differences; a reduced rank stops level VARX fitting and requires explicit
VECM/VECMX consideration. BIC is not compared across level and differenced dependent
variables. On the current longest complete training segments, the vineyard ranks were
8/8 for all three lag settings and UVA gateway A ranks were 9/9, supporting level VAR
for these experiments despite mixed univariate level-test evidence.

## Evidence and limitations

Existing test data has been inspected repeatedly: current experiments remain
exploratory. The UVA data provide a second deployment, but confirmatory evidence still
needs multiple temporal blocks, repeated seeds, dependence-aware intervals and broader
gateway coverage.

The current UVA `optimized_seed42` exploratory run (H=1, history 24, seed 42, fresh
optimization, 16-epoch cap) placed STGNN_AdaptiveOnly first with mean MAE 1.8836 dB and
the lowest MAE on all nine nodes. PhysicalAdaptive_STGNN reached 1.9124, physical-only
1.9249, MultiNode_Seq2Seq 1.9869, no-graph 2.0341, VARX 2.0484,
SingleNode_Seq2Seq 2.0491 and ARIMAX 2.1999 dB. Treat this as exploratory: it is one
seed, adaptive-only inherited the hybrid-selected hyperparameters, its best checkpoint
was at the 16/16 epoch boundary, and ARIMAX residual Ljung--Box tests remain strongly
significant. See `docs/STGNN_EXPLICADA.md` for the architecture and interpretation.
Hourly mean RSSI prediction does not establish per-packet ADR, energy or PDR gains.
Removing simple baselines was the user's scope decision, not evidence against them.

## Environment and validation

The current sandbox `.venv` uses Python 3.12.3 and CPU PyTorch 2.14.0. A prior external
environment detected an NVIDIA GeForce RTX 5060 Ti and CUDA-enabled PyTorch; GPU runs
must record their actual environment in the generated protocol.
Use .venv/bin/python (the shell's generic python may not be configured). The default
history is 24 hours and the benchmark is one-step-ahead (H=1).
Run pytest and compileall after Python changes; never claim unexecuted tests passed.

Validation executed in this revision: 33 pytest tests passed and 2 CUDA tests were
skipped because CUDA is unavailable in the sandbox. Compileall and git diff checks
passed. The UVA one-epoch CPU smoke is an execution check, not scientific evidence.

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
  --history 24 --seed 42 --device auto --graph-ablations \
  --output-dir benchmark_results/experiment_1_vineyard/confirmatory_seed42
```

Always choose a fresh output directory. Do not tune based on observed test results.
