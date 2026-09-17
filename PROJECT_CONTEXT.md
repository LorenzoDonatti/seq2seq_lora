# Project context and handoff

Updated: 2026-09-16. Protocol version: 3.

## Scientific objective and invariants

Compare dedicated per-node, parameter-shared, and integrated cross-node RSSI
forecasting for eight vineyard-2021 LoRaWAN nodes. Per-node accuracy is primary;
global means supplement it. Resource costs cover serving all eight nodes.

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
- ARIMA: eight univariate models, joint likelihood of independent hourly training
  segments per node; no concatenation across gaps; exact statsmodels filtering
  per inference window. Short excluded segments and convergence are reported.
- Joint_VARX: direct Ridge on all RSSI/weather lags; no future weather.
- SingleNode_LSTM: eight dedicated predictors.
- SharedNode_LSTM: same per-node network with shared weights, no cross-node inputs.
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

The v3 config cache rejects v1/v2 winners. New optimization is required after the
architecture/protocol revision. Without --optimize, saved winners are reused;
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

benchmark_results/optimize_32_64 contains the prior v1 reference experiment:
12 neural trials, 32 search epochs, 64 final epochs, seed 42. It remains untouched.
Its NLinear was channel-mixing and its Seq2Seq used a direct head. AR was strongest
in global MAE. These results cannot be represented as results of v2 models.
ARIMA v1 compressed discontinuous origins; v2 fixes this.

Existing test data has been inspected repeatedly: current experiments remain
exploratory. Confirmatory evidence still needs multiple temporal blocks, repeated
seeds, dependence-aware intervals and preferably a second deployment.
Hourly mean RSSI prediction does not establish per-packet ADR, energy or PDR gains.
Removing simple baselines was the user's scope decision, not evidence against them.

## Environment and validation

Local .venv created with Python 3.13.15; dependencies installed from requirements.
GPU detected and verified: NVIDIA GeForce RTX 5060 Ti, 16 GB; PyTorch 2.14.0+cu130.
Sandbox blocks GPU access; GPU runs need the approved external execution permission.
Use .venv/bin/python (the shell's generic python may not be configured). The default
history is 24 hours. The confirmatory benchmark is one-step-ahead (H=1); longer
horizons remain available only for exploratory analysis.
Run pytest and compileall after Python changes; never claim unexecuted tests passed.

Validation executed in this revision: 16 pytest tests passed; compileall and
git diff --check passed. Full CUDA smoke benchmarks completed for H=1 (2 epochs)
and H=24 (1 epoch) in revised_gpu_smoke and revised_gpu_smoke_h24. These are
execution checks, not scientific reference results.

The full v2 optimization was launched in benchmark_results/revised_gpu_seed42:
four horizons, 12 neural trials per family, 64 maximum search/final epochs,
patience 10, seed 42, explicit CUDA. At handoff it is still running. Consult
run.log for progress, run_arguments.json for the exact invocation, and
environment_requirements.txt for installed dependency versions. Do not call
this run complete until multi_horizon_summary.json exists and the process exits
successfully. Historical results are preserved.

## Run commands

```bash
.venv/bin/python -m src.cli.run_benchmark --horizon 1 \
  --optimize --trials 12 --epochs 64 --search-epochs 64 --patience 10 \
  --history 24 --device cuda --graph-ablations \
  --output-dir benchmark_results/v3_gpu_seed42
```

Always choose a fresh output directory. Do not tune based on observed test results.
