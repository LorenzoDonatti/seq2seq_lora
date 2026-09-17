# Experiment 2 — UVA gateway A

External deployment with sensor01–sensor09 jointly forecast from hourly RSSI received
by gateway A. Preprocessing is fixed in `dataset/FORECASTING_VALIDATION.md`.

`smoke_cpu` used one epoch and built-in defaults. It touched the chronological test
partition solely as an end-to-end execution check; its metrics are not scientific
evidence and must not influence model, preprocessing, or hyperparameter decisions.

Every optimized/repeated run must use its own child directory. Do not mix artifacts or
configuration caches with experiment 1. The current cache isolates winners by dataset
hash, topology hash and history length, and accepts only the complete active model set.

`optimized_seed42` is an exploratory H=1 run with history 24, fresh optimization,
16 training epochs and graph ablations. STGNN_AdaptiveOnly obtained the lowest mean MAE
(1.8836 dB) and the lowest per-node MAE for all nine nodes. This is not yet a frozen
confirmatory result: it uses one seed, several neural checkpoints reached the epoch
limit, and the ARIMAX residual diagnostics indicate remaining serial correlation.
