# Experiment 2 — UVA gateway A

External deployment with sensor01–sensor09 jointly forecast from hourly RSSI received
by gateway A. Preprocessing is fixed in `dataset/FORECASTING_VALIDATION.md`.

`smoke_cpu` used one epoch and built-in defaults. It touched the chronological test
partition solely as an end-to-end execution check; its metrics are not scientific
evidence and must not influence model, preprocessing, or hyperparameter decisions.

Every optimized/repeated run must use its own child directory. Do not mix artifacts or
configuration caches with experiment 1; protocol v4 isolates winners by dataset and
topology hashes.
