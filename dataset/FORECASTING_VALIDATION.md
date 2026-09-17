# RSSI forecasting validation for the UVA deployment

## Confirmatory role

This deployment is the external `experiment_2`; the vineyard is `experiment_1`.
Results, optimized configurations, and conclusions must remain separated by dataset.
The UVA test partition must not be used to choose history, architecture, search space,
epoch budget, patience, included links, or preprocessing.

## Link selection fixed before modeling

The prediction unit is the hourly RSSI from sensors 01–09 as received by `gatewayA`.
This yields nine simultaneous series observed by one gateway, matching the scientific
question of forecasting every node in a network with one integrated model.

- Gateway A has approximately 11,300 observed hours for each sensor 01–09.
- Sensor10 is excluded because it entered near the end and has only 384 observed hours.
- Gateways B and C are excluded because several sensor–gateway links are extremely
  sparse; gateway B has no hour containing eight available sensors.
- These choices are based only on availability, not forecasting performance.

Using all 30 sensor–gateway pairs would change the object from a node graph to a sparse
link graph and would make complete-window evaluation depend mainly on packet reception.
That is a valuable future task, but it is not comparable to experiment 1.

## Causal hourly standardization

`lora-prepare-uva` creates `data/experiment_2_uva_gatewayA_hourly.csv` and two sidecars.

- UTC timestamps are floored to hourly bins.
- Target RSSI is the arithmetic mean of packets for each sensor within the hour.
- Temperature, humidity, and barometric pressure are hourly means.
- Rain is the non-negative increment of the accumulated rain gauge using values observed
  through the end of the hour.
- No RSSI or weather value is interpolated or imputed.
- A forecast origin uses completed historical hours to predict the next hourly mean.
- A sample is eligible only if its complete input/target window is finite and strictly
  hourly, using the same policy as experiment 1.

Observed sensor-hours contain a median of 11 packets (5th–95th percentile: 8–12).
Only about 0.086% contain fewer than three packets, so the hourly mean is ordinarily
supported by repeated receptions rather than a single packet. Packet count, frequency,
SF, and SNR are deliberately not predictors in this first cross-dataset comparison;
adding them would change the information set relative to experiment 1.

With history 24 and H=1, the current standardized data produce:

| Split | Valid windows | Discarded candidate windows |
|---|---:|---:|
| Train | 7,145 | 1,046 |
| Validation | 1,369 | 367 |
| Test | 1,015 | 722 |

Weather contains several real outages, including a 790-hour gap. Preserving these gaps
is intentional. Complete-case results apply to periods when all nine links and weather
are simultaneously observable; they do not establish robustness to missing nodes.

## Dataset shift and dependence audit

The raw chronological splits exhibit substantial level shift. For example, sensor02 has
mean RSSI approximately -101.15 dBm in train, -94.37 dBm in validation, and -101.47 dBm
in test. Similar shifts occur for several nodes. Training-only scaling is therefore
essential, and random cross-validation would be invalid.

Within training, pairwise level correlations have median 0.872; correlations of hourly
first differences have median 0.659. This is a strong, pre-modeling reason to evaluate
joint predictors, but correlation is not causality and may reflect shared propagation,
gateway behavior, channel hopping, weather, or deployment-wide changes.

## Topology

The sidecar maps `RSSI_01`–`RSSI_09` to sensor01–sensor09 coordinates and records the
gateway A coordinates. The physical graph uses sensor-to-sensor Haversine distance; the
gateway-distance feature uses sensor-to-gateway distance. Sensor01 and sensor07 are
co-located but use different devices, providing a useful check of whether learned
dependencies capture more than geographic distance.

The UVA deployment mixes indoor and outdoor sensors. Straight-line distance is only a
prior, not a propagation or connectivity claim. Building obstruction/height and LoS are
not available as quantitative edge attributes.

## Execution separation

Experiment 1 artifacts:

```text
benchmark_results/experiment_1_vineyard/
```

Experiment 2 artifacts:

```text
benchmark_results/experiment_2_uva_gatewayA/
```

The existing `smoke_cpu` result used one epoch and built-in defaults. It validates
execution only and must never be cited as scientific evidence.
