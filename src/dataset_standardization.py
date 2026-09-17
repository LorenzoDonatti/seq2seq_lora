"""Causal standardization of the UVA packet-level deployment for RSSI forecasting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


UVA_SENSORS = [f"sensor{i:02d}" for i in range(1, 10)]


def _hourly_rain(accumulated: pd.Series) -> pd.Series:
    """Convert an accumulated rain gauge into non-negative hourly increments."""
    delta = accumulated.diff()
    return delta.where(delta >= 0, accumulated).clip(lower=0)


def prepare_uva_gateway_a(
    source_dir: str = "dataset",
    output_file: str = "data/experiment_2_uva_gatewayA_hourly.csv",
) -> Dict[str, Any]:
    """Create a dense hourly table; missing measurements remain missing."""
    source = Path(source_dir)
    radio_path = source / "lorawan_metadata/lorawan_combined_dataset.parquet"
    weather_path = source / "weather/deployment_weather.parquet"
    metadata = json.loads((source / "deployment_metadata.json").read_text(encoding="utf-8"))
    radio = pd.read_parquet(radio_path)
    weather = pd.read_parquet(weather_path)

    radio = radio[
        (radio["Gateway Alias"] == "gatewayA")
        & radio["Sensor Alias"].isin(UVA_SENSORS)
    ].copy()
    radio["timestamp"] = radio["Timestamp"].dt.floor("h")
    hourly_rssi = radio.pivot_table(
        index="timestamp", columns="Sensor Alias", values="RSSI (dBm)", aggfunc="mean"
    ).reindex(columns=UVA_SENSORS)
    hourly_rssi.columns = [f"RSSI_{i:02d}" for i in range(1, 10)]

    weather = weather.sort_values("Timestamp").set_index("Timestamp")
    hourly_weather = pd.DataFrame({
        "temp": weather["Temperature ( F )"].resample("h").mean(),
        "hum": weather["Humidity ( RH )"].resample("h").mean(),
        "bar": weather["Barometric Pressure ( INHG )"].resample("h").mean(),
        "rain": _hourly_rain(weather["Accumulated Rain ( IN )"].resample("h").last()),
    })

    start = min(hourly_rssi.index.min(), hourly_weather.index.min())
    end = max(hourly_rssi.index.max(), hourly_weather.index.max())
    hourly_index = pd.date_range(start.floor("h"), end.floor("h"), freq="h", tz="UTC")
    combined = hourly_rssi.reindex(hourly_index).join(hourly_weather.reindex(hourly_index))
    combined.index.name = "timestamp"
    combined = combined.reset_index()
    combined["timestamp"] = combined["timestamp"].astype(str)

    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, sep=";", index=False, lineterminator="\n")

    sensor_by_id = {item["id"]: item for item in metadata["sensors"]}
    gateway = next(item for item in metadata["gateways"] if item["id"] == "gatewayA")
    topology = {
        "dataset": "UVA urban LoRaWAN deployment",
        "dataset_doi": metadata["dataset_doi"],
        "gateway": "gatewayA",
        "target_names": [f"RSSI_{i:02d}" for i in range(1, 10)],
        "source_nodes": UVA_SENSORS,
        "node_coordinates": [
            [sensor_by_id[node]["latitude"], sensor_by_id[node]["longitude"]]
            for node in UVA_SENSORS
        ],
        "gateway_coordinates": [gateway["latitude"], gateway["longitude"]],
    }
    topology_path = destination.with_suffix(".topology.json")
    topology_path.write_text(json.dumps(topology, indent=2), encoding="utf-8")

    report = {
        "experiment": "experiment_2_uva_gatewayA",
        "source": str(radio_path),
        "weather_source": str(weather_path),
        "selection": {
            "gateway": "gatewayA",
            "included_sensors": UVA_SENSORS,
            "excluded_sensor": "sensor10",
            "reason": "sensor10 has only 384 observed hours versus about 11,300 for sensors01-09",
            "excluded_gateways": ["gatewayB", "gatewayC"],
            "gateway_reason": "their sensor-link coverage is too sparse for common complete-window forecasting",
        },
        "aggregation": {
            "frequency": "1 hour UTC",
            "rssi": "arithmetic mean of packet RSSI values within each sensor-hour",
            "weather": "hourly mean for temperature/humidity/pressure; causal increment of accumulated rain",
            "missing_data": "preserved as NaN; no interpolation or imputation",
        },
        "rows": len(combined),
        "range": [combined["timestamp"].iloc[0], combined["timestamp"].iloc[-1]],
        "observed_hour_fraction": {
            column: float(combined[column].notna().mean())
            for column in topology["target_names"] + ["temp", "hum", "bar", "rain"]
        },
        "output_sha256": __import__("hashlib").sha256(destination.read_bytes()).hexdigest(),
    }
    destination.with_suffix(".standardization.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report

