# Long-term LoRaWAN Communication Metadata from an Urban Deployment

## Summary

This dataset contains long-term LoRaWAN communication metadata collected from a sensor network deployed on the University of Virginia (UVA) campus. The dataset includes detailed transmission records from **10 sensors** communicating with **3 gateways**, producing **30 unique sensor–gateway pairs**. The data spans from **February 9, 2024 to June 11, 2025** and includes metrics such as RSSI, SNR, spreading factor, frequency, airtime, and more. Weather data from a co-located weather station is also included.


---

## Files Included

### 📁 Main Data (`/dataset/lorawan_metadata/`)
- `lorawan_combined_dataset.parquet`: A single combined file including all 30 sensor–gateway pairs with full transmission metadata.
- 30 individual `.parquet` files — one for each `(Sensor, Gateway)` pair.  
  - File format: `sensorXX_gatewayY.parquet` (e.g., `sensor01_gatewayA.parquet`).  
  - Each file contains raw transmission records specific to that sensor–gateway link.

### 📁 Summary Statistics (`/dataset/summary_statistics/`)
- `lorawan_summary_by_pair.csv`: Combined summary statistics for all sensor–gateway pairs in a single CSV file.  
  - Each row includes: **Sensor Alias**, **Gateway Alias**, **Variable**, **Statistic**, and **Value**.  
  - Reported statistics: **mean**, **standard deviation (std)**, **minimum (min)**, **maximum (max)**, and **count** for each variable.

### 📁 Weather Data (`/dataset/weather/`)
- `deployment_weather.parquet`: Weather station data including temperature, humidity, pressure, wind speed, and precipitation.
  - Note: Contains a ~2-month data gap due to temporary sensor downtime.


---

## Data Description

Each row in the main `.parquet` files represents a single LoRaWAN transmission event received by one gateway. Variables include:

| Column Name               | Unit      | Description                                        |
|---------------------------|-----------|----------------------------------------------------|
| Timestamp                 | ISO 8601  | Reception timestamp in UTC.                        |
| Sensor Alias              | —         | Unique identifier of the sensor device.            |
| Gateway Alias             | —         | Unique identifier of the receiving gateway.        |
| RSSI (dBm)                | dBm       | Received signal strength.                          |
| SNR (dB)                  | dB        | Signal-to-noise ratio.                             |
| Spreading Factor (-)      | —         | LoRa modulation spreading factor.                  |
| Bandwidth (Hz)            | Hz        | Bandwidth used for the transmission.               |
| Frequency (Hz)            | Hz        | Center frequency used.                             |
| Airtime (s)               | seconds   | Time on air for the packet.                        |
| Counter (-)               | —         | Frame counter from the sensor.                     |
| # Receiving Gateways (-)  | —         | Number of gateways that received the same packet.  |


---

## Naming Conventions

- Sensors are labeled as: `sensor01` through `sensor09`, and `sensor10` 
- Gateways are labeled as: `gatewayA`, `gatewayB`, `gatewayC`


---

## Sensor and Gateway Metadata

The table below summarizes the details of all deployed sensors and gateways, including their models and geographic locations (latitude/longitude).

### Sensors
| Sensor Alias | Model               | Latitude    | Longitude    | Notes                                                                   |
|--------------|---------------------|-------------|--------------|-------------------------------------------------------------------------|
| sensor01     | Linovision S500CO2  | 38.0328110  | -78.5095524  | Inside an office on the third floor of a three-story building (indoor). |
| sensor02     | Linovision S500TH   | 38.0321764  | -78.5107493  | Near the entrance door of an office building (outdoor).                 |
| sensor03     | Linovision S500TH   | 38.0321568  | -78.5108207  | Inside a lab on the second floor of a two-story building (indoor).      |
| sensor04     | Linovision S500TH   | 38.0327080  | -78.5097230  | Near the entrance door of an office building (outdoor).                 |
| sensor05     | Linovision S500TH   | 38.0333228  | -78.5111476  | In a corridor between two office buildings (outdoor).                   |
| sensor06     | Linovision S500TH   | 38.0332850  | -78.5111832  | Inside an office on the third floor of a three-story building (indoor). |
| sensor07     | Linovision S500TH   | 38.0328110  | -78.5095524  | Inside an office on the third floor of a three-story building (indoor). |
| sensor08     | Linovision S500TH   | 38.0347279  | -78.5135257  | Near the entrance door of an office building (outdoor).                 |
| sensor09     | Linovision S500TH   | 38.0346453  | -78.5134932  | Inside a classroom in a single-story office building (indoor).          |
| sensor10     | Linovision S500TH   | 38.0318946  | -78.5102145  | Inside an office on the second floor of a two-story building (indoor).  |

### Gateways
| Gateway Alias | Model                     | Latitude    | Longitude    | Notes                                                          |
|---------------|---------------------------|-------------|--------------|----------------------------------------------------------------|
| gatewayA      | Cisco IXM-LPWA-900-16-K9  | 38.0317195  | -78.5108640  | Deployed on the roof of a five-story building (outdoor).       |
| gatewayB      | Cisco IXM-LPWA-900-16-K9  | 38.0365606  | -78.5225695  | Deployed on the roof of a single-story building (outdoor).     |
| gatewayC      | Cisco IXM-LPWA-900-16-K9  | 38.0322023  | -78.5106216  | Deployed on the second floor of a two-story building (indoor). |


---

## File Format

- All transmission data stored as **Apache Parquet** (`.parquet`) for compact, efficient access  
- Summary statistics provided as **CSV** (`lorawan_summary_by_pair.csv`)  
- Weather data provided as **Parquet** (`deployment_weather.parquet`)  


---

## Temporal Coverage

- **Start date:** 2024-02-09  
- **End date:** 2025-06-11  
- Timezone: All timestamps are converted to UTC.


---

## License

This dataset is made available under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. You are free to use, share, and adapt the data, with appropriate attribution.


---

## Citation

> Fateme Nikseresht; Victor Ariel Leal Sobral; Jonathan L. Goodall; Bradford Campbell, 2025, "A LoRaWAN Communication Metadata Dataset from an Urban Deployment", https://doi.org/10.18130/V3/RFTICK, University of Virginia Dataverse


---

## Contact

For questions or collaboration, please contact:  
**Fateme Nikseresht**  
University of Virginia  
Email: [fn5an@virginia.edu]
