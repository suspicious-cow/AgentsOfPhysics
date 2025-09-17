# Data Adapters

Adapters expose a consistent contract for batch loading (`load(params) -> DataFrame`) and streaming (`subscribe(params) -> AsyncIterator[dict]`).

| Adapter | Description |
| --- | --- |
| `CSVAdapter` | Reads CSV files and optionally simulates streaming by yielding rows asynchronously. |
| `ParquetAdapter` | Loads Parquet files and can simulate streaming similar to the CSV adapter. |
| `InfluxDBAdapter` | Stub adapter for InfluxDB; requires the optional `influxdb-client` dependency. |
| `TimescaleAdapter` | Uses SQLAlchemy to issue SQL queries against TimescaleDB. |
| `MQTTAdapter` | Consumes MQTT topics using `paho-mqtt` and yields decoded JSON payloads. |
| `OPCUAAdapter` | Demonstrates how to query OPC-UA nodes using `python-opcua`. |

Each adapter returns time-indexed data with columns `[timestamp, asset_id, channel, value]` so downstream agents can enforce schema and resampling logic.
