# Data Adapters

The platform provides a unified adapter interface with a batch `load` method returning a `pandas.DataFrame` and a streaming `subscribe` async generator.

## File-based

- `CSVAdapter`
- `ParquetAdapter`

Both adapters support optional renaming and timestamp parsing.

## Time-series stores

- `InfluxDBAdapter`
- `TimescaleAdapter`
- `SQLiteAdapter`

These adapters rely on injected callables for query execution to keep optional dependencies light-weight.

## Industrial transports

- `MQTTAdapter`
- `OPCUAAdapter`

The transports require their respective optional dependencies and expose async streams that can be consumed by the streaming scorer. Unit tests validate that clear errors are raised when dependencies are missing.
