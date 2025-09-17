"""Data adapters for ESI ingestion."""
from .base import AdapterNotAvailable, BaseAdapter
from .csv import CSVAdapter
from .parquet import ParquetAdapter
from .influxdb import InfluxDBAdapter
from .timescale import TimescaleAdapter
from .sqlite import SQLiteAdapter
from .mqtt import MQTTAdapter
from .opcua import OPCUAAdapter
from .schema_registry import SchemaRegistry, SignalMetadata

__all__ = [
    "AdapterNotAvailable",
    "BaseAdapter",
    "CSVAdapter",
    "ParquetAdapter",
    "InfluxDBAdapter",
    "TimescaleAdapter",
    "SQLiteAdapter",
    "MQTTAdapter",
    "OPCUAAdapter",
    "SchemaRegistry",
    "SignalMetadata",
]
