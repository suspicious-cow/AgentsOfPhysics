"""Agent responsible for loading and validating data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..adapters import (
    CSVAdapter,
    ParquetAdapter,
    InfluxDBAdapter,
    TimescaleAdapter,
    SQLiteAdapter,
    MQTTAdapter,
    OPCUAAdapter,
    SchemaRegistry,
)


@dataclass
class DataQualitySummary:
    missing_values: int
    flatlines: int
    gap_count: int
    timestamp_monotonic: bool


@dataclass
class IngestResult:
    frame: pd.DataFrame
    quality: DataQualitySummary


_ADAPTERS = {
    "csv": CSVAdapter,
    "parquet": ParquetAdapter,
    "influxdb": InfluxDBAdapter,
    "timescale": TimescaleAdapter,
    "sqlite": SQLiteAdapter,
    "mqtt": MQTTAdapter,
    "opcua": OPCUAAdapter,
}


class DataIngestor:
    """Load data according to configuration."""

    def __init__(self, schema_registry_path: str | None = None):
        self.registry = SchemaRegistry(schema_registry_path) if schema_registry_path else None

    def ingest(self, config: dict[str, Any]) -> IngestResult:
        adapter_name = config.get("adapter", "csv").lower()
        adapter_cls = _ADAPTERS.get(adapter_name)
        if adapter_cls is None:
            raise ValueError(f"Unknown adapter '{adapter_name}'")
        adapter = adapter_cls()
        params = config.get("params", {})
        frame = adapter.load(params)
        if "timestamp" in frame.columns:
            frame = frame.sort_values("timestamp").reset_index(drop=True)
            monotonic = frame["timestamp"].is_monotonic_increasing
        else:
            monotonic = True
        if config.get("target_sampling_hz") and "timestamp" in frame.columns:
            frame = self._resample(frame, float(config["target_sampling_hz"]))
        quality = self._compute_quality(frame)
        if self.registry:
            metadata = self.registry.infer_from_frame(frame)
            self.registry.persist(metadata)
        return IngestResult(frame=frame, quality=quality)

    def _resample(self, frame: pd.DataFrame, target_hz: float) -> pd.DataFrame:
        if target_hz <= 0:
            raise ValueError("target_sampling_hz must be positive")
        resampled = []
        for (asset_id, channel), group in frame.groupby(["asset_id", "channel"]):
            group = group.sort_values("timestamp")
            series = group.set_index("timestamp")
            freq = pd.Timedelta(seconds=1.0 / target_hz)
            numeric = series.select_dtypes(include=[np.number])
            resampled_numeric = numeric.resample(freq).interpolate()
            resampled_numeric["asset_id"] = asset_id
            resampled_numeric["channel"] = channel
            resampled.append(resampled_numeric.reset_index())
        return pd.concat(resampled, ignore_index=True) if resampled else frame

    def _compute_quality(self, frame: pd.DataFrame) -> DataQualitySummary:
        missing = int(frame.isna().sum().sum())
        flatlines = 0
        gap_count = 0
        if "value" in frame.columns:
            for (_, _), group in frame.groupby(["asset_id", "channel"]):
                values = group["value"].to_numpy()
                if np.allclose(values, values[0]):
                    flatlines += 1
        if "timestamp" in frame.columns:
            for (_, _), group in frame.groupby(["asset_id", "channel"]):
                ts = pd.to_datetime(group["timestamp"]).sort_values()
                gaps = ts.diff().dt.total_seconds().dropna()
                if not gaps.empty:
                    median = gaps.median()
                    gap_count += int((gaps > median * 1.5).sum())
            monotonic = frame["timestamp"].is_monotonic_increasing
        else:
            monotonic = True
        return DataQualitySummary(missing, flatlines, gap_count, monotonic)


__all__ = ["DataIngestor", "IngestResult", "DataQualitySummary"]
