"""Data ingestion agent."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..adapters import (
    CSVAdapter,
    ParquetAdapter,
    InfluxDBAdapter,
    TimescaleAdapter,
)
from ..schema import SchemaRegistry
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


_ADAPTERS = {
    "csv": CSVAdapter,
    "parquet": ParquetAdapter,
    "influxdb": InfluxDBAdapter,
    "timescale": TimescaleAdapter,
}


@dataclass
class DataIngestorResult:
    dataframe: pd.DataFrame
    schema: SchemaRegistry
    dq_report: Dict[str, float]


class DataIngestorAgent:
    """Agent responsible for loading data based on configuration."""

    def __init__(self) -> None:
        self.registry = SchemaRegistry()

    def run(self, config: Dict, input_override: Optional[str] = None) -> DataIngestorResult:
        data_cfg = config["data"]
        source_cfg = data_cfg["source"]
        adapter_type = source_cfg["type"].lower()
        adapter_cls = _ADAPTERS.get(adapter_type)
        if adapter_cls is None:
            raise ValueError(f"Unsupported data adapter: {adapter_type}")

        adapter = adapter_cls()
        params = source_cfg.get("params", {}).copy()
        if input_override:
            params["path"] = input_override
        df = adapter.load(params)

        if "timestamp" not in df.columns:
            raise ValueError("Input data must include a 'timestamp' column")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        if "asset_id" not in df.columns:
            df["asset_id"] = data_cfg.get("default_asset_id", "asset_0")
        if "channel" not in df.columns:
            df["channel"] = data_cfg.get("default_channel", "channel_0")
        if "value" not in df.columns:
            raise ValueError("Input data must include a 'value' column")

        target_rate = data_cfg.get("target_sampling_rate_hz")
        if target_rate:
            df = self._resample(df, target_rate)

        self.registry.register_from_dataframe(df)
        dq_report = self._data_quality_report(df)
        return DataIngestorResult(dataframe=df, schema=self.registry, dq_report=dq_report)

    def _resample(self, df: pd.DataFrame, target_rate: float) -> pd.DataFrame:
        LOGGER.info("Resampling data to %.2f Hz", target_rate)
        freq_ms = max(1, int(round(1000 / target_rate)))
        freq = f"{freq_ms}ms"
        frames = []
        for (asset, channel), group in df.groupby(["asset_id", "channel"], as_index=False):
            group = group.sort_values("timestamp")
            numeric = group.select_dtypes(include=["number"]).columns
            resampled = (
                group.set_index("timestamp")[numeric]
                .resample(freq)
                .interpolate()
                .reset_index()
            )
            resampled["asset_id"] = asset
            resampled["channel"] = channel
            frames.append(resampled)
        return pd.concat(frames, ignore_index=True)

    def _data_quality_report(self, df: pd.DataFrame) -> Dict[str, float]:
        report: Dict[str, float] = {}
        report["missing_rate"] = float(df.isna().mean().mean())
        flatlines = df.groupby(["asset_id", "channel"])["value"].apply(lambda x: float(x.nunique() <= 1))
        report["flatline_fraction"] = float(flatlines.mean())
        diffs = df.groupby(["asset_id", "channel"])["timestamp"].apply(lambda x: x.sort_values().diff().dt.total_seconds().dropna())
        report["median_gap_seconds"] = float(diffs.median()) if not diffs.empty else 0.0
        return report


__all__ = ["DataIngestorAgent", "DataIngestorResult"]
