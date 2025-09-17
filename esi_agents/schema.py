"""Schema registry for signal metadata."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from .utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class SignalMetadata:
    """Metadata describing a monitored signal channel."""

    asset_id: str
    channel: str
    unit: str = "unknown"
    sampling_rate_hz: float = 1.0
    channel_type: str = "unknown"
    phase: Optional[str] = None


class SchemaRegistry:
    """Registry that manages signal metadata for datasets."""

    def __init__(self) -> None:
        self._registry: Dict[str, SignalMetadata] = {}

    def register(self, metadata: SignalMetadata) -> None:
        key = self._key(metadata.asset_id, metadata.channel)
        LOGGER.debug("Registering metadata for %s", key)
        self._registry[key] = metadata

    def register_from_dataframe(
        self,
        df: pd.DataFrame,
        asset_column: str = "asset_id",
        channel_column: str = "channel",
        value_column: str = "value",
        timestamp_column: str = "timestamp",
        unit_column: Optional[str] = "unit",
        channel_type_column: Optional[str] = "channel_type",
    ) -> None:
        """Infer and register metadata from a dataframe."""

        if timestamp_column not in df.columns:
            raise ValueError("DataFrame must contain a timestamp column")

        df = df.copy()
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        grouped = df.groupby([asset_column, channel_column])
        for (asset_id, channel), group in grouped:
            deltas = group[timestamp_column].sort_values().diff().dropna()
            if not deltas.empty:
                sampling_rate = 1.0 / deltas.dt.total_seconds().median()
            else:
                sampling_rate = 1.0
            unit = None
            if unit_column and unit_column in group:
                unit_values = group[unit_column].dropna().unique()
                unit = unit_values[0] if len(unit_values) else None
            channel_type = None
            if channel_type_column and channel_type_column in group:
                type_values = group[channel_type_column].dropna().unique()
                channel_type = type_values[0] if len(type_values) else None
            metadata = SignalMetadata(
                asset_id=str(asset_id),
                channel=str(channel),
                unit=str(unit) if unit else "unknown",
                sampling_rate_hz=float(sampling_rate),
                channel_type=str(channel_type) if channel_type else "unknown",
            )
            self.register(metadata)

    def get(self, asset_id: str, channel: str) -> Optional[SignalMetadata]:
        return self._registry.get(self._key(asset_id, channel))

    def to_list(self) -> Iterable[SignalMetadata]:
        return list(self._registry.values())

    def save(self, path: Path) -> None:
        LOGGER.info("Persisting schema registry to %s", path)
        data = [asdict(meta) for meta in self.to_list()]
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data).to_json(path, orient="records", indent=2)

    @staticmethod
    def _key(asset_id: str, channel: str) -> str:
        return f"{asset_id}::{channel}"


__all__ = ["SchemaRegistry", "SignalMetadata"]
