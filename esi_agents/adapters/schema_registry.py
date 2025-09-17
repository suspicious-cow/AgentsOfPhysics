"""Simple schema registry for ESI signals."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class SignalMetadata:
    """Metadata describing a telemetry channel."""

    asset_id: str
    channel: str
    unit: str | None = None
    sampling_rate_hz: float | None = None
    channel_type: str | None = None
    phase: str | None = None


class SchemaRegistry:
    """Persist and retrieve :class:`SignalMetadata` for channels."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def infer_from_frame(
        self, df: pd.DataFrame, defaults: dict[str, Any] | None = None
    ) -> list[SignalMetadata]:
        defaults = defaults or {}
        if not {"asset_id", "channel"}.issubset(df.columns):
            raise ValueError("DataFrame must include 'asset_id' and 'channel'")
        metadata: list[SignalMetadata] = []
        for (asset_id, channel), group in df.groupby(["asset_id", "channel"]):
            sampling_rate = None
            if "timestamp" in group.columns:
                timestamps = pd.to_datetime(group["timestamp"]).sort_values()
                if len(timestamps) > 1:
                    deltas = timestamps.diff().dropna().dt.total_seconds()
                    if not deltas.empty:
                        sampling_rate = 1.0 / deltas.mean()
            metadata.append(
                SignalMetadata(
                    asset_id=str(asset_id),
                    channel=str(channel),
                    unit=defaults.get("unit"),
                    sampling_rate_hz=defaults.get("sampling_rate_hz", sampling_rate),
                    channel_type=defaults.get("channel_type"),
                    phase=defaults.get("phase"),
                )
            )
        return metadata

    def persist(self, metadata: list[SignalMetadata]) -> None:
        serialised = [asdict(item) for item in metadata]
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(serialised, fh, indent=2)

    def load(self) -> list[SignalMetadata]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return [SignalMetadata(**item) for item in raw]


__all__ = ["SchemaRegistry", "SignalMetadata"]
