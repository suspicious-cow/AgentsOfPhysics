"""CSV adapter for batch and streaming ingestion."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseAdapter


class CSVAdapter(BaseAdapter):
    """Adapter capable of reading comma separated files."""

    def load(self, params: Mapping[str, Any]) -> pd.DataFrame:
        path = Path(params["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, **params.get("read_csv_kwargs", {}))
        timestamp_column = params.get("timestamp_column")
        if timestamp_column and timestamp_column in df.columns:
            df["timestamp"] = pd.to_datetime(df[timestamp_column], format='ISO8601', errors='coerce')
            if timestamp_column != "timestamp":
                df = df.drop(columns=[timestamp_column])
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', errors='coerce')
        rename = params.get("rename", {})
        if rename:
            df = df.rename(columns=rename)
        required = {"asset_id", "channel", "value"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        df = df.sort_values(by="timestamp") if "timestamp" in df.columns else df
        return df.reset_index(drop=True)

    async def subscribe(self, params: Mapping[str, Any]) -> AsyncIterator[dict[str, Any]]:
        df = self.load(params)
        interval = float(params.get("emit_interval_s", 0.0))
        for record in df.to_dict(orient="records"):
            yield record
            if interval > 0:
                await asyncio.sleep(interval)


__all__ = ["CSVAdapter"]
