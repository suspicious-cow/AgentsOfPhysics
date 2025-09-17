"""Parquet adapter implementation."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Dict

import pandas as pd

from .base import SupportsBatchAndStream
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class ParquetAdapter(SupportsBatchAndStream):
    """Adapter for Parquet files supporting batch and simulated streaming."""

    def load(self, params: Dict[str, Any]) -> pd.DataFrame:
        path = Path(params["path"])
        kwargs = params.get("read_parquet_kwargs", {})
        LOGGER.info("Loading Parquet data from %s", path)
        df = pd.read_parquet(path, **kwargs)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    async def subscribe(self, params: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        df = self.load(params)
        delay = float(params.get("delay", 0.0))
        for row in df.to_dict(orient="records"):
            if delay:
                await asyncio.sleep(delay)
            yield row


__all__ = ["ParquetAdapter"]
