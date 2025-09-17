"""InfluxDB adapter stub."""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict

import pandas as pd

from .base import BaseBatchAdapter, BaseStreamAdapter
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class InfluxDBAdapter(BaseBatchAdapter, BaseStreamAdapter):
    """Adapter for InfluxDB sources using optional dependencies."""

    def __init__(self) -> None:
        try:
            from influxdb_client import InfluxDBClient  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional
            raise ImportError(
                "InfluxDB support requires the optional 'influxdb-client' dependency"
            ) from exc
        self._InfluxDBClient = InfluxDBClient

    def load(self, params: Dict[str, Any]) -> pd.DataFrame:
        client = self._InfluxDBClient(**params["client_args"])
        query = params["query"]
        LOGGER.info("Querying InfluxDB: %s", query)
        tables = client.query_api().query_data_frame(query)
        return pd.concat(tables) if isinstance(tables, list) else tables

    async def subscribe(self, params: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:  # pragma: no cover - requires service
        client = self._InfluxDBClient(**params["client_args"])
        query = params["query"]
        for table in client.query_api().query_stream(query):
            yield table


__all__ = ["InfluxDBAdapter"]
