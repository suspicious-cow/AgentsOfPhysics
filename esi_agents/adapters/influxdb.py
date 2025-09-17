"""InfluxDB adapter stub with optional dependency."""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

import pandas as pd

from .base import AdapterNotAvailable, BaseAdapter

try:  # pragma: no cover - import guarded for optional dependency
    from influxdb_client import InfluxDBClient  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    InfluxDBClient = None  # type: ignore[misc]


class InfluxDBAdapter(BaseAdapter):
    """Adapter for InfluxDB using lightweight callouts.

    The implementation relies on callables passed in via ``params`` to keep
    the dependency surface small while still being testable. When the
    optional :mod:`influxdb_client` dependency is not installed a clear
    :class:`AdapterNotAvailable` error is raised.
    """

    def _ensure_available(self) -> None:
        if InfluxDBClient is None:
            raise AdapterNotAvailable(
                "influxdb-client is not installed; install optional extra 'influxdb'"
            )

    def load(self, params: Mapping[str, Any]) -> pd.DataFrame:
        self._ensure_available()
        query_fn: Callable[[Mapping[str, Any]], pd.DataFrame] | None = params.get("query_fn")
        if query_fn is None:
            raise ValueError("InfluxDBAdapter requires a 'query_fn' callable in params")
        df = query_fn(params)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("query_fn must return a pandas.DataFrame")
        return df

    async def subscribe(self, params: Mapping[str, Any]) -> AsyncIterator[dict[str, Any]]:
        self._ensure_available()
        generator_fn: Callable[[Mapping[str, Any]], AsyncIterator[dict[str, Any]]] | None = params.get(
            "generator_fn"
        )
        if generator_fn is None:
            raise ValueError("InfluxDBAdapter requires a 'generator_fn' async callable")
        async for item in generator_fn(params):
            yield item


__all__ = ["InfluxDBAdapter"]
