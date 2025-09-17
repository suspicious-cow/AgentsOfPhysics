"""TimescaleDB adapter using SQL callouts."""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

import pandas as pd

from .base import AdapterNotAvailable, BaseAdapter

try:  # pragma: no cover - optional dependency import guard
    import psycopg2  # type: ignore
except Exception:  # pragma: no cover
    psycopg2 = None  # type: ignore[misc]


class TimescaleAdapter(BaseAdapter):
    """Adapter that delegates SQL execution to injected callables."""

    def _ensure_available(self) -> None:
        if psycopg2 is None:
            raise AdapterNotAvailable(
                "psycopg2 is not installed; install optional extra 'timescale'"
            )

    def load(self, params: Mapping[str, Any]) -> pd.DataFrame:
        self._ensure_available()
        query_fn: Callable[[Mapping[str, Any]], pd.DataFrame] | None = params.get("query_fn")
        if query_fn is None:
            raise ValueError("TimescaleAdapter requires a 'query_fn' callable")
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
            raise ValueError("TimescaleAdapter requires a 'generator_fn'")
        async for item in generator_fn(params):
            yield item


__all__ = ["TimescaleAdapter"]
