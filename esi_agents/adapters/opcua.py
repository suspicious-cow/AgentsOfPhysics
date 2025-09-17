"""OPC-UA adapter stub with optional dependency."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

import pandas as pd

from .base import AdapterNotAvailable, BaseAdapter

try:  # pragma: no cover - optional dependency
    from opcua import Client  # type: ignore
except Exception:  # pragma: no cover
    Client = None  # type: ignore[misc]


class OPCUAAdapter(BaseAdapter):
    """Minimal adapter around :mod:`python-opcua` with injectable hooks."""

    def _ensure_available(self) -> None:
        if Client is None:
            raise AdapterNotAvailable(
                "python-opcua is not installed; install optional extra 'opcua'"
            )

    def load(self, params: Mapping[str, Any]) -> pd.DataFrame:
        self._ensure_available()
        read_fn: Callable[[Mapping[str, Any]], pd.DataFrame] | None = params.get("read_fn")
        if read_fn is None:
            raise ValueError("OPCUAAdapter requires a 'read_fn' callable")
        df = read_fn(params)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("read_fn must return a pandas.DataFrame")
        return df

    async def subscribe(self, params: Mapping[str, Any]) -> AsyncIterator[dict[str, Any]]:
        self._ensure_available()
        generator_fn: Callable[[Mapping[str, Any]], AsyncIterator[dict[str, Any]]] | None = params.get(
            "generator_fn"
        )
        if generator_fn is None:
            raise ValueError("OPCUAAdapter requires a 'generator_fn'")
        async for item in generator_fn(params):
            yield item


__all__ = ["OPCUAAdapter"]
