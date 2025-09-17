"""SQLite adapter useful for demos."""
from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Mapping
from typing import Any

import pandas as pd

from .base import BaseAdapter


class SQLiteAdapter(BaseAdapter):
    """Lightweight adapter backed by :mod:`sqlite3`."""

    def load(self, params: Mapping[str, Any]) -> pd.DataFrame:
        database = params.get("database", ":memory:")
        query = params.get("query")
        if not query:
            raise ValueError("SQLiteAdapter requires a SQL 'query'")
        with sqlite3.connect(database) as conn:
            df = pd.read_sql_query(query, conn)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    async def subscribe(self, params: Mapping[str, Any]) -> AsyncIterator[dict[str, Any]]:
        df = self.load(params)
        for record in df.to_dict(orient="records"):
            yield record


__all__ = ["SQLiteAdapter"]
