"""TimescaleDB adapter stub."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .base import BaseBatchAdapter
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class TimescaleAdapter(BaseBatchAdapter):
    """Adapter for TimescaleDB using SQLAlchemy."""

    def __init__(self) -> None:
        try:
            import sqlalchemy  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional
            raise ImportError(
                "TimescaleDB support requires the optional 'sqlalchemy' dependency"
            ) from exc
        self._sqlalchemy = sqlalchemy

    def load(self, params: Dict[str, Any]) -> pd.DataFrame:
        engine = self._sqlalchemy.create_engine(params["connection_string"])
        query = params["query"]
        LOGGER.info("Executing TimescaleDB query: %s", query)
        return pd.read_sql_query(query, engine)


__all__ = ["TimescaleAdapter"]
