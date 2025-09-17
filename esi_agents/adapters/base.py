"""Base interfaces for data adapters."""
from __future__ import annotations

import abc
from typing import Any, AsyncIterator, Dict

import pandas as pd


class BaseBatchAdapter(abc.ABC):
    """Interface for adapters that load batch data."""

    @abc.abstractmethod
    def load(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Load data into a dataframe."""


class BaseStreamAdapter(abc.ABC):
    """Interface for adapters that expose streaming data."""

    @abc.abstractmethod
    def subscribe(self, params: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Subscribe to an asynchronous stream of records."""


class SupportsBatchAndStream(BaseBatchAdapter, BaseStreamAdapter):
    """Helper class for adapters supporting both interfaces."""

    pass


__all__ = ["BaseBatchAdapter", "BaseStreamAdapter", "SupportsBatchAndStream"]
