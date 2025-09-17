"""Adapter interfaces for batch and streaming ingestion."""
from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Mapping
from typing import Any

import pandas as pd


class BaseAdapter(abc.ABC):
    """Abstract base class for data adapters.

    Concrete implementations must provide batch loading via :meth:`load`
    and streaming consumption via :meth:`subscribe`.
    """

    @abc.abstractmethod
    def load(self, params: Mapping[str, Any]) -> pd.DataFrame:
        """Load a batch of observations.

        Parameters
        ----------
        params:
            Adapter specific configuration. Typically includes a path or
            connection information as well as schema hints.
        """

    @abc.abstractmethod
    def subscribe(self, params: Mapping[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Return an asynchronous iterator yielding streaming observations."""


class AdapterNotAvailable(RuntimeError):
    """Raised when an optional adapter dependency is missing."""


__all__ = ["BaseAdapter", "AdapterNotAvailable"]
