"""OPC-UA adapter stub."""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterable

from .base import BaseBatchAdapter, BaseStreamAdapter
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class OPCUAAdapter(BaseBatchAdapter, BaseStreamAdapter):
    """Adapter for OPC-UA servers."""

    def __init__(self) -> None:
        try:
            from opcua import Client  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional
            raise ImportError(
                "OPC-UA streaming requires the optional 'python-opcua' dependency"
            ) from exc
        self._Client = Client

    def load(self, params: Dict[str, Any]) -> Iterable[Dict[str, Any]]:  # pragma: no cover - network
        return list(self._read_nodes(params))

    async def subscribe(self, params: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:  # pragma: no cover - network
        for item in self._read_nodes(params):
            yield item

    def _read_nodes(self, params: Dict[str, Any]):
        client = self._Client(params["endpoint"])
        node_ids = params["nodes"]
        LOGGER.info("Reading OPC-UA nodes: %s", node_ids)
        client.connect()
        try:
            for node_id in node_ids:
                node = client.get_node(node_id)
                yield {"node": node_id, "value": node.get_value()}
        finally:
            client.disconnect()


__all__ = ["OPCUAAdapter"]
