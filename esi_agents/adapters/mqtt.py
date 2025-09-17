"""MQTT streaming adapter."""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict

from .base import BaseStreamAdapter
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class MQTTAdapter(BaseStreamAdapter):
    """Adapter for MQTT streams using paho-mqtt."""

    def __init__(self) -> None:
        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional
            raise ImportError(
                "MQTT streaming requires the optional 'paho-mqtt' dependency"
            ) from exc
        self._mqtt = mqtt

    async def subscribe(self, params: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:  # pragma: no cover - network
        topic = params["topic"]
        host = params.get("host", "localhost")
        port = int(params.get("port", 1883))
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        def on_message(client, userdata, message):
            payload = message.payload.decode("utf-8")
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = {"payload": payload}
            queue.put_nowait(data)

        client = self._mqtt.Client()
        client.on_message = on_message
        client.connect(host, port, keepalive=60)
        client.subscribe(topic)
        client.loop_start()

        try:
            while True:
                yield await queue.get()
        finally:
            client.loop_stop()
            client.disconnect()


__all__ = ["MQTTAdapter"]
