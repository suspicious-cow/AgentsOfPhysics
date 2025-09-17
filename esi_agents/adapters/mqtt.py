"""MQTT adapter using optional :mod:`paho-mqtt`."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

import pandas as pd

from .base import AdapterNotAvailable, BaseAdapter

try:  # pragma: no cover - optional dependency
    import paho.mqtt.client as mqtt  # type: ignore
except Exception:  # pragma: no cover
    mqtt = None  # type: ignore[misc]


class MQTTAdapter(BaseAdapter):
    """Adapter that can attach to an MQTT broker.

    The adapter exposes hooks that allow dependency-free unit tests by
    injecting lightweight message producers.
    """

    def _ensure_available(self) -> None:
        if mqtt is None:
            raise AdapterNotAvailable(
                "paho-mqtt is not installed; install optional extra 'mqtt'"
            )

    def load(self, params: Mapping[str, Any]) -> pd.DataFrame:
        """Convert buffered messages into a DataFrame.

        Parameters
        ----------
        params:
            Should contain a ``messages`` iterable of dictionaries.
        """

        messages = params.get("messages")
        if messages is None:
            raise ValueError("MQTTAdapter.load requires a 'messages' iterable of dicts")
        df = pd.DataFrame(list(messages))
        if not {"asset_id", "channel", "value"}.issubset(df.columns):
            raise ValueError("messages must provide asset_id, channel, and value fields")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
        return df.reset_index(drop=True)

    async def subscribe(self, params: Mapping[str, Any]) -> AsyncIterator[dict[str, Any]]:
        self._ensure_available()
        topic = params.get("topic")
        if not topic:
            raise ValueError("MQTTAdapter.subscribe requires a 'topic'")
        host = params.get("host", "localhost")
        port = int(params.get("port", 1883))
        client_factory: Callable[[], Any] | None = params.get("client_factory")
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def _default_factory():  # pragma: no cover - network setup not tested
            return mqtt.Client()

        client = client_factory() if client_factory else _default_factory()

        def on_message(_client, _userdata, message):  # pragma: no cover - thin wrapper
            payload = message.payload.decode()
            record: dict[str, Any]
            if params.get("decoder"):
                record = params["decoder"](payload)
            else:
                record = {"raw_payload": payload}
            loop.call_soon_threadsafe(queue.put_nowait, record)

        client.on_message = on_message
        client.connect(host, port, keepalive=params.get("keepalive", 60))
        client.subscribe(topic)
        client.loop_start()

        try:
            while True:
                item = await queue.get()
                yield item
        finally:  # pragma: no cover - network cleanup not exercised in unit tests
            client.loop_stop()
            client.disconnect()


__all__ = ["MQTTAdapter"]
