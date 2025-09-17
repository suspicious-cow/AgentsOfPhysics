from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from esi_agents.adapters import (
    AdapterNotAvailable,
    InfluxDBAdapter,
    MQTTAdapter,
    OPCUAAdapter,
    TimescaleAdapter,
)


@pytest.mark.parametrize("adapter_cls", [InfluxDBAdapter, TimescaleAdapter])
def test_db_adapters_raise_when_missing(adapter_cls):
    adapter = adapter_cls()
    with pytest.raises(AdapterNotAvailable):
        adapter.load({})


@pytest.mark.parametrize("adapter_cls", [MQTTAdapter, OPCUAAdapter])
def test_transport_adapters_raise_on_subscribe(adapter_cls):
    adapter = adapter_cls()

    async def consume():
        stream = adapter.subscribe({})
        return [item async for item in stream]

    with pytest.raises(AdapterNotAvailable):
        asyncio.run(consume())
