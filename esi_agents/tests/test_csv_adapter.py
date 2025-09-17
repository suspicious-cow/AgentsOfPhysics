from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd

from esi_agents.adapters import CSVAdapter


def test_csv_adapter_load(tmp_path):
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="s"),
            "asset_id": ["a"] * 5,
            "channel": ["c"] * 5,
            "value": range(5),
        }
    )
    path = tmp_path / "sample.csv"
    data.to_csv(path, index=False)
    adapter = CSVAdapter()
    frame = adapter.load({"path": path})
    assert len(frame) == 5
    assert frame["value"].iloc[-1] == 4


def test_csv_adapter_subscribe(tmp_path):
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="s"),
            "asset_id": ["a"] * 3,
            "channel": ["c"] * 3,
            "value": range(3),
        }
    )
    path = tmp_path / "sample.csv"
    data.to_csv(path, index=False)
    adapter = CSVAdapter()

    async def consume():
        records = []
        async for item in adapter.subscribe({"path": path}):
            records.append(item)
        return records

    records = asyncio.run(consume())
    assert len(records) == 3
    assert records[0]["value"] == 0
