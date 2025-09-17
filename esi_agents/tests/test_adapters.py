"""Adapter tests."""
from __future__ import annotations

import pandas as pd

import pytest

from ..adapters import CSVAdapter, ParquetAdapter


def test_csv_adapter_roundtrip(tmp_path):
    df = pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=5, freq="s"), "value": range(5)})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    adapter = CSVAdapter()
    loaded = adapter.load({"path": path, "read_csv_kwargs": {"parse_dates": ["timestamp"]}})
    assert len(loaded) == len(df)
    assert pd.api.types.is_datetime64_any_dtype(loaded["timestamp"])


def test_parquet_adapter_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=5, freq="s"), "value": range(5)})
    path = tmp_path / "data.parquet"
    df.to_parquet(path)

    adapter = ParquetAdapter()
    loaded = adapter.load({"path": path})
    assert len(loaded) == len(df)
