"""Pytest fixtures for ESI agents."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from ..features.windows import Window


@pytest.fixture
def synthetic_window() -> Window:
    timestamps = pd.date_range("2023-01-01", periods=256, freq="10ms")
    values = np.sin(2 * np.pi * 10 * np.arange(256) / 100)
    return Window(
        asset_id="asset_1",
        channel="sensor_1",
        start=timestamps[0].to_pydatetime(),
        end=timestamps[-1].to_pydatetime(),
        values=values,
        timestamps=timestamps.to_numpy(),
        rpm=1800,
    )


@pytest.fixture
def synthetic_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2023-01-01", periods=512, freq="10ms")
    values = np.sin(2 * np.pi * 5 * np.arange(512) / 100) + 0.1 * rng.standard_normal(512)
    df = pd.DataFrame(
        {
            "timestamp": list(timestamps) * 2,
            "asset_id": ["asset"] * len(timestamps) * 2,
            "channel": ["ch1"] * len(timestamps) + ["ch2"] * len(timestamps),
            "value": list(values) + list(values * 0.5),
            "label": [0] * (len(timestamps) * 2),
            "rpm": [1800] * (len(timestamps) * 2),
        }
    )
    return df


@pytest.fixture
def turbine_config_path() -> Path:
    return Path("esi_agents/configs/turbine_vibration.yaml")


@pytest.fixture
def turbine_data_path() -> Path:
    return Path("data/turbine.csv")


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"
