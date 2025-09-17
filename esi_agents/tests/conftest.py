from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_signal() -> pd.DataFrame:
    timestamps = [datetime(2024, 1, 1) + timedelta(milliseconds=i) for i in range(0, 1000, 10)]
    values = np.sin(np.linspace(0, 10 * np.pi, len(timestamps)))
    rpm = 1800 + 50 * np.sin(np.linspace(0, 2 * np.pi, len(timestamps)))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": "asset_1",
            "channel": "accel",
            "value": values,
            "rpm": rpm,
        }
    )
