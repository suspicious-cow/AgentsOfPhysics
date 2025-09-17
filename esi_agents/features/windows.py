"""Utilities for windowing ESI time-series data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Window:
    """Container representing a sliding window over a signal."""

    asset_id: str
    channel: str
    start: pd.Timestamp
    end: pd.Timestamp
    values: np.ndarray
    sampling_rate_hz: float | None
    extras: dict[str, Any] = field(default_factory=dict)


def _infer_sampling_rate(timestamps: pd.Series) -> float | None:
    timestamps = pd.to_datetime(timestamps).sort_values()
    if len(timestamps) < 2:
        return None
    deltas = timestamps.diff().dropna().dt.total_seconds()
    if deltas.empty:
        return None
    return float(1.0 / deltas.mean()) if deltas.mean() else None


def generate_windows(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    time_col: str = "timestamp",
    value_col: str = "value",
) -> list[Window]:
    """Generate sliding windows for each asset/channel pair.

    Parameters
    ----------
    df:
        Input data frame with at least ``asset_id``, ``channel`` and value
        columns.
    window_size:
        Number of samples per window.
    stride:
        Step size between consecutive windows.
    """

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if not {"asset_id", "channel", value_col}.issubset(df.columns):
        raise ValueError("DataFrame missing required columns")

    windows: list[Window] = []
    group_cols = ["asset_id", "channel"]
    for (asset_id, channel), group in df.groupby(group_cols):
        group = group.sort_values(time_col)
        values = group[value_col].to_numpy()
        timestamps = (
            pd.to_datetime(group[time_col]).to_numpy() if time_col in group.columns else None
        )
        rpm = group["rpm"].to_numpy() if "rpm" in group.columns else None
        sampling_rate = _infer_sampling_rate(group[time_col]) if time_col in group else None
        for start in range(0, len(values) - window_size + 1, stride):
            end = start + window_size
            window_values = values[start:end]
            if timestamps is not None:
                start_ts = pd.Timestamp(timestamps[start])
                end_ts = pd.Timestamp(timestamps[end - 1])
            else:
                start_ts = pd.Timestamp.now()
                end_ts = start_ts
            extras: dict[str, Any] = {}
            if rpm is not None:
                extras["rpm"] = rpm[start:end]
            windows.append(
                Window(
                    asset_id=str(asset_id),
                    channel=str(channel),
                    start=start_ts,
                    end=end_ts,
                    values=window_values,
                    sampling_rate_hz=sampling_rate,
                    extras=extras,
                )
            )
    return windows


__all__ = ["Window", "generate_windows"]
