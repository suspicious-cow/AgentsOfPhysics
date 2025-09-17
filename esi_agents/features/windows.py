"""Windowing utilities for ESI features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Window:
    """Represents a sliding window slice for a single channel."""

    asset_id: str
    channel: str
    start: pd.Timestamp
    end: pd.Timestamp
    values: np.ndarray
    timestamps: np.ndarray
    rpm: float | None = None


def iter_windows(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    asset_col: str = "asset_id",
    channel_col: str = "channel",
    rpm_col: str | None = "rpm",
) -> Iterator[Window]:
    """Yield sliding windows for each asset/channel pair."""

    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive integers")

    df_sorted = df.sort_values(timestamp_col)
    grouped = df_sorted.groupby([asset_col, channel_col])

    for (asset_id, channel), group in grouped:
        values = group[value_col].to_numpy(dtype=float)
        timestamps = group[timestamp_col].to_numpy()
        rpm_values = group[rpm_col].to_numpy(dtype=float) if rpm_col and rpm_col in group else None

        for start_idx in range(0, len(values) - window_size + 1, stride):
            end_idx = start_idx + window_size
            window_values = values[start_idx:end_idx]
            window_timestamps = timestamps[start_idx:end_idx]
            rpm_value = float(np.nanmean(rpm_values[start_idx:end_idx])) if rpm_values is not None else None
            yield Window(
                asset_id=str(asset_id),
                channel=str(channel),
                start=pd.to_datetime(window_timestamps[0]),
                end=pd.to_datetime(window_timestamps[-1]),
                values=window_values,
                timestamps=window_timestamps,
                rpm=rpm_value,
            )


def fuse_windows(windows: Iterable[Window]) -> Dict[Tuple[str, str], List[Window]]:
    """Group windows by asset and their start time for multivariate features."""

    fused: Dict[Tuple[str, str], List[Window]] = {}
    for window in windows:
        key = (window.asset_id, str(window.start))
        fused.setdefault(key, []).append(window)
    return fused


__all__ = ["Window", "iter_windows", "fuse_windows"]
