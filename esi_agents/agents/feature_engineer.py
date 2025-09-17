"""Agent responsible for feature extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..features import (
    Window,
    compute_envelope_features,
    compute_frequency_features,
    compute_order_features,
    compute_sideband_features,
    compute_time_features,
    dominant_frequencies,
    generate_windows,
)


@dataclass
class FeatureResult:
    matrix: pd.DataFrame
    windows: list[Window]


class FeatureEngineer:
    def transform(self, frame: pd.DataFrame, config: dict[str, Any]) -> FeatureResult:
        window_cfg = config.get("window", {})
        window_size = int(window_cfg.get("size", 256))
        stride = int(window_cfg.get("stride", window_size // 2))
        windows = generate_windows(frame, window_size=window_size, stride=stride)
        records: list[dict[str, Any]] = []
        include = config.get("features", {"time": True, "freq": True, "envelope": True, "orders": True})
        for window in windows:
            feats: dict[str, Any] = {
                "asset_id": window.asset_id,
                "channel": window.channel,
                "window_start": window.start,
                "window_end": window.end,
            }
            if include.get("time", True):
                feats.update(compute_time_features(window))
            if include.get("freq", True):
                feats.update(compute_frequency_features(window))
                feats.update(dominant_frequencies(window))
            if include.get("envelope", True):
                feats.update(compute_envelope_features(window))
            if include.get("orders", True):
                feats.update(compute_order_features(window))
                feats.update(compute_sideband_features(window))
            records.append(feats)
        feature_frame = pd.DataFrame(records)
        feature_frame = feature_frame.fillna(0.0)
        return FeatureResult(matrix=feature_frame, windows=windows)


__all__ = ["FeatureEngineer", "FeatureResult"]
