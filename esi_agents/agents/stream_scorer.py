"""Streaming scorer agent."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, AsyncIterator, Deque, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..features import (
    Window,
    compute_envelope_features,
    compute_frequency_features,
    compute_order_features,
    compute_time_domain_features,
)


@dataclass
class StreamAlert:
    payload: Dict[str, Any]


class StreamScorerAgent:
    def __init__(self) -> None:
        self.buffers: Dict[Tuple[str, str], Deque[Dict[str, Any]]] = {}

    async def run(
        self,
        model,
        stream: AsyncIterator[Dict[str, Any]],
        feature_cfg: Dict[str, Any],
        feature_columns: List[str],
        threshold: float,
        callback=None,
    ) -> None:
        window_size = int(feature_cfg.get("window_size", 256))
        stride = int(feature_cfg.get("stride", max(1, window_size // 2)))
        counter: Dict[Tuple[str, str], int] = defaultdict(int)

        async for record in stream:
            asset_id = str(record.get("asset_id", "asset_0"))
            channel = str(record.get("channel", "channel_0"))
            key = (asset_id, channel)
            buffer = self.buffers.get(key)
            if buffer is None:
                buffer = deque(maxlen=window_size)
                self.buffers[key] = buffer
            buffer.append(record)
            counter[key] += 1
            if len(buffer) == window_size and counter[key] % stride == 0:
                window = self._buffer_to_window(asset_id, channel, list(buffer))
                features = self._compute_features(window)
                X = pd.DataFrame([features]).reindex(columns=feature_columns, fill_value=0.0)
                score = float(model.score_samples(X)[0])
                alert = {
                    "asset_id": asset_id,
                    "channel": channel,
                    "window_start": window.start.isoformat(),
                    "window_end": window.end.isoformat(),
                    "score": score,
                    "features": features,
                }
                if score >= threshold:
                    message = json.dumps(alert)
                    if callback:
                        callback(StreamAlert(payload=alert))
                    else:
                        print(message, flush=True)

    def _buffer_to_window(self, asset_id: str, channel: str, buffer: list[Dict[str, Any]]) -> Window:
        timestamps = pd.to_datetime([record["timestamp"] for record in buffer])
        values = np.array([record["value"] for record in buffer], dtype=float)
        rpm = float(np.nanmean([record.get("rpm", np.nan) for record in buffer]))
        return Window(
            asset_id=asset_id,
            channel=channel,
            start=timestamps.min().to_pydatetime(),
            end=timestamps.max().to_pydatetime(),
            values=values,
            timestamps=timestamps.to_numpy(),
            rpm=rpm,
        )

    def _compute_features(self, window: Window) -> Dict[str, float]:
        features: Dict[str, float] = {}
        features.update(compute_time_domain_features(window))
        features.update(compute_frequency_features(window))
        features.update(compute_envelope_features(window))
        features.update(compute_order_features(window))
        return features


__all__ = ["StreamScorerAgent", "StreamAlert"]
