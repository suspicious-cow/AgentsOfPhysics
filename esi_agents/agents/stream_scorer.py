"""Agent that performs streaming anomaly scoring."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from typing import Any

import pandas as pd

from .feature_engineer import FeatureEngineer
from .model_selector import SelectionResult


class StreamScorer:
    def __init__(self, feature_engineer: FeatureEngineer | None = None):
        self.feature_engineer = feature_engineer or FeatureEngineer()

    async def run(
        self,
        events: AsyncIterator[dict[str, Any]],
        config: dict[str, Any],
        selection: SelectionResult,
        emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        window_cfg = config.get("window", {})
        window_size = int(window_cfg.get("size", 256))
        stride = int(window_cfg.get("stride", window_size // 2))
        threshold = float(config.get("threshold", 0.9))
        buffers: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        emit = emit or (lambda msg: print(json.dumps(msg)))
        async for event in events:
            asset_id = str(event["asset_id"])
            channel = str(event["channel"])
            key = (asset_id, channel)
            buffers[key].append(event)
            if len(buffers[key]) < window_size:
                continue
            df = pd.DataFrame(buffers[key])
            features = self.feature_engineer.transform(df, config)
            latest = features.matrix.iloc[-1]
            numeric = latest.select_dtypes(include=[float, int])
            X = numeric.to_numpy(dtype=float).reshape(1, -1)
            score = float(selection.best_model.model.score_samples(X)[0])
            alert = score >= threshold
            emit(
                {
                    "asset_id": asset_id,
                    "channel": channel,
                    "timestamp": latest["window_end"].isoformat()
                    if hasattr(latest["window_end"], "isoformat")
                    else str(latest["window_end"]),
                    "anomaly_score": score,
                    "alert": alert,
                }
            )
            if len(buffers[key]) >= window_size + stride:
                buffers[key] = buffers[key][stride:]


__all__ = ["StreamScorer"]
