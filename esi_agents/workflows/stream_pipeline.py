"""Streaming pipeline leveraging the agent architecture."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

import yaml

from ..adapters import CSVAdapter, MQTTAdapter, OPCUAAdapter, ParquetAdapter
from ..agents import (
    DataIngestor,
    FeatureEngineer,
    ModelSelector,
    ModelTrainer,
    StreamScorer,
)


async def _stream_from_adapter(adapter_name: str, params: dict[str, Any]):
    adapter_name = adapter_name.lower()
    if adapter_name == "csv":
        adapter = CSVAdapter()
    elif adapter_name == "parquet":
        adapter = ParquetAdapter()
    elif adapter_name == "mqtt":
        adapter = MQTTAdapter()
    elif adapter_name == "opcua":
        adapter = OPCUAAdapter()
    else:
        raise ValueError(f"Unsupported streaming adapter {adapter_name}")
    async for item in adapter.subscribe(params):
        yield item


async def run_stream(config_path: str | Path, emit: Callable[[dict[str, Any]], None] | None = None):
    config = yaml.safe_load(Path(config_path).read_text())
    training_cfg = config.get("training", config)
    ingestor = DataIngestor()
    ingest_result = ingestor.ingest(training_cfg)
    feature_engineer = FeatureEngineer()
    feature_result = feature_engineer.transform(ingest_result.frame, training_cfg)
    trainer = ModelTrainer()
    trained = trainer.train(feature_result.matrix, training_cfg)
    selector = ModelSelector()
    selection = selector.select(trained, labels=None)
    stream_cfg = config.get("stream", training_cfg)
    adapter_name = stream_cfg.get("adapter", training_cfg.get("adapter", "csv"))
    params = stream_cfg.get("params", {})
    event_iter = _stream_from_adapter(adapter_name, params)
    scorer = StreamScorer(feature_engineer)
    await scorer.run(event_iter, training_cfg, selection, emit)


__all__ = ["run_stream"]
