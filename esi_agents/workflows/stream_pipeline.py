"""Streaming pipeline orchestration."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import yaml

from ..adapters import CSVAdapter, MQTTAdapter, OPCUAAdapter, ParquetAdapter
from ..agents.orchestrator import OrchestratorAgent
from ..agents.stream_scorer import StreamScorerAgent


_STREAM_ADAPTERS = {
    "csv": CSVAdapter,
    "parquet": ParquetAdapter,
    "mqtt": MQTTAdapter,
    "opcua": OPCUAAdapter,
}


def run_stream_pipeline(config_path: str, output_dir: Optional[str] = None) -> None:
    config = yaml.safe_load(Path(config_path).read_text())
    orchestrator = OrchestratorAgent()
    train_output = Path(output_dir or config.get("artifacts", {}).get("train_dir", "artifacts/stream_train"))
    result = orchestrator.run_batch(Path(config_path), None, train_output)

    stream_cfg = config.get("stream", {})
    source_cfg = stream_cfg.get("source", {})
    adapter_type = source_cfg.get("type", "csv").lower()
    adapter_cls = _STREAM_ADAPTERS.get(adapter_type)
    if adapter_cls is None:
        raise ValueError(f"Unsupported stream adapter: {adapter_type}")
    adapter = adapter_cls()

    async def _run():
        stream = adapter.subscribe(source_cfg.get("params", {}))
        scorer = StreamScorerAgent()
        threshold = float(result.metrics.get("sic_threshold", 0.9))
        await scorer.run(
            result.model,
            stream,
            config["features"],
            result.feature_columns,
            threshold,
        )

    asyncio.run(_run())


__all__ = ["run_stream_pipeline"]
