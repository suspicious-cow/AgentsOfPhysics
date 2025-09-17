"""Batch pipeline entry point."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..agents.orchestrator import OrchestratorAgent


def run_batch_pipeline(config_path: str, input_path: Optional[str], output_dir: str):
    orchestrator = OrchestratorAgent()
    return orchestrator.run_batch(Path(config_path), Path(input_path) if input_path else None, Path(output_dir))


__all__ = ["run_batch_pipeline"]
