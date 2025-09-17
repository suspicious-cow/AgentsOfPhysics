"""Batch pipeline entry point."""
from __future__ import annotations

from pathlib import Path

from ..agents import Orchestrator


def run_batch(config: str | Path, input_path: str | None, output_dir: str | Path, labels: str | None = None):
    orchestrator = Orchestrator()
    return orchestrator.run(config, input_path, output_dir, labels)


__all__ = ["run_batch"]
