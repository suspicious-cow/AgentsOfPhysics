"""Integration tests for orchestrator."""
from __future__ import annotations

from pathlib import Path

from ..agents.orchestrator import OrchestratorAgent


def test_orchestrator_batch_pipeline(turbine_config_path, output_dir):
    orchestrator = OrchestratorAgent()
    result = orchestrator.run_batch(turbine_config_path, None, output_dir)
    assert result.report_path.exists()
    assert result.metrics["roc_auc"] > 0.8
    assert result.metrics["pr_auc"] > 0.8
