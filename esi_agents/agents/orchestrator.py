"""Orchestrator agent coordinating the workflow."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..utils.logging import get_logger
from ..utils.random import set_global_seed
from . import (
    BatchScorerAgent,
    CodeReviewerAgent,
    DataIngestorAgent,
    DriftMonitorAgent,
    EvaluatorAgent,
    FeatureEngineerAgent,
    LogicReviewerAgent,
    ModelSelectorAgent,
    ModelTrainerAgent,
    ReportContext,
    ReportWriterAgent,
)

LOGGER = get_logger(__name__)


@dataclass
class OrchestratorResult:
    output_dir: Path
    report_path: Path
    metrics: Dict[str, float]
    model: object
    feature_columns: List[str]
    config: Dict


class OrchestratorAgent:
    def __init__(self) -> None:
        self.data_ingestor = DataIngestorAgent()
        self.feature_engineer = FeatureEngineerAgent()
        self.model_trainer = ModelTrainerAgent()
        self.evaluator = EvaluatorAgent()
        self.model_selector = ModelSelectorAgent(self.evaluator)
        self.drift_monitor = DriftMonitorAgent()
        self.batch_scorer = BatchScorerAgent()
        self.report_writer = ReportWriterAgent()
        self.logic_reviewer = LogicReviewerAgent()
        self.code_reviewer = CodeReviewerAgent()

    def run_batch(self, config_path: Path, input_path: Optional[Path], output_dir: Path) -> OrchestratorResult:
        config = yaml.safe_load(config_path.read_text())
        output_dir.mkdir(parents=True, exist_ok=True)
        set_global_seed()

        LOGGER.info("Starting batch pipeline")
        ingest_result = self.data_ingestor.run(config, input_override=str(input_path) if input_path else None)
        feature_result = self.feature_engineer.run(ingest_result.dataframe, ingest_result.schema, config)
        trainer_result = self.model_trainer.run(feature_result.features, feature_result.feature_columns, config)
        selection_result = self.model_selector.run(
            trainer_result,
            feature_result.features,
            output_dir,
            config,
        )

        test_X, test_y = trainer_result.splits["test"]
        evaluation = self.evaluator.run(
            selection_result.best_model,
            test_X,
            test_y,
            feature_result.features.loc[test_X.index],
            output_dir,
            prefix=f"{selection_result.best_model_name}_test",
        )

        drift_report = self.drift_monitor.run(
            trainer_result.splits["train"][0],
            feature_result.features[feature_result.feature_columns],
        )

        score_result = self.batch_scorer.run(
            selection_result.best_model,
            feature_result.features,
            feature_result.feature_columns,
            output_dir,
        )

        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(evaluation.metrics, indent=2))

        report_context = ReportContext(
            config=config,
            dq_report=ingest_result.dq_report,
            metrics=evaluation.metrics,
            drift=drift_report.metrics,
            band_scan=evaluation.band_scan,
        )
        report_path = self.report_writer.run(report_context, output_dir)

        self.logic_reviewer.run(report_path, evaluation.metrics)
        self.code_reviewer.run(
            [
                report_path,
                score_result.path,
                metrics_path,
                output_dir / f"{selection_result.best_model_name}_test_roc.png",
                output_dir / f"{selection_result.best_model_name}_test_pr.png",
                output_dir / f"{selection_result.best_model_name}_test_band_scan.png",
            ]
        )

        return OrchestratorResult(
            output_dir=output_dir,
            report_path=report_path,
            metrics=evaluation.metrics,
            model=selection_result.best_model,
            feature_columns=feature_result.feature_columns,
            config=config,
        )


__all__ = ["OrchestratorAgent", "OrchestratorResult"]
