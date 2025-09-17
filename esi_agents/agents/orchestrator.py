"""Top-level orchestrator composing the agent workflow."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .batch_scorer import BatchScorer
from .code_reviewer import CodeReviewer
from .data_ingestor import DataIngestor
from .drift_monitor import DriftMonitor
from .evaluator import Evaluator
from .feature_engineer import FeatureEngineer
from .logic_reviewer import LogicReviewer
from .model_selector import ModelSelector
from .model_trainer import ModelTrainer
from .report_writer import ReportWriter


@dataclass
class OrchestratorResult:
    metrics: dict[str, Any]
    report_path: Path
    scores_path: Path


class Orchestrator:
    def __init__(self) -> None:
        self.ingestor = DataIngestor()
        self.features = FeatureEngineer()
        self.trainer = ModelTrainer()
        self.selector = ModelSelector()
        self.evaluator = Evaluator()
        self.reporter = ReportWriter()
        self.logic_reviewer = LogicReviewer()
        self.code_reviewer = CodeReviewer()
        self.batch_scorer = BatchScorer()
        self.drift_monitor = DriftMonitor()

    def _window_labels(self, feature_result, labels_df: pd.DataFrame) -> np.ndarray:
        if "timestamp" not in labels_df.columns:
            return labels_df[labels_df.columns[-1]].to_numpy()
        labels_df["timestamp"] = pd.to_datetime(labels_df["timestamp"], format="ISO8601", errors="coerce")
        labels_df = labels_df.sort_values("timestamp")
        label_series = labels_df.set_index("timestamp")[labels_df.columns[-1]]
        window_labels = []
        for window in feature_result.windows:
            start = window.start
            end = window.end
            window_data = label_series.loc[start:end]
            if window_data.empty:
                window_labels.append(0)
            else:
                window_labels.append(int(window_data.max()))
        return np.asarray(window_labels, dtype=int)

    def run(
        self,
        config_path: str | Path,
        input_path: str | None,
        output_dir: str | Path,
        labels_path: str | None = None,
    ) -> OrchestratorResult:
        config = yaml.safe_load(Path(config_path).read_text())
        if input_path:
            config.setdefault("params", {})["path"] = input_path
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        ingest_result = self.ingestor.ingest(config)
        feature_result = self.features.transform(ingest_result.frame, config)
        labels = None
        if labels_path and Path(labels_path).exists():
            labels_df = pd.read_csv(labels_path)
            labels = self._window_labels(feature_result, labels_df)
        trained = self.trainer.train(feature_result.matrix, config, labels)
        selection = self.selector.select(trained, labels)
        evaluation = self.evaluator.evaluate(selection, feature_result, labels, output / "evaluation")
        drift = None
        if config.get("reference_features"):
            ref_path = Path(config["reference_features"])
            if ref_path.exists():
                reference = pd.read_parquet(ref_path)
                numeric_cols = feature_result.matrix.select_dtypes(include=[float, int]).columns.tolist()
                drift = self.drift_monitor.assess(reference, feature_result.matrix, numeric_cols)
        scores_path = output / "scores.parquet"
        _, scores_path = self.batch_scorer.score(selection, feature_result.matrix, scores_path)
        report_path = self.reporter.write(
            config,
            ingest_result.quality,
            feature_result,
            selection,
            evaluation,
            drift,
            output / "report.md",
        )
        logic_review = self.logic_reviewer.review(evaluation, report_path)
        code_review = self.code_reviewer.review([scores_path, report_path])
        reviews = {"logic": logic_review.__dict__, "code": code_review.__dict__}
        (output / "reviews.json").write_text(json.dumps(reviews, indent=2), encoding="utf-8")
        return OrchestratorResult(
            metrics=evaluation.metrics.__dict__,
            report_path=report_path,
            scores_path=scores_path,
        )


__all__ = ["Orchestrator", "OrchestratorResult"]
