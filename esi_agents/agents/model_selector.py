"""Model selection agent."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from .evaluator import EvaluatorAgent, EvaluationResult
from .model_trainer import ModelTrainerResult


@dataclass
class ModelSelectionResult:
    best_model_name: str
    best_model: object
    validation_metrics: Dict[str, Dict[str, float]]
    validation_scores: Dict[str, pd.DataFrame]


class ModelSelectorAgent:
    def __init__(self, evaluator: EvaluatorAgent) -> None:
        self.evaluator = evaluator

    def run(
        self,
        trainer_result: ModelTrainerResult,
        feature_frame: pd.DataFrame,
        output_dir: Path,
        config: Dict,
    ) -> ModelSelectionResult:
        val_X, val_y = trainer_result.splits["val"]
        validation_metrics: Dict[str, Dict[str, float]] = {}
        validation_scores: Dict[str, pd.DataFrame] = {}

        best_model_name = None
        best_model = None
        best_key = (-1.0, -1.0, -1.0)

        for name, model in trainer_result.models.items():
            raw_scores = model.raw_scores(val_X)
            model.calibrate(raw_scores)
            eval_result = self.evaluator.run(
                model,
                val_X,
                val_y,
                feature_frame.loc[val_X.index],
                output_dir,
                prefix=f"{name}_val",
            )
            validation_metrics[name] = eval_result.metrics
            validation_scores[name] = eval_result.scores

            key = (
                eval_result.metrics.get("pr_auc", 0.0),
                eval_result.metrics.get("roc_auc", 0.0),
                eval_result.metrics.get("sic_max", 0.0),
            )
            if key > best_key:
                best_key = key
                best_model_name = name
                best_model = model

        if best_model is None or best_model_name is None:
            raise RuntimeError("No valid model found during selection")

        return ModelSelectionResult(
            best_model_name=best_model_name,
            best_model=best_model,
            validation_metrics=validation_metrics,
            validation_scores=validation_scores,
        )


__all__ = ["ModelSelectorAgent", "ModelSelectionResult"]
