"""Agent that selects the best performing model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..eval import MetricsResult, compute_classification_metrics
from .model_trainer import TrainedModel


@dataclass
class SelectionResult:
    best_model: TrainedModel
    metrics: dict[str, MetricsResult]


class ModelSelector:
    def select(
        self,
        trained: list[TrainedModel],
        labels: np.ndarray | None,
        top_k: int = 10,
    ) -> SelectionResult:
        metrics: dict[str, MetricsResult] = {}
        best_model: TrainedModel | None = None
        best_score = -np.inf
        for tm in trained:
            result = compute_classification_metrics(labels, tm.scores, top_k=top_k)
            metrics[tm.name] = result
            primary = result.pr_auc if result.pr_auc is not None else -np.inf
            secondary = result.roc_auc if result.roc_auc is not None else -np.inf
            tertiary = result.sic_surrogate if result.sic_surrogate is not None else -np.inf
            score = (primary, secondary, tertiary)
            if best_model is None or score > best_score:
                best_model = tm
                best_score = score
        if best_model is None:
            raise RuntimeError("No trained models provided")
        return SelectionResult(best_model=best_model, metrics=metrics)


__all__ = ["ModelSelector", "SelectionResult"]
