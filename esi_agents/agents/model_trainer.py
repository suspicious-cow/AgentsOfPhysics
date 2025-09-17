"""Agent that trains anomaly detection models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..eval import fit_platt_scaler
from ..models import (
    ARIMAResidualDetector,
    HBOSDetector,
    IsolationForestDetector,
    LOFDetector,
    OneClassSVMDetector,
    STLResidualDetector,
)


@dataclass
class TrainedModel:
    name: str
    model: Any
    scores: np.ndarray


_MODEL_FACTORY = {
    "isolation_forest": IsolationForestDetector,
    "ocsvm": OneClassSVMDetector,
    "lof": LOFDetector,
    "hbos": HBOSDetector,
    "stl_resid": STLResidualDetector,
    "arima_resid": ARIMAResidualDetector,
}


class ModelTrainer:
    def train(
        self,
        features: pd.DataFrame,
        config: dict[str, Any],
        labels: np.ndarray | None = None,
    ) -> list[TrainedModel]:
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        X = features[numeric_cols].to_numpy(dtype=float)
        models_cfg = config.get("models", [{"name": "isolation_forest"}, {"name": "lof"}])
        trained: list[TrainedModel] = []
        for model_cfg in models_cfg:
            name = model_cfg["name"].lower()
            params = model_cfg.get("params", {})
            cls = _MODEL_FACTORY.get(name)
            if cls is None:
                raise ValueError(f"Unknown model '{name}'")
            model = cls(**params)
            model.fit(X, labels)
            scores = model.score_samples(X)
            if labels is not None and len(np.unique(labels)) > 1 and hasattr(model, "calibrator"):
                calibrator = fit_platt_scaler(scores, labels)
                model.calibrator = calibrator
                scores = model.calibrator.transform(scores)
            trained.append(TrainedModel(name=name, model=model, scores=scores))
        return trained


__all__ = ["ModelTrainer", "TrainedModel"]
