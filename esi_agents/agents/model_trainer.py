"""Model training agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..models import (
    ARIMAResidualModel,
    HBOSModel,
    IsolationForestModel,
    LOFModel,
    OneClassSVMModel,
    STLResidualModel,
)


_MODEL_FACTORY = {
    "isolation_forest": IsolationForestModel,
    "lof": LOFModel,
    "hbos": HBOSModel,
    "ocsvm": OneClassSVMModel,
    "stl_resid": STLResidualModel,
    "arima_resid": ARIMAResidualModel,
}


@dataclass
class ModelTrainerResult:
    models: Dict[str, object]
    feature_columns: List[str]
    splits: Dict[str, Tuple[pd.DataFrame, pd.Series]]


class ModelTrainerAgent:
    """Train candidate models using time-aware splits."""

    def run(self, feature_frame: pd.DataFrame, feature_columns: List[str], config: Dict) -> ModelTrainerResult:
        model_cfg = config["models"]
        candidate_names = model_cfg.get("candidates", list(_MODEL_FACTORY.keys()))
        train_ratio = float(model_cfg.get("train_ratio", 0.6))
        val_ratio = float(model_cfg.get("val_ratio", 0.2))

        features_sorted = feature_frame.sort_values("window_start").reset_index(drop=True)
        X = features_sorted[feature_columns].fillna(0.0)
        y = (
            features_sorted["label"]
            if "label" in features_sorted.columns
            else pd.Series(np.zeros(len(X)), index=X.index)
        )

        n = len(X)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_X, train_y = X.iloc[:train_end], y.iloc[:train_end]
        val_X, val_y = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        test_X, test_y = X.iloc[val_end:], y.iloc[val_end:]

        splits = {
            "train": (train_X, train_y),
            "val": (val_X, val_y),
            "test": (test_X, test_y),
        }

        models: Dict[str, object] = {}
        for name in candidate_names:
            key = name.lower()
            model_class = _MODEL_FACTORY.get(key)
            if model_class is None:
                raise ValueError(f"Unknown model type: {name}")
            params = model_cfg.get("params", {}).get(key, {})
            model = model_class(**params)
            model.fit(train_X)
            models[key] = model

        return ModelTrainerResult(models=models, feature_columns=feature_columns, splits=splits)


__all__ = ["ModelTrainerAgent", "ModelTrainerResult"]
