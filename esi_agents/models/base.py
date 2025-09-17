"""Base anomaly model interface."""
from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from ..eval.calibration import ScoreCalibrator


class BaseAnomalyModel(abc.ABC):
    """Base class for anomaly detectors."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calibrator: Optional[ScoreCalibrator] = None

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Fit the model."""

    @abc.abstractmethod
    def raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw anomaly scores (higher means more anomalous)."""

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        scores = self.raw_scores(X)
        if self.calibrator:
            return self.calibrator.transform(scores)
        return scores

    def calibrate(self, scores: np.ndarray, reference: Optional[np.ndarray] = None) -> None:
        self.calibrator = ScoreCalibrator()
        self.calibrator.fit(scores, reference)

    def save(self, path: str) -> None:
        joblib.dump({"model": self, "calibrator": self.calibrator}, path)

    @classmethod
    def load(cls, path: str) -> "BaseAnomalyModel":
        payload = joblib.load(path)
        model = payload["model"]
        model.calibrator = payload.get("calibrator")
        return model

    def get_params(self) -> Dict[str, Any]:
        return {}


__all__ = ["BaseAnomalyModel"]
