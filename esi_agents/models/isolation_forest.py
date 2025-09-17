"""Isolation forest detector wrapper."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import CalibrationModel


@dataclass
class IsolationForestDetector:
    n_estimators: int = 200
    max_samples: int | float = "auto"
    contamination: float | str | None = "auto"
    random_state: int | None = 42
    calibrator: CalibrationModel | None = None

    def __post_init__(self) -> None:
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "IsolationForestDetector":
        self.model.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        raw = -self.model.score_samples(X)
        scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        if self.calibrator:
            return self.calibrator.transform(scores)
        return scores


__all__ = ["IsolationForestDetector"]
