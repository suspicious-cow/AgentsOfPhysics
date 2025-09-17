"""Local Outlier Factor detector."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from .base import CalibrationModel


@dataclass
class LOFDetector:
    n_neighbors: int = 20
    contamination: float | str | None = "auto"
    metric: str = "minkowski"
    calibrator: CalibrationModel | None = None

    def __post_init__(self) -> None:
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
            metric=self.metric,
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "LOFDetector":
        self.model.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        raw = -self.model.score_samples(X)
        scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        if self.calibrator:
            return self.calibrator.transform(scores)
        return scores


__all__ = ["LOFDetector"]
