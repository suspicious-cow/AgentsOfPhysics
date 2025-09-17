"""Base interfaces for anomaly detectors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class AnomalyDetector(Protocol):
    """Protocol describing the estimator interface used across the project."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AnomalyDetector":
        ...

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        ...


@dataclass
class CalibrationModel:
    """Stores calibration parameters for affine rescaling."""

    slope: float
    intercept: float

    def transform(self, scores: np.ndarray) -> np.ndarray:
        logits = self.slope * scores + self.intercept
        return 1.0 / (1.0 + np.exp(-logits))


__all__ = ["AnomalyDetector", "CalibrationModel"]
