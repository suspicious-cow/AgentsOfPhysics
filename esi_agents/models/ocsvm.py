"""One-Class SVM detector."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.svm import OneClassSVM

from .base import CalibrationModel


@dataclass
class OneClassSVMDetector:
    kernel: str = "rbf"
    gamma: str | float = "scale"
    nu: float = 0.05
    calibrator: CalibrationModel | None = None

    def __post_init__(self) -> None:
        self.model = OneClassSVM(kernel=self.kernel, gamma=self.gamma, nu=self.nu)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "OneClassSVMDetector":
        self.model.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        raw = -self.model.score_samples(X)
        scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        if self.calibrator:
            return self.calibrator.transform(scores)
        return scores


__all__ = ["OneClassSVMDetector"]
