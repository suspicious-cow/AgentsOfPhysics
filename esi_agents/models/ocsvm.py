"""One-Class SVM wrapper."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

from .base import BaseAnomalyModel


class OneClassSVMModel(BaseAnomalyModel):
    def __init__(self, kernel: str = "rbf", nu: float = 0.05) -> None:
        super().__init__("OneClassSVM")
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma="scale")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.model.fit(X)

    def raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(X)


__all__ = ["OneClassSVMModel"]
