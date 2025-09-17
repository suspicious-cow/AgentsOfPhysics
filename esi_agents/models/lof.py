"""Local Outlier Factor wrapper."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from .base import BaseAnomalyModel


class LOFModel(BaseAnomalyModel):
    def __init__(self, n_neighbors: int = 20, metric: str = "minkowski") -> None:
        super().__init__("LocalOutlierFactor")
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, metric=metric)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.model.fit(X)

    def raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(X)


__all__ = ["LOFModel"]
