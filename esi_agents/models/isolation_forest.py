"""Isolation Forest anomaly detector wrapper."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .base import BaseAnomalyModel


class IsolationForestModel(BaseAnomalyModel):
    def __init__(self, random_state: Optional[int] = 42) -> None:
        super().__init__("IsolationForest")
        self.model = IsolationForest(random_state=random_state, contamination="auto")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.model.fit(X)

    def raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(X)


__all__ = ["IsolationForestModel"]
