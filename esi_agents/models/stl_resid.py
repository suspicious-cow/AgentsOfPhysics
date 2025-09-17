"""Residual-based model using STL decomposition."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from .base import BaseAnomalyModel


class STLResidualModel(BaseAnomalyModel):
    def __init__(self, period: int = 24) -> None:
        super().__init__("STLResidual")
        self.period = period
        self._baseline: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        series = X.mean(axis=1)
        stl = STL(series, period=self.period, robust=True)
        result = stl.fit()
        self._baseline = (result.trend + result.seasonal).to_numpy()

    def raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        series = X.mean(axis=1).to_numpy()
        if self._baseline is None:
            return np.abs(series)
        baseline = np.interp(
            np.linspace(0, len(self._baseline) - 1, num=len(series)),
            np.arange(len(self._baseline)),
            self._baseline,
        )
        return np.abs(series - baseline)


__all__ = ["STLResidualModel"]
