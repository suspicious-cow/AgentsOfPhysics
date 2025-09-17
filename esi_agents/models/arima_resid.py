"""ARIMA residual anomaly model."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .base import BaseAnomalyModel


class ARIMAResidualModel(BaseAnomalyModel):
    def __init__(self, order: tuple[int, int, int] = (1, 0, 1)) -> None:
        super().__init__("ARIMAResidual")
        self.order = order
        self._model = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        series = X.mean(axis=1)
        self._model = ARIMA(series, order=self.order).fit()

    def raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        series = X.mean(axis=1)
        if self._model is None:
            return np.abs(series.to_numpy())
        forecast = self._model.predict(start=series.index[0], end=series.index[-1])
        forecast = forecast.reindex(series.index, method="nearest")
        residuals = series - forecast
        return np.abs(residuals.to_numpy())


__all__ = ["ARIMAResidualModel"]
