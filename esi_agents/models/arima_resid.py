"""ARIMA-lite residual based detector."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.arima.model import ARIMA  # type: ignore

from .base import CalibrationModel


@dataclass
class ARIMAResidualDetector:
    order: tuple[int, int, int] = (1, 0, 0)
    feature_index: int = 0
    calibrator: CalibrationModel | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ARIMAResidualDetector":
        series = X[:, self.feature_index].astype(float)
        if series.size < max(self.order) + 5:
            raise ValueError("ARIMAResidualDetector requires more observations")
        model = ARIMA(series, order=self.order)
        self.result_ = model.fit()
        resid = self.result_.resid
        self.scale_ = float(np.median(np.abs(resid)) + 1e-6)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "result_"):
            raise RuntimeError("ARIMAResidualDetector must be fitted before scoring")
        series = X[:, self.feature_index].astype(float)
        forecast = self.result_.forecast(steps=len(series))
        residual = np.abs(series - forecast) / self.scale_
        residual = residual.astype(float)
        residual = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
        if self.calibrator:
            return self.calibrator.transform(residual)
        return residual


__all__ = ["ARIMAResidualDetector"]
