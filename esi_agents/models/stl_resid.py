"""Detector based on robust STL residuals."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.seasonal import STL  # type: ignore

from .base import CalibrationModel


@dataclass
class STLResidualDetector:
    period: int = 24
    feature_index: int = 0
    calibrator: CalibrationModel | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "STLResidualDetector":
        series = X[:, self.feature_index].astype(float)
        if series.size < self.period * 2:
            raise ValueError("STLResidualDetector requires at least two periods of data")
        stl = STL(series, period=self.period, robust=True)
        result = stl.fit()
        self.trend_level_ = float(np.median(result.trend))
        self.seasonal_pattern_ = result.seasonal[: self.period]
        resid = result.resid
        self.scale_ = float(np.median(np.abs(resid)) + 1e-6)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "seasonal_pattern_"):
            raise RuntimeError("STLResidualDetector must be fitted before scoring")
        series = X[:, self.feature_index].astype(float)
        pattern = self.seasonal_pattern_
        scores = []
        for idx, value in enumerate(series):
            expected = self.trend_level_ + pattern[idx % len(pattern)]
            residual = abs(value - expected) / self.scale_
            scores.append(residual)
        scores_arr = np.asarray(scores)
        scores_arr = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min() + 1e-8)
        if self.calibrator:
            return self.calibrator.transform(scores_arr)
        return scores_arr


__all__ = ["STLResidualDetector"]
