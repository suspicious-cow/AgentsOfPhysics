"""Score calibration helpers."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..models.base import CalibrationModel


def fit_platt_scaler(scores: np.ndarray, labels: np.ndarray) -> CalibrationModel:
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    model = LogisticRegression()
    model.fit(scores.reshape(-1, 1), labels)
    slope = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    return CalibrationModel(slope, intercept)


def calibrate_scores(scores: np.ndarray, calibrator: CalibrationModel) -> np.ndarray:
    return calibrator.transform(scores)


__all__ = ["fit_platt_scaler", "calibrate_scores"]
