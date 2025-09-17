"""Histogram Based Outlier Score (HBOS)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import CalibrationModel


@dataclass
class HBOSDetector:
    n_bins: int = 15
    calibrator: CalibrationModel | None = None
    histograms: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "HBOSDetector":
        self.histograms.clear()
        for i in range(X.shape[1]):
            hist, edges = np.histogram(X[:, i], bins=self.n_bins, density=True)
            hist = np.where(hist == 0, 1e-8, hist)
            self.histograms.append((hist, edges))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if not self.histograms:
            raise RuntimeError("HBOSDetector must be fitted before scoring")
        scores = []
        for row in X:
            log_density = 0.0
            for value, (hist, edges) in zip(row, self.histograms):
                idx = np.searchsorted(edges, value, side="right") - 1
                idx = np.clip(idx, 0, len(hist) - 1)
                density = hist[idx]
                log_density += np.log(density)
            scores.append(float(-log_density))
        scores_arr = np.asarray(scores)
        scores_arr = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min() + 1e-8)
        if self.calibrator:
            return self.calibrator.transform(scores_arr)
        return scores_arr


__all__ = ["HBOSDetector"]
