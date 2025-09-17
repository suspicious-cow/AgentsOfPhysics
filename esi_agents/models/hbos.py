"""Histogram-based outlier score."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseAnomalyModel


class HBOSModel(BaseAnomalyModel):
    """Lightweight HBOS implementation without external dependencies."""

    def __init__(self, n_bins: int = 10) -> None:
        super().__init__("HBOS")
        self.n_bins = n_bins
        self.histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.histograms.clear()
        for column in X.columns:
            values = X[column].to_numpy()
            hist, edges = np.histogram(values, bins=self.n_bins, density=True)
            hist = np.where(hist == 0, 1e-8, hist)
            self.histograms[column] = (hist, edges)

    def raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        scores = np.zeros(len(X))
        for column, (hist, edges) in self.histograms.items():
            values = X[column].to_numpy()
            bin_indices = np.clip(np.digitize(values, edges) - 1, 0, len(hist) - 1)
            probs = hist[bin_indices]
            scores += -np.log(probs)
        return scores


__all__ = ["HBOSModel"]
