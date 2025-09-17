"""Score calibration utilities."""
from __future__ import annotations

import numpy as np


class ScoreCalibrator:
    """Simple min-max calibrator to map scores to [0, 1]."""

    def __init__(self) -> None:
        self.min_score: float = 0.0
        self.max_score: float = 1.0

    def fit(self, scores: np.ndarray, reference: np.ndarray | None = None) -> None:
        all_scores = scores if reference is None else np.concatenate([scores, reference])
        self.min_score = float(np.min(all_scores))
        self.max_score = float(np.max(all_scores))
        if np.isclose(self.min_score, self.max_score):
            self.max_score = self.min_score + 1.0

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scaled = (scores - self.min_score) / (self.max_score - self.min_score)
        return np.clip(scaled, 0.0, 1.0)


__all__ = ["ScoreCalibrator"]
