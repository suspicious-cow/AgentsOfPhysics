"""Evaluator tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..agents.evaluator import EvaluatorAgent
from ..eval import compute_classification_metrics, scan_frequency_bands


def test_compute_classification_metrics():
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    labels = np.array([0, 0, 1, 1])
    metrics = compute_classification_metrics(scores, labels)
    assert metrics.roc_auc > 0.9
    assert metrics.pr_auc > 0.9


def test_band_scan_detects_peak():
    feature_frame = pd.DataFrame(
        {
            "fft_peak_freq": np.concatenate([np.linspace(0, 10, 20), np.linspace(20, 30, 20)]),
            "fft_peak_magnitude": np.concatenate([np.ones(20), np.full(20, 5)]),
        }
    )
    bands = scan_frequency_bands(feature_frame, n_bands=5)
    assert not bands.empty
    assert bands.iloc[0]["p_value"] <= 0.5


def test_evaluator_runs(tmp_path):
    scores = pd.DataFrame({"feature1": [0, 1, 2, 3], "feature2": [1, 1, 1, 1]})
    labels = pd.Series([0, 0, 1, 1])
    feature_frame = pd.DataFrame(
        {
            "window_start": pd.date_range("2023-01-01", periods=4, freq="s"),
            "fft_peak_freq": [1, 2, 3, 4],
            "fft_peak_magnitude": [1, 1, 2, 3],
        }
    )
    feature_frame = pd.concat([feature_frame, scores], axis=1)

    class DummyModel:
        def __init__(self):
            self.calibrator = None

        def score_samples(self, X):
            return X["feature1"].to_numpy()

    evaluator = EvaluatorAgent()
    result = evaluator.run(DummyModel(), scores, labels, feature_frame, tmp_path, prefix="dummy")
    assert "roc_auc" in result.metrics
    assert result.scores.shape[0] == 4
