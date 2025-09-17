from __future__ import annotations

import numpy as np

from esi_agents.models import (
    HBOSDetector,
    IsolationForestDetector,
    LOFDetector,
    OneClassSVMDetector,
)


def test_detectors_produce_normalised_scores():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 4))
    detectors = [
        IsolationForestDetector(random_state=42),
        LOFDetector(),
        OneClassSVMDetector(),
        HBOSDetector(),
    ]
    for detector in detectors:
        detector.fit(X)
        scores = detector.score_samples(X)
        assert scores.shape == (100,)
        assert scores.min() >= 0
        assert scores.max() <= 1 + 1e-6
