from __future__ import annotations

import numpy as np

from esi_agents.agents import Evaluator, FeatureEngineer, ModelSelector, ModelTrainer


def test_evaluator_creates_artifacts(tmp_path, synthetic_signal):
    config = {
        "window": {"size": 50, "stride": 25},
        "features": {"time": True, "freq": True, "envelope": True, "orders": True},
        "models": [{"name": "isolation_forest"}],
    }
    engineer = FeatureEngineer()
    feature_result = engineer.transform(synthetic_signal, config)
    trainer = ModelTrainer()
    trained = trainer.train(feature_result.matrix, config)
    selector = ModelSelector()
    selection = selector.select(trained, labels=None)
    evaluator = Evaluator()
    artifacts = evaluator.evaluate(selection, feature_result, labels=None, output_dir=tmp_path)
    assert artifacts.metrics.alert_rate >= 0
    assert (tmp_path / "metrics.json").exists()
