"""Model tests."""
from __future__ import annotations

import numpy as np

from ..agents.feature_engineer import FeatureEngineerAgent
from ..models import (
    HBOSModel,
    IsolationForestModel,
    LOFModel,
    OneClassSVMModel,
)
from ..schema import SchemaRegistry


def test_models_produce_scores(synthetic_dataframe):
    schema = SchemaRegistry()
    schema.register_from_dataframe(synthetic_dataframe)
    features = FeatureEngineerAgent().run(
        synthetic_dataframe,
        schema,
        {"features": {"window_size": 64, "stride": 32, "rpm_column": "rpm"}},
    )
    X = features.features[features.feature_columns].fillna(0.0)
    models = [IsolationForestModel(), LOFModel(), HBOSModel(), OneClassSVMModel()]
    for model in models:
        model.fit(X)
        raw = model.raw_scores(X)
        model.calibrate(raw)
        scores = model.score_samples(X)
        assert scores.shape[0] == X.shape[0]
        assert np.all(np.isfinite(scores))
