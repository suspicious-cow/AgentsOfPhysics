from __future__ import annotations

import numpy as np

from esi_agents.features import (
    compute_envelope_features,
    compute_frequency_features,
    compute_order_features,
    compute_time_features,
    generate_windows,
)


def test_feature_generation(synthetic_signal):
    windows = generate_windows(synthetic_signal, window_size=50, stride=25)
    assert windows
    time_feats = compute_time_features(windows[0])
    freq_feats = compute_frequency_features(windows[0])
    env_feats = compute_envelope_features(windows[0])
    order_feats = compute_order_features(windows[0])
    assert time_feats["time_rms"] > 0
    assert freq_feats["freq_power"] >= 0
    assert env_feats["envelope_peak"] >= env_feats["envelope_mean"]
    assert "order_1_amplitude" in order_feats
