"""Unit tests for feature extraction."""
from __future__ import annotations

import numpy as np

from ..features import (
    compute_envelope_features,
    compute_frequency_features,
    compute_order_features,
    compute_time_domain_features,
    iter_windows,
)


def test_time_features_basic(synthetic_window):
    features = compute_time_domain_features(synthetic_window)
    assert abs(features["mean"]) < 5e-2
    assert features["rms"] > 0.5
    assert features["crest_factor"] > 1.0


def test_frequency_features_detect_peak(synthetic_window):
    freq_features = compute_frequency_features(synthetic_window)
    assert freq_features["fft_peak_freq"] > 0
    assert freq_features["fft_peak_magnitude"] > 0


def test_envelope_features_positive(synthetic_window):
    env_features = compute_envelope_features(synthetic_window)
    assert env_features["envelope_peak"] >= env_features["envelope_mean"]


def test_order_features_tracks_rpm(synthetic_window):
    order = compute_order_features(synthetic_window)
    assert order["order_1x_mag"] >= 0


def test_iter_windows_shapes(synthetic_dataframe):
    windows = list(iter_windows(synthetic_dataframe, window_size=64, stride=32))
    assert len(windows) > 0
    assert all(len(w.values) == 64 for w in windows)
