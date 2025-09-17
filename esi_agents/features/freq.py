"""Frequency domain features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .windows import Window


@dataclass
class FrequencyFeatureConfig:
    """Configuration for FFT-based features."""

    sampling_rate_hz: float | None = None


def compute_frequency_features(window: Window, config: FrequencyFeatureConfig | None = None) -> Dict[str, float]:
    """Compute FFT-based features."""

    cfg = config or FrequencyFeatureConfig(sampling_rate_hz=window.rpm)
    values = window.values
    n = len(values)
    if n <= 1:
        return {"fft_peak_freq": 0.0, "fft_peak_magnitude": 0.0, "spectral_centroid": 0.0}

    sampling_rate = cfg.sampling_rate_hz or window.rpm or 1.0
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    fft = np.fft.rfft(values)
    magnitudes = np.abs(fft)

    peak_idx = int(np.argmax(magnitudes))
    peak_freq = float(freqs[peak_idx])
    peak_mag = float(magnitudes[peak_idx])

    total_mag = np.sum(magnitudes) or 1.0
    spectral_centroid = float(np.sum(freqs * magnitudes) / total_mag)

    thirds = np.array_split(magnitudes, 3)
    bandpowers = [float(np.mean(chunk)) if len(chunk) else 0.0 for chunk in thirds]

    return {
        "fft_peak_freq": peak_freq,
        "fft_peak_magnitude": peak_mag,
        "spectral_centroid": spectral_centroid,
        "bandpower_low": bandpowers[0] if len(bandpowers) > 0 else 0.0,
        "bandpower_mid": bandpowers[1] if len(bandpowers) > 1 else 0.0,
        "bandpower_high": bandpowers[2] if len(bandpowers) > 2 else 0.0,
    }


__all__ = ["FrequencyFeatureConfig", "compute_frequency_features"]
