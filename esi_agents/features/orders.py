"""Order-tracking features."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .windows import Window


def compute_order_features(window: Window) -> Dict[str, float]:
    """Compute order-normalised spectral indicators."""

    rpm = window.rpm
    if not rpm or rpm <= 0:
        return {
            "order_1x_mag": 0.0,
            "order_2x_mag": 0.0,
            "order_sideband_ratio": 0.0,
        }

    fundamental_hz = rpm / 60.0
    values = window.values
    n = len(values)
    if n <= 1:
        return {"order_1x_mag": 0.0, "order_2x_mag": 0.0, "order_sideband_ratio": 0.0}

    freqs = np.fft.rfftfreq(n, d=1.0 / fundamental_hz)
    magnitudes = np.abs(np.fft.rfft(values))

    def closest_magnitude(target_freq: float) -> float:
        idx = int(np.argmin(np.abs(freqs - target_freq)))
        return float(magnitudes[idx]) if len(magnitudes) else 0.0

    mag_1x = closest_magnitude(fundamental_hz)
    mag_2x = closest_magnitude(2 * fundamental_hz)
    upper_side = closest_magnitude(fundamental_hz * 1.1)
    lower_side = closest_magnitude(fundamental_hz * 0.9)
    denominator = mag_1x or 1.0
    sideband_ratio = float((upper_side + lower_side) / denominator)

    return {
        "order_1x_mag": mag_1x,
        "order_2x_mag": mag_2x,
        "order_sideband_ratio": sideband_ratio,
    }


__all__ = ["compute_order_features"]
