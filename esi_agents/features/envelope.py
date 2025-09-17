"""Envelope and demodulation features."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import hilbert

from .windows import Window


def compute_envelope_features(window: Window) -> Dict[str, float]:
    """Compute envelope-based bearing indicators."""

    values = window.values
    if len(values) == 0:
        return {"envelope_mean": 0.0, "envelope_peak": 0.0, "envelope_kurtosis": 0.0}

    analytic_signal = hilbert(values)
    envelope = np.abs(analytic_signal)
    mean_env = float(np.mean(envelope))
    peak_env = float(np.max(envelope))
    kurt_env = float(
        (np.mean((envelope - mean_env) ** 4) / (np.std(envelope) ** 4)) if np.std(envelope) else 0.0
    )

    return {
        "envelope_mean": mean_env,
        "envelope_peak": peak_env,
        "envelope_kurtosis": kurt_env,
    }


__all__ = ["compute_envelope_features"]
