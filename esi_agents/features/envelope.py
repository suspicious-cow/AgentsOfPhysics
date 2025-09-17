"""Envelope and demodulation features."""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert  # type: ignore

from .windows import Window


def compute_envelope_features(window: Window) -> dict[str, float]:
    values = window.values.astype(float)
    if values.size == 0:
        return {
            "envelope_mean": 0.0,
            "envelope_rms": 0.0,
            "envelope_peak": 0.0,
        }
    analytic_signal = hilbert(values)
    envelope = np.abs(analytic_signal)
    mean_env = float(np.mean(envelope))
    rms_env = float(np.sqrt(np.mean(envelope**2)))
    peak_env = float(np.max(envelope))
    return {
        "envelope_mean": mean_env,
        "envelope_rms": rms_env,
        "envelope_peak": peak_env,
    }


def envelope_spectrum(window: Window, n_fft: int | None = None) -> dict[str, float]:
    values = window.values.astype(float)
    if values.size == 0:
        return {"envelope_peak_freq": 0.0}
    analytic_signal = hilbert(values)
    envelope = np.abs(analytic_signal)
    n = n_fft or (1 << (envelope.size - 1).bit_length())
    spectrum = np.fft.rfft(envelope * np.hanning(envelope.size), n=n)
    magnitudes = np.abs(spectrum)
    if window.sampling_rate_hz:
        freqs = np.fft.rfftfreq(n, d=1.0 / window.sampling_rate_hz)
    else:
        freqs = np.fft.rfftfreq(n)
    idx = int(np.argmax(magnitudes))
    return {"envelope_peak_freq": float(freqs[idx])}


__all__ = ["compute_envelope_features", "envelope_spectrum"]
