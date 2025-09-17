"""Frequency-domain features."""
from __future__ import annotations

import numpy as np

from .windows import Window


def _next_pow_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def compute_frequency_features(window: Window, n_fft: int | None = None) -> dict[str, float]:
    values = window.values.astype(float)
    if values.size == 0:
        return {
            "freq_power": 0.0,
            "freq_centroid": 0.0,
            "freq_bandpower_low": 0.0,
            "freq_bandpower_mid": 0.0,
            "freq_bandpower_high": 0.0,
        }
    n_fft = n_fft or _next_pow_two(values.size)
    window_fn = np.hanning(values.size)
    fft_values = np.fft.rfft(values * window_fn, n=n_fft)
    magnitudes = np.abs(fft_values)
    power_spectrum = magnitudes**2
    total_power = float(power_spectrum.sum())
    if window.sampling_rate_hz:
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / window.sampling_rate_hz)
    else:
        freqs = np.fft.rfftfreq(n_fft)
    centroid = float((freqs * power_spectrum).sum() / power_spectrum.sum()) if total_power else 0.0

    thirds = np.array_split(power_spectrum, 3)
    bandpowers = [float(part.sum()) for part in thirds]
    while len(bandpowers) < 3:
        bandpowers.append(0.0)

    return {
        "freq_power": total_power,
        "freq_centroid": centroid,
        "freq_bandpower_low": bandpowers[0],
        "freq_bandpower_mid": bandpowers[1],
        "freq_bandpower_high": bandpowers[2],
    }


def dominant_frequencies(window: Window, top_k: int = 3, n_fft: int | None = None) -> dict[str, float]:
    values = window.values.astype(float)
    if values.size == 0:
        return {f"freq_peak_{i}": 0.0 for i in range(1, top_k + 1)}
    n_fft = n_fft or _next_pow_two(values.size)
    fft_values = np.fft.rfft(values * np.hanning(values.size), n=n_fft)
    magnitudes = np.abs(fft_values)
    freqs = (
        np.fft.rfftfreq(n_fft, d=1.0 / window.sampling_rate_hz)
        if window.sampling_rate_hz
        else np.fft.rfftfreq(n_fft)
    )
    indices = np.argsort(magnitudes)[::-1][:top_k]
    return {f"freq_peak_{i+1}": float(freqs[idx]) for i, idx in enumerate(indices)}


__all__ = ["compute_frequency_features", "dominant_frequencies"]
