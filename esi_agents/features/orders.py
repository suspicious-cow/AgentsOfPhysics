"""Order tracking features for rotating machinery."""
from __future__ import annotations

import numpy as np

from .windows import Window


def compute_order_features(
    window: Window, orders: tuple[int, ...] = (1, 2, 3), n_fft: int | None = None
) -> dict[str, float]:
    rpm = window.extras.get("rpm") if window.extras else None
    if rpm is None or len(rpm) == 0 or not window.sampling_rate_hz:
        return {f"order_{order}_amplitude": 0.0 for order in orders}
    base_freq = float(np.median(rpm) / 60.0)
    if base_freq <= 0:
        return {f"order_{order}_amplitude": 0.0 for order in orders}
    values = window.values.astype(float)
    if values.size == 0:
        return {f"order_{order}_amplitude": 0.0 for order in orders}
    n_fft = n_fft or (1 << (values.size - 1).bit_length())
    spectrum = np.fft.rfft(values * np.hanning(values.size), n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / window.sampling_rate_hz)
    magnitudes = np.abs(spectrum)
    results: dict[str, float] = {}
    for order in orders:
        target_freq = order * base_freq
        idx = int(np.argmin(np.abs(freqs - target_freq)))
        results[f"order_{order}_amplitude"] = float(magnitudes[idx])
    return results


def compute_sideband_features(window: Window, sideband_offset_hz: float = 1.0) -> dict[str, float]:
    values = window.values.astype(float)
    if values.size == 0 or not window.sampling_rate_hz:
        return {"sideband_ratio": 0.0}
    n_fft = 1 << (values.size - 1).bit_length()
    spectrum = np.fft.rfft(values * np.hanning(values.size), n=n_fft)
    magnitudes = np.abs(spectrum)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / window.sampling_rate_hz)
    peak_idx = int(np.argmax(magnitudes))
    peak_freq = freqs[peak_idx]
    lower_idx = int(np.argmin(np.abs(freqs - (peak_freq - sideband_offset_hz))))
    upper_idx = int(np.argmin(np.abs(freqs - (peak_freq + sideband_offset_hz))))
    carrier = magnitudes[peak_idx]
    sideband = magnitudes[lower_idx] + magnitudes[upper_idx]
    ratio = float(sideband / carrier) if carrier else 0.0
    return {"sideband_ratio": ratio}


__all__ = ["compute_order_features", "compute_sideband_features"]
