"""Frequency/order band scanning utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm  # type: ignore


@dataclass
class BandScanResult:
    band_start: float
    band_end: float
    z_score: float
    p_value: float


def band_scan(
    freqs: np.ndarray,
    magnitudes: np.ndarray,
    window_size: int = 5,
) -> list[BandScanResult]:
    if freqs.shape != magnitudes.shape:
        raise ValueError("freqs and magnitudes must have the same shape")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if freqs.size == 0:
        return []
    window = min(window_size, freqs.size)
    results: list[BandScanResult] = []
    rolling = np.convolve(magnitudes, np.ones(window), mode="valid") / window
    mean = float(np.mean(rolling))
    std = float(np.std(rolling) + 1e-6)
    for idx, value in enumerate(rolling):
        z = (value - mean) / std
        start = freqs[idx]
        end = freqs[min(idx + window - 1, freqs.size - 1)]
        p = float(norm.sf(abs(z)) * 2)
        results.append(BandScanResult(start, end, float(z), p))
    return results


def top_bands(results: list[BandScanResult], k: int = 3) -> list[BandScanResult]:
    return sorted(results, key=lambda r: abs(r.z_score), reverse=True)[:k]


__all__ = ["BandScanResult", "band_scan", "top_bands"]
