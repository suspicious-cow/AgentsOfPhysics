"""Band-scan style evaluation mirroring bump hunts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class BandScanResult:
    band_start: float
    band_end: float
    score: float
    p_value: float


def scan_frequency_bands(
    feature_frame: pd.DataFrame,
    freq_column: str = "fft_peak_freq",
    magnitude_column: str = "fft_peak_magnitude",
    n_bands: int = 10,
) -> pd.DataFrame:
    """Aggregate anomalies across frequency bands and compute p-values."""

    if feature_frame.empty:
        return pd.DataFrame(columns=["band_start", "band_end", "score", "p_value"])

    frequencies = feature_frame[freq_column].to_numpy()
    magnitudes = feature_frame[magnitude_column].to_numpy()
    freq_min, freq_max = np.min(frequencies), np.max(frequencies)
    bins = np.linspace(freq_min, freq_max, num=n_bands + 1)

    results: list[BandScanResult] = []
    global_mean = float(np.mean(magnitudes)) or 1e-6
    global_std = float(np.std(magnitudes)) or 1.0

    for start, end in zip(bins[:-1], bins[1:]):
        mask = (frequencies >= start) & (frequencies < end)
        if not np.any(mask):
            continue
        band_mags = magnitudes[mask]
        score = float(np.mean(band_mags))
        z = (score - global_mean) / global_std
        p_value = float(1 - norm.cdf(np.abs(z)))
        results.append(BandScanResult(band_start=float(start), band_end=float(end), score=score, p_value=p_value))

    return pd.DataFrame([r.__dict__ for r in results]).sort_values("p_value")


__all__ = ["scan_frequency_bands", "BandScanResult"]
