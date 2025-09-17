"""Time-domain features for ESI signals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .windows import Window


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def compute_time_features(window: Window) -> dict[str, float]:
    """Compute summary statistics for a window."""

    values = window.values.astype(float)
    mean = float(np.mean(values))
    std = _safe_std(values)
    rms = float(np.sqrt(np.mean(values**2))) if values.size else 0.0
    centered = values - mean
    kurtosis = float(
        ((centered**4).mean() / (std**4)) - 3 if std and values.size > 3 else 0.0
    )
    crest_factor = float(np.max(np.abs(values)) / rms) if rms else 0.0
    peak_to_peak = float(np.max(values) - np.min(values)) if values.size else 0.0
    if values.size > 4:
        rolling = pd.Series(values).rolling(max(2, values.size // 4)).std(ddof=0)
        moving_std = float(rolling.dropna().mean()) if not rolling.dropna().empty else std
    else:
        moving_std = std
    if std:
        zscores = np.abs((values - mean) / std)
        rolling_z = float(np.max(zscores))
    else:
        rolling_z = 0.0
    idx = np.arange(values.size)
    if values.size > 1:
        slope, intercept = np.polyfit(idx, values, 1)
        trend = slope
        seasonal_component = values - (slope * idx + intercept)
        seasonal_energy = float(np.var(seasonal_component))
    else:
        trend = 0.0
        seasonal_energy = 0.0

    return {
        "time_mean": mean,
        "time_std": std,
        "time_rms": rms,
        "time_kurtosis": kurtosis,
        "time_crest_factor": crest_factor,
        "time_peak_to_peak": peak_to_peak,
        "time_moving_std": float(moving_std),
        "time_rolling_zscore_max": rolling_z,
        "time_trend_slope": float(trend),
        "time_seasonal_energy": seasonal_energy,
    }


__all__ = ["compute_time_features"]
