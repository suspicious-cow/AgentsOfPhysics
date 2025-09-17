"""Time domain feature computation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from .windows import Window


@dataclass
class TimeFeatureConfig:
    """Configuration for time-domain feature extraction."""

    moving_std_window: int = 10


def compute_time_domain_features(window: Window, config: TimeFeatureConfig | None = None) -> Dict[str, float]:
    """Compute time-domain features for a window."""

    cfg = config or TimeFeatureConfig()
    values = window.values
    series = pd.Series(values)

    rms = float(np.sqrt(np.mean(np.square(values))))
    std = float(series.std(ddof=0))
    mean = float(series.mean())
    kurt = float(stats.kurtosis(values, fisher=False)) if len(values) > 3 else 0.0
    crest = float(np.max(np.abs(values)) / rms) if rms else 0.0
    peak_to_peak = float(series.max() - series.min())
    moving_window = max(1, min(cfg.moving_std_window, len(values)))
    moving_std = float(series.rolling(window=moving_window, min_periods=1).std(ddof=0).mean())
    z_scores = (values - mean) / std if std else np.zeros_like(values)
    rolling_z = float(np.nanmax(np.abs(z_scores))) if len(z_scores) else 0.0

    x = np.arange(len(values))
    if len(values) > 1:
        slope, intercept = np.polyfit(x, values, 1)
    else:  # pragma: no cover - degenerate
        slope, intercept = 0.0, float(values[0]) if len(values) else 0.0
    trend_strength = float(slope)
    half = len(values) // 2
    seasonal_strength = float(np.mean(values[half:]) - np.mean(values[:half])) if half else 0.0

    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "kurtosis": kurt,
        "crest_factor": crest,
        "peak_to_peak": peak_to_peak,
        "moving_std": moving_std,
        "rolling_z_score": rolling_z,
        "trend_slope": trend_strength,
        "seasonal_offset": seasonal_strength,
    }


__all__ = ["TimeFeatureConfig", "compute_time_domain_features"]
