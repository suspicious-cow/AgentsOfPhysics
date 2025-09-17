"""Evaluation utilities."""
from .metrics import MetricResults, compute_classification_metrics, metrics_to_frame
from .band_scan import scan_frequency_bands, BandScanResult
from .calibration import ScoreCalibrator
from .plots import plot_roc_curve, plot_pr_curve, plot_band_scan

__all__ = [
    "MetricResults",
    "compute_classification_metrics",
    "metrics_to_frame",
    "scan_frequency_bands",
    "BandScanResult",
    "ScoreCalibrator",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_band_scan",
]
