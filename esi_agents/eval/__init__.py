"""Evaluation toolkit for the anomaly detection platform."""
from .metrics import MetricsResult, compute_classification_metrics, precision_recall_table
from .band_scan import BandScanResult, band_scan, top_bands
from .plots import plot_roc_curve, plot_pr_curve, plot_band_scan
from .calibration import fit_platt_scaler, calibrate_scores

__all__ = [
    "MetricsResult",
    "compute_classification_metrics",
    "precision_recall_table",
    "BandScanResult",
    "band_scan",
    "top_bands",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_band_scan",
    "fit_platt_scaler",
    "calibrate_scores",
]
