"""Evaluation metrics for anomaly detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics


@dataclass
class MetricResults:
    roc_auc: float
    pr_auc: float
    sic_max: float
    sic_threshold: float
    topk_precision: float
    alert_rate: float


def compute_classification_metrics(scores: np.ndarray, labels: np.ndarray, top_k: int = 10) -> MetricResults:
    """Compute standard metrics for anomaly detection."""

    roc_auc = skmetrics.roc_auc_score(labels, scores)
    precision, recall, thresholds = skmetrics.precision_recall_curve(labels, scores)
    pr_auc = skmetrics.auc(recall, precision)

    sic_values, sic_thresholds = _sic_curve(labels, scores)
    if len(sic_values):
        best_idx = int(np.argmax(sic_values))
        sic_max = float(sic_values[best_idx])
        sic_threshold = float(sic_thresholds[best_idx])
    else:  # pragma: no cover - degenerate
        sic_max, sic_threshold = 0.0, 0.0

    topk_precision = _top_k_precision(scores, labels, top_k=top_k)
    alert_rate = float(np.mean(scores >= sic_threshold)) if sic_threshold else float(np.mean(scores > np.quantile(scores, 0.95)))

    return MetricResults(
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
        sic_max=sic_max,
        sic_threshold=sic_threshold,
        topk_precision=topk_precision,
        alert_rate=alert_rate,
    )


def _sic_curve(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SIC-like surrogate curve using a threshold sweep."""

    thresholds = np.linspace(scores.min(), scores.max(), num=50)
    signal_efficiencies: list[float] = []
    background_efficiencies: list[float] = []
    for threshold in thresholds:
        selected = scores >= threshold
        if not np.any(selected):
            signal_efficiencies.append(0.0)
            background_efficiencies.append(0.0)
            continue
        signal_eff = np.mean(labels[selected]) if np.any(selected) else 0.0
        background_eff = np.mean(1 - labels[selected]) if np.any(selected) else 0.0
        signal_efficiencies.append(signal_eff)
        background_efficiencies.append(background_eff)

    signal_efficiencies = np.asarray(signal_efficiencies)
    background_efficiencies = np.asarray(background_efficiencies)

    with np.errstate(divide="ignore", invalid="ignore"):
        sic = np.where(background_efficiencies > 0, signal_efficiencies / np.sqrt(background_efficiencies), 0.0)

    return sic, thresholds


def _top_k_precision(scores: np.ndarray, labels: np.ndarray, top_k: int = 10) -> float:
    idx = np.argsort(scores)[::-1][:top_k]
    if len(idx) == 0:
        return 0.0
    return float(np.mean(labels[idx]))


def metrics_to_frame(metrics: MetricResults) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "roc_auc": metrics.roc_auc,
            "pr_auc": metrics.pr_auc,
            "sic_max": metrics.sic_max,
            "sic_threshold": metrics.sic_threshold,
            "topk_precision": metrics.topk_precision,
            "alert_rate": metrics.alert_rate,
        }
    ])


__all__ = ["MetricResults", "compute_classification_metrics", "metrics_to_frame"]
