"""Evaluation metrics for anomaly detection."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class MetricsResult:
    roc_auc: float | None
    pr_auc: float | None
    sic_surrogate: float | None
    top_k_precision: float | None
    alert_rate: float

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def _sic_surrogate(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    thresholds = np.linspace(scores.min(), scores.max(), 50)
    best = 0.0
    for threshold in thresholds:
        signal = (scores >= threshold) & (y_true == 1)
        background = (scores >= threshold) & (y_true == 0)
        eps_s = signal.sum() / max((y_true == 1).sum(), 1)
        eps_b = background.sum() / max((y_true == 0).sum(), 1)
        if eps_b == 0:
            continue
        metric = eps_s / np.sqrt(eps_b)
        if metric > best:
            best = float(metric)
    return best if best > 0 else None


def compute_classification_metrics(
    y_true: np.ndarray | None, scores: np.ndarray, top_k: int = 10
) -> MetricsResult:
    if y_true is None:
        alert_rate = float(np.mean(scores >= 0.5))
        return MetricsResult(None, None, None, None, alert_rate)
    y_true = y_true.astype(int)
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = None
    try:
        pr_auc = float(average_precision_score(y_true, scores))
    except ValueError:
        pr_auc = None
    sic = _sic_surrogate(y_true, scores)
    order = np.argsort(scores)[::-1]
    k = min(top_k, len(scores))
    topk_labels = y_true[order[:k]]
    top_k_precision = float(topk_labels.mean()) if k > 0 else None
    alert_rate = float(np.mean(scores >= 0.5))
    return MetricsResult(roc_auc, pr_auc, sic, top_k_precision, alert_rate)


def precision_recall_table(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    return np.column_stack([precision[:-1], recall[:-1], thresholds])


__all__ = ["MetricsResult", "compute_classification_metrics", "precision_recall_table"]
