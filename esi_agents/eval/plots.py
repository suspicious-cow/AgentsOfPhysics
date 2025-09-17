"""Plotting helpers that adhere to reviewer constraints."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from .band_scan import BandScanResult


_FIGSIZE = (5.12, 5.12)
_DPI = 100


def plot_roc_curve(y_true: np.ndarray, scores: np.ndarray, path: str | Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.grid(True)
    ax.legend()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, scores: np.ndarray, path: str | Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    ax.plot(recall, precision, label="PR")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(True)
    ax.legend()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_band_scan(results: Iterable[BandScanResult], path: str | Path) -> None:
    bands = list(results)
    if not bands:
        raise ValueError("band scan results are empty")
    starts = [b.band_start for b in bands]
    ends = [b.band_end for b in bands]
    z_scores = [b.z_score for b in bands]
    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    centers = [(s + e) / 2 for s, e in zip(starts, ends)]
    widths = [e - s for s, e in zip(starts, ends)]
    ax.bar(centers, z_scores, width=widths, align="center")
    ax.set_xlabel("Frequency / Order")
    ax.set_ylabel("Z-score")
    ax.set_title("Band Scan Anomalies")
    ax.grid(True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


__all__ = ["plot_roc_curve", "plot_pr_curve", "plot_band_scan"]
