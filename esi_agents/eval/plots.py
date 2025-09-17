"""Plotting utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGSIZE = (5.12, 5.12)
DPI = 100


def plot_roc_curve(fpr: Iterable[float], tpr: Iterable[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_pr_curve(recall: Iterable[float], precision: Iterable[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(recall, precision, label="PR")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_band_scan(bands: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    if bands.empty:
        ax.text(0.5, 0.5, "No bands", ha="center", va="center")
    else:
        centers = (bands["band_start"].to_numpy() + bands["band_end"].to_numpy()) / 2.0
        ax.bar(centers, bands["score"].to_numpy(), width=np.diff(bands[["band_start", "band_end"]], axis=1).flatten(), align="center")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Band Score")
    ax.set_title("Band Scan")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


__all__ = ["plot_roc_curve", "plot_pr_curve", "plot_band_scan"]
