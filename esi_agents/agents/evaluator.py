"""Evaluator agent for anomaly models."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics

from ..eval import (
    compute_classification_metrics,
    metrics_to_frame,
    plot_band_scan,
    plot_pr_curve,
    plot_roc_curve,
    scan_frequency_bands,
)
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    scores: pd.DataFrame
    band_scan: pd.DataFrame


class EvaluatorAgent:
    def run(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        feature_frame: pd.DataFrame,
        output_dir: Path,
        prefix: str,
    ) -> EvaluationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        scores = model.score_samples(X)
        metrics = compute_classification_metrics(scores, y.to_numpy())
        metrics_df = metrics_to_frame(metrics)

        roc_fpr, roc_tpr, _ = skmetrics.roc_curve(y_true=y, y_score=scores)
        pr_precision, pr_recall, _ = skmetrics.precision_recall_curve(y, scores)

        plot_roc_curve(roc_fpr, roc_tpr, output_dir / f"{prefix}_roc.png")
        plot_pr_curve(pr_recall, pr_precision, output_dir / f"{prefix}_pr.png")

        band_scan_df = scan_frequency_bands(feature_frame)
        plot_band_scan(band_scan_df, output_dir / f"{prefix}_band_scan.png")

        metrics_path = output_dir / f"{prefix}_metrics.json"
        metrics_path.write_text(json.dumps(metrics_df.iloc[0].to_dict(), indent=2))

        window_starts = feature_frame.loc[X.index, "window_start"].values
        score_frame = pd.DataFrame(
            {
                "score": scores,
                "label": y.to_numpy(),
                "window_start": window_starts,
            },
            index=X.index,
        )
        scores_path = output_dir / f"{prefix}_scores.parquet"
        try:
            score_frame.to_parquet(scores_path)
        except ImportError:  # pragma: no cover - optional dependency
            csv_path = scores_path.with_suffix(".csv")
            score_frame.to_csv(csv_path, index=False)
            scores_path = csv_path

        return EvaluationResult(metrics=metrics_df.iloc[0].to_dict(), scores=score_frame, band_scan=band_scan_df)


__all__ = ["EvaluatorAgent", "EvaluationResult"]
