"""Agent that evaluates a trained model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np

from ..eval import (
    MetricsResult,
    band_scan,
    compute_classification_metrics,
    plot_band_scan,
    plot_pr_curve,
    plot_roc_curve,
)
from .feature_engineer import FeatureResult
from .model_selector import SelectionResult


@dataclass
class EvaluationArtifacts:
    metrics: MetricsResult
    band_scan_results: list[Any]
    plots: dict[str, Path]


class Evaluator:
    def evaluate(
        self,
        selection: SelectionResult,
        features: FeatureResult,
        labels: np.ndarray | None,
        output_dir: str | Path,
    ) -> EvaluationArtifacts:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        best = selection.best_model
        metrics = compute_classification_metrics(labels, best.scores)
        (output / "metrics.json").write_text(metrics.to_json(), encoding="utf-8")
        plots: dict[str, Path] = {}
        if labels is not None and len(np.unique(labels)) > 1:
            roc_path = output / "roc_curve.png"
            pr_path = output / "pr_curve.png"
            plot_roc_curve(labels, best.scores, roc_path)
            plot_pr_curve(labels, best.scores, pr_path)
            plots["roc"] = roc_path
            plots["pr"] = pr_path
        freq_column = None
        for col in ["freq_peak_1", "freq_centroid"]:
            if col in features.matrix.columns:
                freq_column = col
                break
        if freq_column:
            freqs = features.matrix[freq_column].to_numpy(dtype=float)
            magnitudes = best.scores[: len(freqs)]
            order = np.argsort(freqs)
            freqs = freqs[order]
            magnitudes = magnitudes[order]
            bands = band_scan(freqs, magnitudes)
            band_path = output / "band_scan.png"
            plot_band_scan(bands, band_path)
            plots["band"] = band_path
            bands_json = [band.__dict__ for band in bands]
            (output / "band_scan.json").write_text(json.dumps(bands_json, indent=2), encoding="utf-8")
        else:
            bands = []
        return EvaluationArtifacts(metrics=metrics, band_scan_results=bands, plots=plots)


__all__ = ["Evaluator", "EvaluationArtifacts"]
