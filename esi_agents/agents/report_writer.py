"""Agent that produces Markdown reports."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_ingestor import DataQualitySummary
from .evaluator import EvaluationArtifacts
from .feature_engineer import FeatureResult
from .model_selector import SelectionResult
from .drift_monitor import DriftResult


class ReportWriter:
    def write(
        self,
        config: dict[str, Any],
        quality: DataQualitySummary,
        features: FeatureResult,
        selection: SelectionResult,
        evaluation: EvaluationArtifacts,
        drift: DriftResult | None,
        output_path: str | Path,
    ) -> Path:
        best = selection.best_model
        matrix = features.matrix
        numeric = matrix.select_dtypes(include=[float, int])
        correlations = {}
        if not numeric.empty:
            scores = best.scores[: len(numeric)]
            for column in numeric.columns:
                corr = np.corrcoef(numeric[column], scores)[0, 1]
                if np.isnan(corr):
                    continue
                correlations[column] = float(corr)
        top_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        lines = ["# ESI Anomaly Detection Report", ""]
        lines.append("## Configuration")
        lines.append(f"- Adapter: {config.get('adapter', 'csv')}")
        lines.append(f"- Window Size: {config.get('window', {}).get('size', 'n/a')}")
        lines.append(f"- Models: {', '.join(m['name'] for m in config.get('models', []))}")
        lines.append("")
        lines.append("## Data Quality")
        lines.append(f"- Missing values: {quality.missing_values}")
        lines.append(f"- Flatlines detected: {quality.flatlines}")
        lines.append(f"- Gap count: {quality.gap_count}")
        lines.append(f"- Timestamp monotonic: {quality.timestamp_monotonic}")
        lines.append("")
        lines.append("## Model Selection")
        lines.append(f"- Selected model: **{best.name}**")
        lines.append(f"- Metrics: {evaluation.metrics.__dict__}")
        lines.append("")
        if top_features:
            lines.append("### Top correlated features")
            for name, corr in top_features:
                lines.append(f"- {name}: correlation {corr:.3f}")
            lines.append("")
        if evaluation.band_scan_results:
            lines.append("## Band Scan Highlights")
            for band in evaluation.band_scan_results[:5]:
                lines.append(
                    f"- {band.band_start:.2f}–{band.band_end:.2f} Hz/order, z={band.z_score:.2f}, p={band.p_value:.3g}"
                )
            if "band" in evaluation.plots:
                lines.append(f"![Band Scan]({evaluation.plots['band']})")
            lines.append("")
        if drift:
            lines.append("## Drift & Data Quality")
            lines.append(f"- PSI: {drift.psi}")
            lines.append(f"- KL divergence: {drift.kl_divergence}")
            lines.append(f"- RPM shift: {drift.rpm_shift}")
            lines.append("")
        if "roc" in evaluation.plots:
            lines.append(f"![ROC Curve]({evaluation.plots['roc']})")
        if "pr" in evaluation.plots:
            lines.append(f"![PR Curve]({evaluation.plots['pr']})")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines), encoding="utf-8")
        return output


__all__ = ["ReportWriter"]
