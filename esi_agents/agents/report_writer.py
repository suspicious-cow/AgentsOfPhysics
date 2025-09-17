"""Report writer agent."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ReportContext:
    config: Dict
    dq_report: Dict[str, float]
    metrics: Dict[str, float]
    drift: pd.DataFrame
    band_scan: pd.DataFrame


class ReportWriterAgent:
    def run(self, context: ReportContext, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "report.md"
        band_lines = (
            context.band_scan.head(5).to_string(index=False)
            if not context.band_scan.empty
            else "No significant bands"
        )
        drift_lines = (
            context.drift.to_string(index=False)
            if not context.drift.empty
            else "No drift detected"
        )
        lines = [
            "# ESI Anomaly Detection Report",
            "",
            "## Configuration",
            f"Data source: `{context.config['data']['source']['type']}`",
            "",
            "## Data Quality",
            "",
            "| Metric | Value |",
            "| --- | --- |",
        ]
        for key, value in context.dq_report.items():
            lines.append(f"| {key} | {value:.4f} |")
        lines.extend(
            [
                "",
                "## Selected Model Performance",
                "",
                "| Metric | Value |",
                "| --- | --- |",
            ]
        )
        for key, value in context.metrics.items():
            lines.append(f"| {key} | {value:.4f} |")
        lines.extend(
            [
                "",
                "## Drift Assessment",
                "",
                drift_lines,
                "",
                "## Band Scan Highlights",
                "",
                band_lines,
            ]
        )
        report_path.write_text("\n".join(lines))
        LOGGER.info("Report written to %s", report_path)
        return report_path


__all__ = ["ReportWriterAgent", "ReportContext"]
