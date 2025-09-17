"""Agent that performs consistency checks on reports."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluator import EvaluationArtifacts


@dataclass
class LogicReview:
    approved: bool
    issues: list[str]


class LogicReviewer:
    def review(self, evaluation: EvaluationArtifacts, report_path: str | Path) -> LogicReview:
        text = Path(report_path).read_text(encoding="utf-8")
        issues: list[str] = []
        for key, value in evaluation.metrics.__dict__.items():
            if value is None:
                continue
            formatted = f"{value:.3f}" if isinstance(value, float) else str(value)
            if formatted not in text:
                issues.append(f"Metric {key}={formatted} missing from report")
        if evaluation.band_scan_results and "Band Scan" not in text:
            issues.append("Band scan results missing from report")
        approved = not issues
        return LogicReview(approved=approved, issues=issues)


__all__ = ["LogicReviewer", "LogicReview"]
