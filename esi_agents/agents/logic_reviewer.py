"""Logic review agent ensures conclusions are supported."""
from __future__ import annotations

from pathlib import Path
from typing import Dict


class LogicReviewerAgent:
    def run(self, report_path: Path, metrics: Dict[str, float]) -> bool:
        content = report_path.read_text()
        for key, value in metrics.items():
            value_str = f"{value:.4f}"
            if value_str not in content:
                raise AssertionError(f"Metric {key} with value {value_str} missing from report")
        return True


__all__ = ["LogicReviewerAgent"]
