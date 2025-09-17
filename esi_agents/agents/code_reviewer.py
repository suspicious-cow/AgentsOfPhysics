"""Agent that checks artifact availability before promotion."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class CodeReview:
    approved: bool
    issues: list[str]


class CodeReviewer:
    def review(self, artifacts: Iterable[str | Path]) -> CodeReview:
        issues: list[str] = []
        for item in artifacts:
            path = Path(item)
            if not path.exists():
                issues.append(f"Missing artifact: {path}")
            elif path.is_file() and path.stat().st_size == 0:
                issues.append(f"Empty artifact: {path}")
        approved = not issues
        return CodeReview(approved=approved, issues=issues)


__all__ = ["CodeReviewer", "CodeReview"]
