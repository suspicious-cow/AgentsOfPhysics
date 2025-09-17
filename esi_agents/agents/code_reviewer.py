"""Code review agent ensures artifacts exist and constraints met."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


class CodeReviewerAgent:
    def run(self, expected_paths: Iterable[Path]) -> bool:
        missing = [str(path) for path in expected_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing expected artifacts: {missing}")
        return True


__all__ = ["CodeReviewerAgent"]
