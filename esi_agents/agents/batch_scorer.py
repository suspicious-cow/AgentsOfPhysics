"""Agent that performs batch scoring."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import pandas as pd

from .model_selector import SelectionResult


class BatchScorer:
    def score(
        self,
        selection: SelectionResult,
        features: pd.DataFrame,
        output_path: str | Path,
        threshold: float = 0.9,
    ) -> tuple[pd.DataFrame, Path]:
        numeric_cols = features.select_dtypes(include=[float, int]).columns
        X = features[numeric_cols].to_numpy(dtype=float)
        scores = selection.best_model.model.score_samples(X)
        result = features.copy()
        result["anomaly_score"] = scores
        result["alert"] = result["anomaly_score"] >= threshold
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            try:
                result.to_parquet(path, index=False)
            except ImportError:
                path = path.with_suffix(".csv")
                result.to_csv(path, index=False)
        else:
            result.to_csv(path, index=False)
        return result, path


__all__ = ["BatchScorer"]
