"""Batch scoring agent."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


@dataclass
class BatchScorerResult:
    scores: pd.DataFrame
    path: Path


class BatchScorerAgent:
    def run(
        self,
        model,
        feature_frame: pd.DataFrame,
        feature_columns: List[str],
        output_dir: Path,
    ) -> BatchScorerResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        X = feature_frame[feature_columns].fillna(0.0)
        scores = model.score_samples(X)
        contributions = self._compute_contributions(X)
        score_frame = feature_frame[["asset_id", "channel", "window_start", "window_end"]].copy()
        score_frame["anomaly_score"] = scores
        score_frame["top_features"] = [", ".join(items) for items in contributions]
        path = output_dir / "scores.parquet"
        try:
            score_frame.to_parquet(path)
        except ImportError:  # pragma: no cover - optional dependency
            path = path.with_suffix(".csv")
            score_frame.to_csv(path, index=False)
        return BatchScorerResult(scores=score_frame, path=path)

    def _compute_contributions(self, X: pd.DataFrame) -> List[List[str]]:
        zscores = np.abs((X - X.mean()) / (X.std() + 1e-6))
        top_features = []
        for _, row in zscores.iterrows():
            ranked = row.sort_values(ascending=False).head(3)
            top_features.append([f"{idx}:{value:.2f}" for idx, value in ranked.items()])
        return top_features


__all__ = ["BatchScorerAgent", "BatchScorerResult"]
