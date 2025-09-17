"""Feature engineering agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from ..features import (
    compute_envelope_features,
    compute_frequency_features,
    compute_order_features,
    compute_time_domain_features,
    iter_windows,
)
from ..schema import SchemaRegistry


@dataclass
class FeatureEngineerResult:
    features: pd.DataFrame
    feature_columns: List[str]


class FeatureEngineerAgent:
    """Compute sliding-window feature matrices."""

    def run(self, df: pd.DataFrame, schema: SchemaRegistry, config: Dict) -> FeatureEngineerResult:
        feature_cfg = config["features"]
        window_size = int(feature_cfg.get("window_size", 256))
        stride = int(feature_cfg.get("stride", window_size // 2))
        rpm_col = feature_cfg.get("rpm_column", "rpm")

        rows = []
        for window in iter_windows(df, window_size=window_size, stride=stride, rpm_col=rpm_col):
            features = {
                "asset_id": window.asset_id,
                "channel": window.channel,
                "window_start": window.start,
                "window_end": window.end,
            }
            features.update(compute_time_domain_features(window))
            features.update(compute_frequency_features(window))
            features.update(compute_envelope_features(window))
            features.update(compute_order_features(window))

            if "label" in df.columns:
                mask = (
                    (df["asset_id"] == window.asset_id)
                    & (df["channel"] == window.channel)
                    & (df["timestamp"] >= window.start)
                    & (df["timestamp"] <= window.end)
                )
                label = float(df.loc[mask, "label"].max()) if mask.any() else 0.0
                features["label"] = label
            rows.append(features)

        feature_frame = pd.DataFrame(rows)
        feature_frame.sort_values("window_start", inplace=True)
        feature_frame.reset_index(drop=True, inplace=True)
        numeric_cols = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in {"label"}]

        # Fused multivariate features across channels per asset/time
        fused = (
            feature_frame.groupby(["asset_id", "window_start"]).mean(numeric_only=True).add_prefix("fused_")
        )
        fused = fused.reset_index()
        feature_frame = feature_frame.merge(fused, on=["asset_id", "window_start"], how="left")
        fused_cols = [
            col
            for col in fused.columns
            if col not in {"asset_id", "window_start"} and "label" not in col
        ]
        feature_cols.extend([col for col in fused_cols if col not in feature_cols])

        return FeatureEngineerResult(features=feature_frame, feature_columns=feature_cols)


__all__ = ["FeatureEngineerAgent", "FeatureEngineerResult"]
