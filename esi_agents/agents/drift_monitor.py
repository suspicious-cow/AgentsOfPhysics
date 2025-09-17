"""Drift monitoring agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import entropy


@dataclass
class DriftReport:
    metrics: pd.DataFrame


class DriftMonitorAgent:
    def run(self, reference: pd.DataFrame, current: pd.DataFrame) -> DriftReport:
        rows = []
        for column in reference.columns:
            ref_values = reference[column].to_numpy()
            cur_values = current[column].to_numpy()
            psi = self._population_stability_index(ref_values, cur_values)
            kl = self._kl_divergence(ref_values, cur_values)
            rows.append({"feature": column, "psi": psi, "kl_divergence": kl})
        metrics = pd.DataFrame(rows)
        return DriftReport(metrics=metrics)

    def _population_stability_index(self, ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
        if len(ref) == 0 or len(cur) == 0:
            return 0.0
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(ref, quantiles))
        if len(edges) < 2:
            edges = np.linspace(ref.min(), ref.max() + 1e-6, bins + 1)
        ref_counts, _ = np.histogram(ref, bins=edges)
        cur_counts, _ = np.histogram(cur, bins=edges)
        ref_dist = ref_counts / (ref_counts.sum() or 1)
        cur_dist = cur_counts / (cur_counts.sum() or 1)
        ref_dist = np.where(ref_dist == 0, 1e-6, ref_dist)
        cur_dist = np.where(cur_dist == 0, 1e-6, cur_dist)
        return float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))

    def _kl_divergence(self, ref: np.ndarray, cur: np.ndarray, bins: int = 20) -> float:
        if len(ref) == 0 or len(cur) == 0:
            return 0.0
        min_value = min(ref.min(), cur.min())
        max_value = max(ref.max(), cur.max())
        edges = np.linspace(min_value, max_value, bins)
        ref_hist, _ = np.histogram(ref, bins=edges, density=True)
        cur_hist, _ = np.histogram(cur, bins=edges, density=True)
        ref_hist = np.where(ref_hist == 0, 1e-6, ref_hist)
        cur_hist = np.where(cur_hist == 0, 1e-6, cur_hist)
        return float(entropy(ref_hist, cur_hist))


__all__ = ["DriftMonitorAgent", "DriftReport"]
