"""Agent that monitors feature drift."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import entropy  # type: ignore


@dataclass
class DriftResult:
    psi: dict[str, float]
    kl_divergence: dict[str, float]
    rpm_shift: float | None


def _population_stability_index(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    lower = min(ref.min(), cur.min())
    upper = max(ref.max(), cur.max())
    if lower == upper:
        return 0.0
    edges = np.linspace(lower, upper, bins + 1)
    hist_ref, _ = np.histogram(ref, bins=edges, density=True)
    hist_cur, _ = np.histogram(cur, bins=edges, density=True)
    hist_ref = np.where(hist_ref == 0, 1e-6, hist_ref)
    hist_cur = np.where(hist_cur == 0, 1e-6, hist_cur)
    psi = np.sum((hist_cur - hist_ref) * np.log(hist_cur / hist_ref))
    return float(abs(psi))


class DriftMonitor:
    def assess(
        self, reference: pd.DataFrame, current: pd.DataFrame, features: list[str]
    ) -> DriftResult:
        psi: dict[str, float] = {}
        kl: dict[str, float] = {}
        for feature in features:
            ref_vals = reference[feature].to_numpy(dtype=float)
            cur_vals = current[feature].to_numpy(dtype=float)
            if ref_vals.size < 5 or cur_vals.size < 5:
                continue
            psi[feature] = _population_stability_index(ref_vals, cur_vals)
            hist_ref, _ = np.histogram(ref_vals, bins=20, density=True)
            hist_cur, _ = np.histogram(cur_vals, bins=20, density=True)
            hist_ref = np.where(hist_ref == 0, 1e-6, hist_ref)
            hist_cur = np.where(hist_cur == 0, 1e-6, hist_cur)
            kl[feature] = float(entropy(hist_ref, hist_cur))
        rpm_shift = None
        if {"rpm"}.issubset(reference.columns) and {"rpm"}.issubset(current.columns):
            rpm_shift = float(abs(reference["rpm"].median() - current["rpm"].median()))
        return DriftResult(psi=psi, kl_divergence=kl, rpm_shift=rpm_shift)


__all__ = ["DriftMonitor", "DriftResult"]
