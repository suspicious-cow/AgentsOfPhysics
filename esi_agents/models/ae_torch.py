"""Optional PyTorch autoencoder for anomaly detection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import CalibrationModel

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[misc]
    nn = None  # type: ignore[misc]


@dataclass
class AutoencoderDetector:
    input_dim: int
    hidden_dim: int = 32
    latent_dim: int = 8
    lr: float = 1e-3
    epochs: int = 50
    calibrator: CalibrationModel | None = None

    def __post_init__(self) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not available; install optional extra 'torch'")
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AutoencoderDetector":
        data = torch.tensor(X, dtype=torch.float32)
        for _ in range(self.epochs):
            self.optim.zero_grad()
            recon = self.model(data)
            loss = self.loss_fn(recon, data)
            loss.backward()
            self.optim.step()
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        data = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            recon = self.model(data)
            residual = torch.mean((data - recon) ** 2, dim=1)
        scores = residual.numpy()
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        if self.calibrator:
            return self.calibrator.transform(scores)
        return scores


__all__ = ["AutoencoderDetector"]
