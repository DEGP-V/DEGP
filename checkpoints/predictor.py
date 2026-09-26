"""Residual predictor backbone (reference configuration satisfying
Theorem 2's premise by construction). Minimal API-identical copy."""
from __future__ import annotations
import torch
from torch import nn


class ResidualPredictorBackbone(nn.Module):
    """g(z, a) = z + f(z, a), f's final layer small-initialized:
    J_g = I + J_f starts near the identity BY CONSTRUCTION."""

    def __init__(self, latent_dim: int, action_dim: int, init_scale: float = 0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(init_scale)
            self.net[-1].bias.mul_(init_scale)

    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], dim=-1))
