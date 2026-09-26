"""Asymmetric Quasi-Metric planning cost (paper Section 3.3, Eq. 3).
Minimal API-identical copy so the v1.2 notebook is self-contained in a
fresh session; if the full package exists, interfaces are the same."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricQuasiMetric(nn.Module):
    """d_q(z_A, z_B) via two distinct linear projections W1 != W2 (Eq. 3)."""

    def __init__(self, latent_dim: int, proj_dim: int):
        super().__init__()
        self.w1 = nn.Linear(latent_dim, proj_dim, bias=False)
        self.w2 = nn.Linear(latent_dim, proj_dim, bias=False)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        diff1 = self.w1(z_a) - self.w1(z_b)
        diff2 = self.w2(z_a) - self.w2(z_b)
        term = F.relu(diff1) - F.relu(diff2)
        return (term ** 2).mean(dim=-1)

    def gradient_wrt_a(self, z_a, z_b):
        z_a = z_a.detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(self.forward(z_a, z_b).sum(), z_a)
        return grad
