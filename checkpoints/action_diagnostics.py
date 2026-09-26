"""v1.2 math: unrolled action Jacobian J_A (Prop 3), composite controllability
sigma_min(J_A), the effective (nonzero-spectrum) condition number, the Prop 3
bound, and the lambda_3 probe.

Engineering notes verified by the test run:
1. (v1.2.2) The lambda_3 probe computes J_a EXACTLY via a reverse-mode
   jacobian (one (K, N_a) matrix per transition) and Monte-Carlo's the probe
   directions vectorized. The earlier forward-mode (dual-number) JVP silently
   dropped the tangent through matmul on some torch builds ('Tensor' object
   has no attribute 'tangent'); this version is version-proof and faster.
2. kappa_effective: H_a = J_A^T H_loc J_A is singular whenever T*N_a > K
   (null space = action variations that leave z_T unchanged). Those
   directions carry exactly zero gradient and do not slow cost convergence;
   the conditioning that governs descent is the NONZERO spectrum
   (kappa^+ = lambda_max / lambda_min^+), or H_a + rho*I under the standard
   MPC trust-region term rho*||a||^2.
Systems-theory fact (verified by tests): for linear g(z,a)=Az+Ba, J_A is the
REACHABILITY matrix; a pure integrator (A=I, epsilon=0 exactly) has
sigma_min(J_A)=0 whenever N_a < K -- cross-time cancellation. Per-step
sensitivity (the lambda_3 probe target) is necessary, NOT sufficient; hence
direct spectral logging of J_A (Assumption 1)."""
from __future__ import annotations
import torch
from torch import nn


class LinearPredictor(nn.Module):
    """g(z,a) = A z + B a with exact Jacobians J_g=A, J_a=B."""
    def __init__(self, A, B):
        super().__init__()
        self.A, self.B = nn.Parameter(A.clone()), nn.Parameter(B.clone())
    def forward(self, z, a):
        return z @ self.A.t() + a @ self.B.t()


def rollout_T(predictor, z0, actions):
    z = z0
    for t in range(actions.shape[0]):
        z = predictor(z.unsqueeze(0), actions[t].unsqueeze(0)).squeeze(0)
    return z


def unrolled_action_jacobian(predictor, z0, actions):
    """J_A = d z_T / d a_{1:T}: (K, T*N_a), exact."""
    return torch.autograd.functional.jacobian(
        lambda af: rollout_T(predictor, z0, af.view(actions.shape)),
        actions.flatten())


def unrolled_state_jacobian(predictor, z0, actions):
    """J_Phi = d z_T / d z_0: (K, K), exact."""
    return torch.autograd.functional.jacobian(lambda z: rollout_T(predictor, z, actions), z0)


def action_jacobian(backbone, z, a):
    """Exact J_a = d g / d a at (z, a): (K, N_a), via reverse-mode jacobian.
    Handles batched (z: (1,K), a: (1,N_a)) and unbatched calls:
    functional.jacobian appends input dims to output dims, so a batched call
    yields (1, K, 1, N_a) -> jac[0, :, 0, :]."""
    jac = torch.autograd.functional.jacobian(lambda aa: backbone(z, aa), a)
    if jac.dim() == 4:
        return jac[0, :, 0, :]
    if jac.dim() == 3:
        return jac[:, 0, :]
    return jac


def lambda3_probe(backbone, z, a, c0, n_probes=128, gen=None):
    """mean_i max(0, c0 - ||J_a u_i||), u ~ N(0, I): Eq. 2's fourth term,
    evaluated at the transition (z, a). J_a computed exactly once; the probe
    draws are vectorized Monte Carlo over directions u."""
    J = action_jacobian(backbone, z, a).detach()
    us = torch.randn(n_probes, J.shape[1], generator=gen,
                     device=J.device, dtype=J.dtype)
    norms = (us @ J.t()).norm(dim=-1)          # ||J_a u|| per draw
    vals = torch.clamp(c0 - norms, min=0.0)
    return vals.mean().item(), float((norms < c0).float().mean())


@torch.no_grad()
def kappa(M):
    """Plain condition number from singular values (fine for rectangular
    J_A: svdvals returns min(m, n) values)."""
    s = torch.linalg.svdvals(M).clamp_min(1e-300)
    return (s.max() / s.min()).item() ** 2


@torch.no_grad()
def kappa_effective(H, rel_tol=1e-10):
    """kappa^+(H) = lambda_max / lambda_min^+ over POSITIVE eigenvalues of a
    symmetric PSD H. Use for the action Hessian H_a, which is singular
    whenever T*N_a > K (module docstring, note 2)."""
    ev = torch.linalg.eigvalsh(H)
    pos = ev[ev > ev.max().item() * rel_tol]
    return ev.max().item() / pos.min().item()


@torch.no_grad()
def prop3_bound(eps, T, Na, cbar, cA, C_W=None, M=None, mu=None):
    """kappa^+(H_a) <= (1 + 4 C_W^2/(M mu)) * T Na (1+eps)^{2(T-1)} (cbar/cA)^2."""
    f = 1.0 + 4.0 * C_W ** 2 / (M * mu) if None not in (C_W, M, mu) else 1.0
    return f * T * Na * (1.0 + eps) ** (2 * (T - 1)) * (cbar / cA) ** 2
