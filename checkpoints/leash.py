"""v1.2 math: the mu-leashed quasi-metric cost (paper Eq. leash), weight
normalization (P3), the blind-cone tripwire (paper Sec. 6), the analytic
local Hessian (Prop 1), and the EXACT zero-set predicate (Lemma 1,
corrected -- see test_v12_leash.py: cone branch + equality branch)."""
from __future__ import annotations
import torch
from degp.planner import AsymmetricQuasiMetric


def leashed_planning_loss(dq, z_T, z_goal, mu: float):
    """L_plan = d_q(z_T, z_goal) + mu ||z_T - z_goal||^2  (mu=0 -> unleashed)."""
    delta = z_T - z_goal
    return dq(z_T, z_goal) + mu * (delta * delta).sum(dim=-1)


@torch.no_grad()
def weight_normalize_(dq: AsymmetricQuasiMetric, C_W: float):
    """Project W1, W2 onto {||W||_2 <= C_W}  (premise P3), in place."""
    for lin in (dq.w1, dq.w2):
        s = torch.linalg.matrix_norm(lin.weight, ord=2)
        if s > C_W:
            lin.weight.mul_(C_W / s)


@torch.no_grad()
def blind_cone_tripwire(dq, z_T, z_goal, dq_thresh=1e-9, delta_thresh=0.1):
    """Alarm iff d_q ~ 0 while ||Delta|| > tau (both spurious zero-set branches)."""
    dq_val = dq(z_T, z_goal)
    dist = (z_T - z_goal).norm(dim=-1)
    return (dq_val <= dq_thresh) & (dist > delta_thresh), dq_val, dist


@torch.no_grad()
def dq_zero_set_predicate(dq, delta, tol=1e-7):
    """Exact: d_q(delta)=0 iff per i: [a_i<=0 & b_i<=0] OR [a_i=b_i>=0]."""
    a = delta @ dq.w1.weight.t(); b = delta @ dq.w2.weight.t()
    cone_i = (a <= tol) & (b <= tol)
    eq_i = ((a - b).abs() <= 2 * tol) & (a > -2 * tol)
    ok_i = cone_i | eq_i
    return ok_i.all(dim=-1), ok_i


@torch.no_grad()
def quasi_local_hessian(dq, delta, mu=0.0):
    """H_loc = (2/M) sum_i v_i v_i^T + 2 mu I,  v_i = d1_i W_{1,i} - d2_i W_{2,i}
    (paper Prop 1; valid away from activation boundaries)."""
    assert delta.dim() == 1
    W1, W2 = dq.w1.weight, dq.w2.weight
    a = delta @ W1.t(); b = delta @ W2.t()
    d1 = (a > 0).to(W1.dtype); d2 = (b > 0).to(W2.dtype)
    v = d1.unsqueeze(1) * W1 - d2.unsqueeze(1) * W2          # (M, K)
    H = (2.0 / v.shape[0]) * (v.t() @ v)
    if mu > 0:
        H = H + 2.0 * mu * torch.eye(delta.shape[0], dtype=H.dtype)
    return H
