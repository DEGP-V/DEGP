"""Experiment 2 (Scalability Test), Phase-B instruments.

Arms: standard (plain MLP), residual_only (g = z + f, calibrated small-init),
degp_full (residual + lambda_2 Hutchinson penalty, Eq. 2).
Teacher: z' = A z + B a with A = blockdiag(R(theta)) -- orthogonal, so the
teacher's SPECTRAL eps = 2 sin(theta/2) is K-independent while its FROBENIUS
eps ~ theta*sqrt(2K) grows: the Remark-2 sandwich, live, across K.
All kappa measured through the predictor's OWN rollouts (D_plan semantics),
float64, full singular spectra of J_g - I logged, eps on D_train-like AND
D_plan samples (G5 coverage gap), action chain (sigma_min(J_A), kappa^+)."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from degp.planner import AsymmetricQuasiMetric
from degp.leash import weight_normalize_, quasi_local_hessian
from degp.action_diagnostics import kappa_effective


def jac(fn, x):
    try:
        return torch.autograd.functional.jacobian(fn, x, vectorize=True)
    except Exception:
        return torch.autograd.functional.jacobian(fn, x)


def build_teacher(K, Na, theta, seed):
    A = torch.eye(K)
    c, s = math.cos(theta), math.sin(theta)
    for i in range(K // 2):
        A[2*i:2*i+2, 2*i:2*i+2] = torch.tensor([[c, -s], [s, c]])
    B = torch.linalg.qr(torch.randn(K, Na,
        generator=torch.Generator().manual_seed(seed)))[0]
    return A, B


class PlainMLP(nn.Module):
    def __init__(self, K, Na, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(K + Na, hidden), nn.ReLU(),
                                 nn.Linear(hidden, K))
    def forward(self, z, a): return self.net(torch.cat([z, a], -1))


class ResidualMLP(nn.Module):
    def __init__(self, K, Na, hidden, init_scale):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(K + Na, hidden), nn.ReLU(),
                                 nn.Linear(hidden, K))
        with torch.no_grad():
            self.net[-1].weight.mul_(init_scale); self.net[-1].bias.mul_(init_scale)
    def forward(self, z, a): return z + self.net(torch.cat([z, a], -1))


def build_arm(kind, K, Na, seed, target_init_norm=0.02):
    torch.manual_seed(seed + 5)
    hidden = 2 * K
    if kind == 'standard':
        m = PlainMLP(K, Na, hidden); m.is_deg = False; return m
    m = ResidualMLP(K, Na, hidden, init_scale=1e-3)
    m.is_deg = (kind == 'degp_full')
    with torch.no_grad():                       # calibrate ||J_f||_F to target at all K
        z = torch.randn(4, K); a = torch.randn(4, Na) * 0.5
        nrms = [ (jac(lambda zz: m(zz.unsqueeze(0), a[i].unsqueeze(0)).squeeze(0),
                      z[i]) - torch.eye(K)).norm().item() for i in range(4) ]
        nrm = np.mean(nrms)
        if nrm > 1e-12: m.net[-1].weight.mul_(target_init_norm / nrm)
    return m


def train_arm(model, A, B, pr, seed):
    torch.manual_seed(seed + 777)
    opt = torch.optim.Adam(model.parameters(), lr=pr['lr'])
    gen = torch.Generator().manual_seed(seed + 999)
    K, Na = A.shape[0], B.shape[1]
    for _ in range(pr['train_steps']):
        z = torch.randn(pr['batch'], K, generator=gen)
        a = torch.randn(pr['batch'], Na, generator=gen) * 0.5
        loss = F.mse_loss(model(z, a), z @ A.t() + a @ B.t())
        if model.is_deg:                        # lambda_2 Hutchinson term (Eq. 2)
            v = torch.randn(pr['batch'], K, generator=gen)
            z_ = z.detach().requires_grad_(True)
            (vjp,) = torch.autograd.grad(model(z_, a), z_,
                                         grad_outputs=v, create_graph=True)
            loss = loss + pr['lam2'] * ((vjp - v) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def rollout_states(model, z0, actions):
    z = z0; states = [z]
    for t in range(actions.shape[0]):
        z = model(z.unsqueeze(0), actions[t].unsqueeze(0)).squeeze(0)
        states.append(z)
    return torch.stack(states)


def safe_kappa_from_sv(sv, cap=1e250):
    return min((sv.max() / sv.min().clamp_min(1e-300)).item() ** 2, cap)


def measure(model, K, Na, T, seed, pr):
    g = torch.Generator().manual_seed(seed + 4242)
    z0 = torch.randn(K, generator=g)
    acts = torch.randn(T, Na, generator=g) * 0.5
    states = rollout_states(model, z0, acts)               # (T+1, K)
    Jphi = jac(lambda z: rollout_states(model, z, acts)[-1], z0)
    try: kap_state = safe_kappa_from_sv(torch.linalg.svdvals(Jphi))
    except Exception: kap_state = 1e250
    JA = jac(lambda af: rollout_states(model, z0, af.view(T, Na))[-1], acts.flatten())
    svA = torch.linalg.svdvals(JA)
    cA = svA.min().item()
    kap_a = kappa_effective(JA.t() @ (2.0 * torch.eye(K)) @ JA)
    # step-Jacobian spectra: D_train-like vs D_plan (G5 coverage instrument)
    idx = torch.randperm(T, generator=g)[:8]
    plan_pts = [(states[t], acts[t]) for t in idx]
    train_pts = [(torch.randn(K, generator=g), torch.randn(Na, generator=g) * 0.5)
                 for _ in range(8)]
    def eps_stats(pts):
        eF, espec, ers, spec0 = [], [], [], None
        for i, (z, a) in enumerate(pts):
            Jg = jac(lambda zz: model(zz.unsqueeze(0), a.unsqueeze(0)).squeeze(0), z)
            E = Jg - torch.eye(K)
            s = torch.linalg.svdvals(E)
            eF.append(E.norm().item()); espec.append(s.max().item())
            ers.append(((s.sum() ** 2) / (s ** 2).sum()).item())
            if i == 0: spec0 = s.sort(descending=True).values.numpy()
        return float(np.mean(eF)), float(np.max(espec)), float(np.median(ers)), spec0
    eF_tr, es_tr, _, _ = eps_stats(train_pts)
    eF_pl, es_pl, er_pl, spec0 = eps_stats(plan_pts)
    # leashed quasi-metric (Prop 1): median kappa over 8 valid Delta samples
    dq = AsymmetricQuasiMetric(K, max(1, K // 2)); weight_normalize_(dq, pr['C_W'])
    kqs, tries = [], 0
    while len(kqs) < 8 and tries < 100:
        tries += 1
        dl = torch.randn(K, generator=g)
        if (dl @ dq.w1.weight.t()).abs().min() < 1e-3 or \
           (dl @ dq.w2.weight.t()).abs().min() < 1e-3: continue
        H = Jphi.t() @ quasi_local_hessian(dq, dl, pr['mu']) @ Jphi
        ev = torch.linalg.eigvalsh(H)
        pos = ev[ev > ev.max().item() * 1e-10]
        if len(pos): kqs.append((ev.max() / pos.min()).item())
    return dict(kap_state=kap_state, kap_q=float(np.median(kqs)) if kqs else float('nan'),
                kap_a=kap_a, cA=cA, epsF_train=eF_tr, epsF_plan=eF_pl,
                eps_spec_train=es_tr, eps_spec_plan=es_pl, effrank_plan=er_pl,
                drift=float(states[-1].norm().item()), spec_plan_seed0=spec0)
