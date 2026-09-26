"""M2': Experiment 1 (Modularity) on MimicGen, DEGP arms. Sits on the
LeJEPA Phase-3 pipeline; the 'standard' column is the companion paper's
joint training, verbatim. Adds: Eq. 1 encoder loss; decoupled predictor
training (Remark 1: stop-grad BOTH endpoints); residual action predictor
with calibrated small-init; conditioning instruments through the
predictor's own rollouts on encoded real frames."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def entropic_encoder_loss(z, gamma=1.0, lambda_cov=1.0):
    zc = z - z.mean(dim=0, keepdim=True).detach()          # hard centering
    var = zc.var(dim=0, unbiased=False)
    var_loss = (((var - gamma) / gamma) ** 2).mean()
    K = z.shape[1]
    cov = (zc.T @ zc) / z.shape[0]
    cov2 = cov ** 2
    cov_loss = (cov2.sum() - torch.diagonal(cov2).sum()) / (K * (K - 1) * gamma ** 2)
    return var_loss + lambda_cov * cov_loss


class ResidualActionPredictor(nn.Module):
    def __init__(self, K, Na, hidden=256, layers=3):
        super().__init__()
        mods, d = [], K + Na
        for _ in range(layers - 1):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
            d = hidden
        mods += [nn.Linear(d, K)]
        self.net = nn.Sequential(*mods)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01); self.net[-1].bias.mul_(0.01)

    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], dim=-1))

    def calibrate(self, K, Na, target=0.02):
        dt = next(self.parameters()).dtype
        with torch.no_grad():
            z0 = torch.zeros(K, dtype=dt)
            Jf = torch.autograd.functional.jacobian(
                lambda zz: self(zz.unsqueeze(0), torch.zeros(1, Na, dtype=dt)
                                ).squeeze(0) - zz, z0, vectorize=True)
            n = Jf.norm().item()
            if n > 1e-12:
                self.net[-1].weight.mul_(target / n); self.net[-1].bias.mul_(target / n)
        return self


def hutchinson_term(pred, z_in, a, v):
    z_ = z_in.detach().requires_grad_(True)
    out = pred(z_, a)
    (vjp,) = torch.autograd.grad(out, z_, grad_outputs=v, create_graph=True)
    return ((vjp - v) ** 2).sum(-1).mean()


def degp_train(encoder, pred, loader, epochs, steps, lr_enc, lr_pred,
               lam2=0.1, gamma=1.0, joint=True, dev='cuda'):
    """joint=False: encoder frozen (Corollary-1 cell). joint=True: encoder
    continues on Eq. 1 ONLY (representation drift, never gradient
    interference -- Theorem 1 holds in both cells by construction)."""
    params = [{'params': pred.parameters(), 'lr': lr_pred}]
    if joint:
        params.append({'params': encoder.parameters(), 'lr': lr_enc})
    opt = torch.optim.AdamW(params, weight_decay=1e-5)
    encoder.train(); pred.train()
    for ep in range(epochs):
        for i, b in enumerate(loader):
            if i >= steps: break
            img_t, p_t = b['img_t'].to(dev), b['prop_t'].to(dev)
            img_n, p_n = b['img_next'].to(dev), b['prop_next'].to(dev)
            ba = b['action'].to(dev)
            z_t = encoder(img_t, p_t)
            with torch.no_grad():
                z_nx = encoder(img_n, p_n)          # target-side isolation
            z_in = z_t.detach()                      # input-side isolation
            z_hat = pred(z_in, ba)
            l_pred = F.mse_loss(z_hat, z_nx) / gamma \
                + lam2 * hutchinson_term(pred, z_in, ba, torch.randn_like(z_in))
            loss = l_pred + (entropic_encoder_loss(z_t, gamma) if joint
                             else torch.tensor(0.0, device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
    encoder.eval(); pred.eval()
    return encoder, pred


def standard_train(encoder, pred, dec, sigreg, loader, epochs, steps, dev,
                   sig_w=20.0, pred_w=1.0, rec_w=1.0, freeze_enc=False):
    """Companion paper's 'no_physics' arm, parameterized by loader."""
    params = list(pred.parameters()) + list(dec.parameters()) + list(sigreg.parameters())
    if not freeze_enc:
        params += list(encoder.parameters())
    opt = torch.optim.AdamW(params, weight_decay=1e-5)  # lrs via param groups omitted
    encoder.train(); pred.train(); dec.train()
    for ep in range(epochs):
        for i, b in enumerate(loader):
            if i >= steps: break
            z_t = encoder(b['img_t'].to(dev), b['prop_t'].to(dev))
            z_nx = encoder(b['img_next'].to(dev), b['prop_next'].to(dev))
            z_pv = encoder(b['img_prev'].to(dev), b['prop_prev'].to(dev))
            ba = b['action'].to(dev)
            l_pred = F.mse_loss(pred(z_t, ba), z_nx)
            l_sig = sigreg(torch.cat([z_pv, z_t, z_nx], 0))
            l_rec = (F.mse_loss(dec(z_t), b['prop_t'].to(dev))
                     + F.mse_loss(dec(z_nx), b['prop_next'].to(dev))
                     + F.mse_loss(dec(z_pv), b['prop_prev'].to(dev))) / 3
            loss = pred_w * l_pred + sig_w * l_sig + rec_w * l_rec
            opt.zero_grad(); loss.backward(); opt.step()
    for m in (encoder, pred, dec): m.eval()
    return encoder, pred, dec


def lens_shift(img_u8, rng):
    """The validated unseen shift: vignette + fixed blob + blur + PER-SAMPLE
    noise (per-sample is load-bearing; batch-shared noise is removed
    exactly by mean-centering)."""
    x = torch.from_numpy(img_u8.astype(np.float32) / 255.0)
    x = x.permute(0, 3, 1, 2)
    n, c, h, w = x.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w),
                            indexing='ij')
    rad = (yy ** 2 + xx ** 2).sqrt().clamp(max=1.4)
    vig = (1 - 0.6 * rad / rad.max()).clamp(0.1, 1.0)
    x = x * vig
    blob = torch.exp(-((yy - 0.3) ** 2 + (xx + 0.2) ** 2) / (2 * 0.35 ** 2))
    x = (x + 0.5 * blob).clamp(0, 1)
    x = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(x, (4, 4, 4, 4),
        mode='replicate'), 9, stride=1)
    noise = torch.randn(n, 1, h, w, generator=rng).repeat(1, 3, 1, 1) * 0.10
    return (x + noise).clamp(0, 1)


def mc_cos(Za, Zb):
    Za = Za - Za.mean(0, keepdims=True); Zb = Zb - Zb.mean(0, keepdims=True)
    num = (Za * Zb).sum(1)
    return float((num / (np.linalg.norm(Za, axis=1)
                         * np.linalg.norm(Zb, axis=1) + 1e-12)).mean())


@torch.no_grad()
def rollout_conditioning(encoder, pred, demos, demo_start, acts, T, n_starts,
                         K, dev, seed=0):
    """kappa/eps through the predictor's OWN rollouts from encoded frames."""
    rng = np.random.RandomState(seed)
    cand = [s for d in demos for s in range(demo_start[d], demo_start[d + 1] - T - 1)]
    starts = rng.choice(cand, min(n_starts, len(cand)), replace=False)
    Z0 = torch.from_numpy(encoder_encode(encoder, starts, dev)).float().to(dev)
    I = torch.eye(K, device=dev)
    eps_all, smin, smax = [], [], []
    for si in range(len(starts)):
        z = Z0[si].clone(); J = I.clone()
        for t in range(T):
            a = torch.from_numpy(acts[starts[si] + t]).float().to(dev)
            Jg = torch.autograd.functional.jacobian(
                lambda zz: pred(zz.unsqueeze(0), a.unsqueeze(0)).squeeze(0),
                z.detach(), vectorize=True)
            E = Jg - I
            eps_all.append(torch.linalg.svdvals(E).max().item())
            with torch.no_grad():
                z = pred(z.detach().unsqueeze(0), a.unsqueeze(0)).squeeze(0)
            J = Jg @ J
        s = torch.linalg.svdvals(J)
        smin.append(s.min().item()); smax.append(s.max().item())
    kap = (max(smax) / max(min(smin), 1e-300)) ** 2
    return kap, float(np.mean(eps_all))


def encoder_encode(encoder, frame_idx, dev, img_arr=None, prop_n=None):
    """Encode frames via the LeJEPA encode_frames when available."""
    return encode_frames(encoder, np.asarray(frame_idx))
