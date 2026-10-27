"""Probabilistic ensemble of hand-object world models (PETS-style: Chua et al. 2018), stage 1.5.

The deterministic model (model.py) cannot tell contact vibration (unpredictable, zero-mean) from its own
errors. Here E members are trained together (batched matrix multiplies): each has the deterministic mean path
of model.HandObjectWorldModel (k-step autoregressive loss) plus a per-dimension log-variance of the one-step
state change, fit by Gaussian likelihood on the first (true-state) step with PETS's soft bounds. Then
  aleatoric  = predicted variance (vibration and other noise the state cannot explain),
  epistemic  = disagreement between members (where the model does not know),
and the questions for stage 2 are whether the variance is calibrated and whether disagreement predicts the
ensemble's errors (a per-sample gate for model gradients). EnsembleMean wraps the ensemble as a single model
(member-mean states, projected mean rotations, averaged probabilities) for the existing evaluation code.
"""
import math

import torch
from torch import nn

from .data import Layout, Normalizer
from .geometry import matrix_to_rot6, so3_exp, so3_log
from .model import HandObjectWorldModel, STATE_WEIGHTS, one_step_deltas, state_error


class BatchedLinear(nn.Module):
    def __init__(self, members, fan_in, fan_out):
        super().__init__()
        bound = 1/math.sqrt(fan_in)
        self.weight = nn.Parameter(torch.empty(members, fan_in, fan_out).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(members, 1, fan_out).uniform_(-bound, bound))

    def forward(self, x):                      # (E, B, fan_in) -> (E, B, fan_out)
        return torch.baddbmm(self.bias, x, self.weight)


def batched_mlp(members, sizes, last_act=False):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(BatchedLinear(members, a, b))
        if i < len(sizes)-2 or last_act:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class EnsembleWorldModel(nn.Module):
    def __init__(self, layout: Layout, members=5, hidden=(512, 512, 512), context_embed=64, exo_mode='pre'):
        super().__init__()
        self.L = L = layout
        self.E = E = members
        self.exo_mode = exo_mode
        for name, dim in (('state', L.state), ('forces', L.forces), ('action', L.actuated), ('exo', L.exo),
                          ('context', L.context), ('delta', L.delta)):
            setattr(self, 'norm_'+name, Normalizer(dim))
        self.context_net = batched_mlp(E, (L.context, 256, context_embed), last_act=True)
        self.trunk = batched_mlp(E, (L.state+L.forces+L.actuated+L.exo+context_embed,)+tuple(hidden), last_act=True)
        h = hidden[-1]
        self.head_delta = BatchedLinear(E, h, L.delta)
        self.head_logvar = BatchedLinear(E, h, L.delta)
        self.head_forces = BatchedLinear(E, h, L.forces)
        self.head_logits = BatchedLinear(E, h, 10)
        with torch.no_grad():
            for head in (self.head_delta, self.head_forces, self.head_logvar, self.head_logits):
                head.weight.zero_(); head.bias.zero_()
            self.head_logits.bias[..., 9] = -8.
        self.max_logvar = nn.Parameter(torch.full((1, 1, L.delta), .5))
        self.min_logvar = nn.Parameter(torch.full((1, 1, L.delta), -10.))

    fit_normalizers = HandObjectWorldModel.fit_normalizers

    def embed(self, context):
        c = self.norm_context(context)[None].expand(self.E, -1, -1)
        return self.context_net(c)

    def step(self, x, R, f, a, u, z):
        """x (E, B, state), R (E, B, 3, 3), f (E, B, forces); a, u shared (B, .); z (E, B, embed)."""
        E = self.E
        inp = torch.cat((self.norm_state(x), self.norm_forces(f), self.norm_action(a)[None].expand(E, -1, -1),
                         self.norm_exo(u)[None].expand(E, -1, -1), z), -1)
        h = self.trunk(inp)
        d_n = self.head_delta(h)
        logvar = self.max_logvar-nn.functional.softplus(self.max_logvar-self.head_logvar(h))
        logvar = self.min_logvar+nn.functional.softplus(logvar-self.min_logvar)
        d = d_n*self.norm_delta.std+self.norm_delta.mean
        H = self.L.hand
        R_next = R@so3_exp(d[..., 3:6])
        x_next = torch.cat((x[..., :3]+d[..., :3], matrix_to_rot6(R_next), x[..., 9:15]+d[..., 6:12],
                            x[..., 15:15+H]+d[..., 12:12+H], x[..., 15+H:]+d[..., 12+H:]), -1)
        f_next = (f+self.head_forces(h)*self.norm_forces.std).clamp_min(0.)
        return x_next, R_next, f_next, self.head_logits(h), d_n, logvar

    def rollout(self, w, k):
        E = self.E
        x = w['state'][:, 0][None].expand(E, -1, -1)
        R = w['rot'][:, 0][None].expand(E, -1, -1, -1)
        f = w['forces'][:, 0][None].expand(E, -1, -1)
        z = self.embed(w['context'][:, 0])
        xs, Rs, fs, ls, first = [], [], [], [], None
        for j in range(1, k+1):
            x, R, f, logit, d_n, logvar = self.step(x, R, f, w['action'][:, j], w['exo'][:, j], z)
            if j == 1:
                first = (d_n, logvar)
            xs.append(x); Rs.append(R); fs.append(f); ls.append(logit)
        return xs, Rs, fs, ls, first


def true_normalized_delta(model, w):
    """One-step logged change from the window's first state, in the delta normaliser's units (B, delta)."""
    x0, x1, R0, R1 = w['state'][:, 0], w['state'][:, 1], w['rot'][:, 0], w['rot'][:, 1]
    H = model.L.hand
    d = torch.cat((x1[:, :3]-x0[:, :3], so3_log(R0.transpose(-1, -2)@R1), x1[:, 9:15]-x0[:, 9:15],
                   x1[:, 15:15+H]-x0[:, 15:15+H], x1[:, 15+H:]-x0[:, 15+H:]), -1)
    return (d-model.norm_delta.mean)/model.norm_delta.std


def ensemble_window_loss(model, w, k, nll_weight=.1, lost_pos_weight=1., weights=STATE_WEIGHTS, window_weight=None,
                         support_in_secure=True):
    """Mean over members of model.window_loss's terms on their own rollouts, plus the one-step Gaussian NLL of
    the state change (aleatoric variance) and PETS's log-variance bound penalty."""
    xs, Rs, fs, ls, (d_n, logvar) = model.rollout(w, k)
    E, B = model.E, w['state'].shape[0]
    bce = nn.functional.binary_cross_entropy_with_logits
    wt = torch.ones(B, device=w['state'].device) if window_weight is None else window_weight
    total, logs = 0., {}
    n_state = torch.zeros((), device=wt.device)
    flat = lambda v: v.reshape(E*B, *v.shape[2:])
    rep = lambda v: v[None].expand(E, *v.shape).reshape(E*B, *v.shape[1:])
    for j in range(1, k+1):
        has_next, alive = w['has_next'][:, j-1].float()*wt, w['alive'][:, j-1].float()*wt
        lost_mask = alive*(~w['censor'][:, j]).float() if 'censor' in w else alive
        support_mask = has_next if support_in_secure or 'bucket' not in w else has_next*(w['bucket'][:, j] != 0).float()
        err = state_error(model, flat(xs[j-1]), flat(Rs[j-1]), rep(w['state'][:, j]), rep(w['rot'][:, j]))
        state = sum(weights[name]*e for name, e in err.items()).reshape(E, B)
        forces = (((fs[j-1]-w['forces'][:, j][None])/model.norm_forces.std)**2).mean(-1)
        logit = ls[j-1]
        target = torch.cat((w['contact'][:, j], (w['forces'][:, j, 6:8] > math.log1p(.5)).float()), -1)[None].expand(E, -1, -1)
        contact = bce(logit[..., :8], target, reduction='none').mean(-1)
        support = bce(logit[..., 8], w['supported'][:, j][None].expand(E, -1), reduction='none')
        lost = bce(logit[..., 9], w['lost'][:, j][None].expand(E, -1), reduction='none',
                   pos_weight=torch.tensor(lost_pos_weight, device=logit.device))
        total = total+(((state+forces+contact)*has_next).sum()+(support*support_mask).sum()+(lost*lost_mask).sum())/E
        n_state = n_state+has_next.sum()
    target = true_normalized_delta(model, w)[None]
    m = w['has_next'][:, 0].float()*wt
    nll = (.5*((target-d_n)**2*torch.exp(-logvar)+logvar)).mean(-1)              # (E, B)
    nll = (nll*m).sum()/(E*m.sum().clamp_min(1.))
    bound = .01*(model.max_logvar.sum()-model.min_logvar.sum())
    loss = total/n_state.clamp_min(1.)+nll_weight*nll+bound
    with torch.no_grad():
        logs['nll'] = float(nll)
        logs['logvar_mean'] = float(logvar.mean())
    return loss, logs


class EnsembleMean(nn.Module):
    """The ensemble as one model for evaluate.py: member-mean state and forces, projected mean rotation,
    probabilities averaged (logit of the mean probability)."""

    def __init__(self, ens):
        super().__init__()
        self.ens = ens
        self.L, self.exo_mode = ens.L, ens.exo_mode
        self.norm_state = ens.norm_state
        self.dt = getattr(ens, 'dt', 1/60)

    def rollout(self, w, k):
        from .evaluate import project_so3
        xs, Rs, fs, ls, _ = self.ens.rollout(w, k)
        mx, mR, mf, ml = [], [], [], []
        for x, R, f, logit in zip(xs, Rs, fs, ls):
            Rm = project_so3(R.mean(0))
            xm = x.mean(0).clone()
            xm[:, 3:9] = matrix_to_rot6(Rm)
            p = torch.sigmoid(logit).mean(0).clamp(1e-7, 1-1e-7)
            mx.append(xm); mR.append(Rm); mf.append(f.mean(0)); ml.append(torch.log(p)-torch.log1p(-p))
        return mx, mR, mf, ml


@torch.no_grad()
def uncertainty_report(ens, t, envs, horizons=(1, 8, 16), samples=8192, generator=None):
    """Held-out: one-step aleatoric calibration (share of |true - mean| <= 1 / 2 predicted std for object
    position and rotation change; Gaussian: 0.68 / 0.95) and, per horizon, the Spearman rank correlation
    between member disagreement and the ensemble-mean error (object position mm, rotation deg), plus the AUC
    of disagreement for the worst 10% errors."""
    from .data import gather
    from .evaluate import auc, project_so3
    from .geometry import rotation_angle
    K = max(horizons)
    dev = t['state'].device
    T = t['state'].shape[0]
    s = torch.randint(0, T-K, (samples,), generator=generator).to(dev)
    e = envs.to(dev)[torch.randint(0, len(envs), (samples,), generator=generator).to(dev)]
    w = gather(t, s, e, K)
    xs, Rs, fs, ls, (d_n, logvar) = ens.rollout(w, K)
    m = w['has_next'][:, 0]
    target = true_normalized_delta(ens, w)
    err = (target[None]-d_n).abs()/torch.exp(.5*logvar)
    out = {}
    for name, sl in (('pos', slice(0, 3)), ('rot', slice(3, 6))):
        z = err[..., sl][:, m]
        out[f'aleatoric_{name}'] = dict(within_1sd=round(float((z <= 1).float().mean()), 4), within_2sd=round(float((z <= 2).float().mean()), 4))

    def rank(v):
        r = torch.empty_like(v)
        r[torch.argsort(v)] = torch.arange(len(v), dtype=v.dtype, device=v.device)
        return r

    def spearman(a, b):
        ra, rb = rank(a.double()), rank(b.double())
        ra, rb = ra-ra.mean(), rb-rb.mean()
        return float((ra*rb).sum()/(ra.norm()*rb.norm()).clamp_min(1e-12))
    for h in horizons:
        mk = w['has_next'][:, h-1]
        if int(mk.sum()) < 20:
            continue
        x, R = xs[h-1][:, mk], Rs[h-1][:, mk]
        truth_p, truth_R = w['state'][mk, h, :3], w['rot'][mk, h]
        pm = x[..., :3].mean(0)
        Rm = project_so3(R.mean(0))
        err_p = 1e3*(pm-truth_p).norm(dim=-1)
        err_r = torch.rad2deg(rotation_angle(Rm, truth_R))
        dis_p = 1e3*(x[..., :3]-pm[None]).norm(dim=-1).mean(0)
        dis_r = torch.rad2deg(rotation_angle(R, Rm[None].expand_as(R))).mean(0)
        out[f'h{h}'] = dict(n=int(mk.sum()),
                            spearman_pos=round(spearman(dis_p, err_p), 4), spearman_rot=round(spearman(dis_r, err_r), 4),
                            auc_worst10_pos=round(auc(dis_p, err_p > err_p.quantile(.9)), 4),
                            auc_worst10_rot=round(auc(dis_r, err_r > err_r.quantile(.9)), 4),
                            disagreement_pos_mm_median=round(float(dis_p.median()), 4),
                            error_pos_mm_median=round(float(err_p.median()), 4))
    return out
