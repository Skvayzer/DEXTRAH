"""Go / no-go metrics of the hand-object world model on held-out envs (objects never seen in training).

* Open-loop prediction error at horizons 1 / 8 / 32 steps (seeded at a logged state, logged actions and
  exogenous inputs): object position (mm), object rotation (deg), hand joints (rad), contact forces
  (log1p N), against the persistence baseline (nothing changes from the seed state).
* Contact / support classification (AUC) at horizons 1 and 8.
* Loss of the object within the next 0.5 s: P = 1 - prod_j (1 - p_j) over the rolled-out lost head.
  AUC on all held-out loss events (starts 1..H steps before) plus random starts; calibration (Brier,
  binned reliability) on random starts only, at the natural rate. Windows where the episode ends for
  another reason (timeout, fall) inside the horizon are excluded.
"""
import math

import torch

from .data import gather
from .geometry import rotation_angle


def auc(score, label):
    """Area under the ROC curve (ties averaged); nan if one class is missing."""
    score, label = score.double().flatten(), label.bool().flatten()
    pos, neg = int(label.sum()), int((~label).sum())
    if pos == 0 or neg == 0:
        return float('nan')
    order = torch.argsort(score)
    ranks = torch.empty_like(score)
    ranks[order] = torch.arange(1, len(score)+1, dtype=score.dtype, device=score.device)
    # average ranks over ties
    uniq, inv, counts = torch.unique(score, return_inverse=True, return_counts=True)
    sums = torch.zeros(len(uniq), dtype=score.dtype, device=score.device).index_add_(0, inv, ranks)
    ranks = (sums/counts)[inv]
    return float((ranks[label].sum()-pos*(pos+1)/2)/(pos*neg))


@torch.no_grad()
def open_loop(model, t, envs, horizons=(1, 8, 32), samples=8192, generator=None):
    K = max(horizons)
    dev = t['state'].device
    T = t['state'].shape[0]
    s = torch.randint(0, T-K, (samples,), generator=generator).to(dev)
    e = envs.to(dev)[torch.randint(0, len(envs), (samples,), generator=generator).to(dev)]
    w = gather(t, s, e, K)
    xs, Rs, fs, ls = model.rollout(w, K)
    H = model.L.hand
    out = {}
    for h in horizons:
        m = w['has_next'][:, h-1]
        if int(m.sum()) < 10:
            continue
        x, R, f = w['state'][m, h], w['rot'][m, h], w['forces'][m, h]
        x0, R0, f0 = w['state'][m, 0], w['rot'][m, 0], w['forces'][m, 0]
        xh, Rh, fh, lh = xs[h-1][m], Rs[h-1][m], fs[h-1][m], ls[h-1][m]
        stats = lambda v: dict(median=round(float(v.median()), 4), mean=round(float(v.mean()), 4), p90=round(float(v.quantile(.9)), 4))
        row = dict(n=int(m.sum()))
        for name, model_err, base_err in (
                ('pos_mm', 1e3*(xh[:, :3]-x[:, :3]).norm(dim=-1), 1e3*(x0[:, :3]-x[:, :3]).norm(dim=-1)),
                ('rot_deg', torch.rad2deg(rotation_angle(Rh, R)), torch.rad2deg(rotation_angle(R0, R))),
                ('hand_q_rad', (xh[:, 15:15+H]-x[:, 15:15+H]).abs().mean(-1), (x0[:, 15:15+H]-x[:, 15:15+H]).abs().mean(-1)),
                ('force_log1p', (fh-f).abs().mean(-1), (f0-f).abs().mean(-1))):
            row[name] = dict(model=stats(model_err), persistence=stats(base_err))
        # windows where the object really moves in the hand (>= 5 mm or >= 5 deg over the horizon): slips,
        # re-grips, drops in progress -- persistence is a strong baseline only when nothing happens
        moving = ((x0[:, :3]-x[:, :3]).norm(dim=-1) >= .005) | (torch.rad2deg(rotation_angle(R0, R)) >= 5.)
        if int(moving.sum()) >= 10:
            row['moving'] = dict(n=int(moving.sum()), fraction=round(float(moving.float().mean()), 4))
            for name, model_err, base_err in (
                    ('pos_mm', 1e3*(xh[moving, :3]-x[moving, :3]).norm(dim=-1), 1e3*(x0[moving, :3]-x[moving, :3]).norm(dim=-1)),
                    ('rot_deg', torch.rad2deg(rotation_angle(Rh[moving], R[moving])), torch.rad2deg(rotation_angle(R0[moving], R[moving])))):
                row['moving'][name] = dict(model=stats(model_err), persistence=stats(base_err))
        contact = w['contact'][m, h]
        row['contact_auc'] = [round(auc(lh[:, i], contact[:, i]), 4) for i in range(6)]
        row['support_auc'] = round(auc(lh[:, 8], w['supported'][m, h]), 4)
        row['contact_rate'] = [round(float(contact[:, i].mean()), 4) for i in range(6)]
        out[h] = row
    return out


@torch.no_grad()
def loss_prediction(model, t, envs, horizon=30, random_samples=20000, generator=None, bins=10):
    dev = t['state'].device
    T = t['state'].shape[0]
    envs = envs.to(dev)
    in_test = torch.zeros(t['lost'].shape[1], dtype=torch.bool, device=dev)
    in_test[envs] = True
    rows, cols = torch.nonzero((t['lost'] > .5) & in_test[None], as_tuple=True)
    # positives: starts 1..horizon steps before each loss event
    back = torch.randint(1, horizon+1, (len(rows),), generator=generator).to(dev)
    s_pos, e_pos = (rows-back).clamp(min=0), cols
    s_rand = torch.randint(0, T-horizon, (random_samples,), generator=generator).to(dev)
    e_rand = envs[torch.randint(0, len(envs), (random_samples,), generator=generator).to(dev)]
    s = torch.cat((s_pos, s_rand)).clamp(max=T-horizon-1)
    e = torch.cat((e_pos, e_rand))
    P, label, clean = [], [], []
    for a in range(0, len(s), 4096):
        w = gather(t, s[a:a+4096], e[a:a+4096], horizon)
        _, _, _, ls = model.rollout(w, horizon)
        hazard = torch.stack([torch.sigmoid(l[:, 9]) for l in ls], 1)*w['alive'].float()
        P.append(1-torch.prod(1-hazard, 1))
        lost_any = (w['lost'][:, 1:]*w['alive'].float()).amax(1)
        ended = (~w['has_next'] & w['alive']).any(1)        # an episode end inside the horizon
        label.append(lost_any)
        clean.append(~ended | (lost_any > .5))               # drop windows ending in another way
    P, label, clean = torch.cat(P), torch.cat(label), torch.cat(clean)
    n_pos = len(s_pos)
    rand = torch.zeros_like(clean); rand[n_pos:] = True
    keep = clean
    res = dict(horizon_s=round(horizon*float(model.dt if hasattr(model, 'dt') else 1/60), 3), events=int(n_pos),
               auc=round(auc(P[keep], label[keep]), 4))
    r = keep & rand
    p, y = P[r], label[r]
    res.update(random_windows=int(r.sum()), base_rate=round(float(y.mean()), 5), mean_predicted=round(float(p.mean()), 5),
               brier=round(float(((p-y)**2).mean()), 6),
               brier_climatology=round(float(((y.mean()-y)**2).mean()), 6))
    edges = torch.quantile(p, torch.linspace(0, 1, bins+1, device=dev))
    rel = []
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i+1]) if i == bins-1 else (p >= edges[i]) & (p < edges[i+1])
        if int(m.sum()):
            rel.append(dict(predicted=round(float(p[m].mean()), 5), observed=round(float(y[m].mean()), 5), n=int(m.sum())))
    res['reliability'] = rel
    return res
