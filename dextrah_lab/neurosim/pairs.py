"""Paired counterfactual windows and the difference loss (2026-10-28).

scripts/neurosim/collect_counterfactual_pairs.py restores groups of K bit-identical worlds (copy 0 = the carry
expert, copies 1..K-1 = perturbed finger commands) and logs rows t0-2 .. t0+H; row 1 is the last unperturbed
state. Over the short window the simulator is deterministic, so the difference between a perturbed copy and the
baseline is the effect of the command change. The difference loss rolls the model out from row 1 for both and
fits the predicted difference (state, contact forces) to the simulated one, each dimension scaled by the spread
of the simulated differences; it supervises the action response the expert's own data barely excite.
"""
from pathlib import Path

import torch

from .data import FIELDS, build_tensors, gather

WORLD_KEYS = ('bps', 'asset')                 # (N, ...) per world; every other tensor is (T, N, ...)


def load_pairs(directory, *, exo_mode, lowpass_tau, labels, dt, copies):
    """Rounds of one collection concatenated along worlds (global world = round * n + group * K + copy).
    Returns the build_tensors dict plus 'group_end' (groups,) and 'copies'."""
    files = sorted(Path(directory).glob('round_*.pt'))
    if not files:
        raise FileNotFoundError(f'no round_*.pt in {directory}')
    parts, ends = [], []
    for f in files:
        raw = torch.load(f, map_location='cpu', weights_only=False)
        data = {k: raw[k].float() for k in FIELDS if k in raw}
        context = dict(bps=raw['bps'].float(), asset=raw['asset'], dt=dt)
        if data['done'].shape[1] % copies:
            raise ValueError(f'{f}: {data["done"].shape[1]} worlds is not a multiple of {copies} copies')
        parts.append(build_tensors(data, context, exo_mode=exo_mode, lowpass_tau=lowpass_tau, labels=labels))
        ends.append(raw['group_end'].bool())
    out = {}
    for k in parts[0]:
        out[k] = torch.cat([p[k] for p in parts], 0 if k in WORLD_KEYS else 1)
    out['group_end'] = torch.cat(ends)
    out['copies'] = copies
    return out


def difference_scales(tp, groups, k, floor_state, floor_forces):
    """Per-dimension spread of the simulated copy - baseline differences over the window (rows 2 .. k+1)."""
    K = tp['copies']
    base = groups*K
    others = base[:, None]+torch.arange(1, K, device=groups.device)
    dx = tp['state'][2:k+2, others]-tp['state'][2:k+2, base][:, :, None]
    df = tp['forces'][2:k+2, others]-tp['forces'][2:k+2, base][:, :, None]
    sx = dx.flatten(0, 2).std(0)
    sf = df.flatten(0, 2).std(0)
    return torch.maximum(sx, floor_state), torch.maximum(sf, floor_forces)


def pair_loss(model, tp, groups, k, scale_state, scale_forces, generator=None):
    """Difference loss for (B,) global group ids: one random perturbed copy per group against its baseline,
    both rolled out k steps from row 1 with their own logged actions and exogenous inputs. Returns (loss, logs)
    with the mean scaled squared error and the median cosine of predicted vs simulated differences."""
    K = tp['copies']
    dev = groups.device
    c = torch.randint(1, K, (len(groups),), generator=generator, device='cpu').to(dev)
    base, other = groups*K, groups*K+c
    s = torch.ones(len(groups), dtype=torch.long, device=dev)
    wb, wc = gather(tp, s, base, k), gather(tp, s, other, k)
    xb, _, fb, _ = model.rollout(wb, k)
    xc, _, fc, _ = model.rollout(wc, k)
    pred_x = torch.stack(xc, 1)-torch.stack(xb, 1)
    pred_f = torch.stack(fc, 1)-torch.stack(fb, 1)
    sim_x = wc['state'][:, 1:]-wb['state'][:, 1:]
    sim_f = wc['forces'][:, 1:]-wb['forces'][:, 1:]
    mask = (wb['has_next'] & wc['has_next']).float()
    err = (((pred_x-sim_x)/scale_state)**2).mean(-1)+(((pred_f-sim_f)/scale_forces)**2).mean(-1)
    loss = (err*mask).sum()/mask.sum().clamp_min(1.)
    with torch.no_grad():
        p = torch.cat(((pred_x/scale_state).flatten(1), (pred_f/scale_forces).flatten(1)), 1)
        q = torch.cat(((sim_x/scale_state).flatten(1), (sim_f/scale_forces).flatten(1)), 1)
        cos = (p*q).sum(1)/(p.norm(dim=1)*q.norm(dim=1)).clamp_min(1e-12)
        ratio = p.norm(dim=1)/q.norm(dim=1).clamp_min(1e-12)
    return loss, dict(pair_err=float(loss), pair_cos_median=float(cos.median()), pair_norm_ratio_median=float(ratio.median()))
