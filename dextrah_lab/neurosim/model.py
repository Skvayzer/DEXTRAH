"""Hand-object world model ("neural simulator" for the carry hand), SuperMimic-style.

One deterministic MLP step  (x_t, f_t, a_{t+1}, u_{t+1}, xi) -> (x_{t+1}, f_{t+1}, heads):
  x   continuous state (object pose / velocity in the palm frame, hand joints; data.Layout.state),
  f   log(1 + contact force) of the 5 pads, palm, object-table, object-floor (fed back: the next step sees
      the predicted forces, which carries the grip state a pure kinematic state misses),
  a   hand joint targets, u exogenous inputs (palm motion, gravity, table, pushes), xi object context
      (BPS shape, mass, COM offset) through a small embedding.
Outputs: normalised state change (orientation as a rotation vector composed on the right), next forces,
and logits for pad / palm contact (> 0.5 N), object-table / object-floor contact, supported by the hand,
and lost (the episode ends with a drop) -- the survival / termination head of SuperMimic.

Training (k-step autoregressive, SuperMimic eq. 1): seeded at a logged state, rolled forward on logged
actions / exogenous inputs, losses at every step until the episode ends in the window.
"""
import math

import torch
from torch import nn

from .data import Layout, Normalizer
from .geometry import matrix_to_rot6, rot6_to_matrix, so3_exp, so3_log


def mlp(sizes, act=nn.SiLU, last_act=False):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes)-2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


class HandObjectWorldModel(nn.Module):
    def __init__(self, layout: Layout, hidden=(1024, 1024, 1024), context_embed=64):
        super().__init__()
        self.L = L = layout
        self.norm_state = Normalizer(L.state)
        self.norm_forces = Normalizer(L.forces)
        self.norm_action = Normalizer(L.actuated)
        self.norm_exo = Normalizer(L.exo)
        self.norm_context = Normalizer(L.context)
        self.norm_delta = Normalizer(L.delta)
        self.context_net = mlp((L.context, 256, context_embed), last_act=True)
        inp = L.state+L.forces+L.actuated+L.exo+context_embed
        self.trunk = mlp((inp,)+tuple(hidden), last_act=True)
        h = hidden[-1]
        self.head_delta = nn.Linear(h, L.delta)
        self.head_forces = nn.Linear(h, L.forces)
        self.head_logits = nn.Linear(h, 6+2+1+1)     # pads + palm contact, table, floor, supported, lost
        for head in (self.head_delta, self.head_forces):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        nn.init.zeros_(self.head_logits.weight)
        self.head_logits.bias.data[-1] = -8.          # lost is rare (~1e-4 per step)

    def embed(self, context):
        return self.context_net(self.norm_context(context))

    def step(self, x, R, f, a, u, z):
        """x (B, state) with its orientation R (B, 3, 3) (x carries rot6(R)); f forces; a action; u exo;
        z context embedding. Returns next (x, R, f) and the logits."""
        inp = torch.cat((self.norm_state(x), self.norm_forces(f), self.norm_action(a), self.norm_exo(u), z), -1)
        h = self.trunk(inp)
        d = self.head_delta(h)*self.norm_delta.std+self.norm_delta.mean
        H = self.L.hand
        R_next = R@so3_exp(d[:, 3:6])
        x_next = torch.cat((x[:, :3]+d[:, :3], matrix_to_rot6(R_next), x[:, 9:15]+d[:, 6:12],
                            x[:, 15:15+H]+d[:, 12:12+H], x[:, 15+H:]+d[:, 12+H:]), -1)
        f_next = (f+self.head_forces(h)*self.norm_forces.std).clamp_min(0.)
        return x_next, R_next, f_next, self.head_logits(h)

    def rollout(self, w, k):
        """Open-loop k-step rollout of window tensors w (data.gather): seeded at row 0, logged actions and
        exogenous inputs of rows 1..k. Returns lists of predicted x, R, f, logits for steps 1..k."""
        x, R, f = w['state'][:, 0], w['rot'][:, 0], w['forces'][:, 0]
        z = self.embed(w['context'][:, 0])
        xs, Rs, fs, ls = [], [], [], []
        for j in range(1, k+1):
            x, R, f, logit = self.step(x, R, f, w['action'][:, j], w['exo'][:, j], z)
            xs.append(x); Rs.append(R); fs.append(f); ls.append(logit)
        return xs, Rs, fs, ls

    def fit_normalizers(self, t, rows):
        """Normalisers from the training rows (t: data.build_tensors; rows: (T', N') index tuple)."""
        sel = lambda x: x[rows]
        self.norm_state.fit(sel(t['state']))
        self.norm_forces.fit(sel(t['forces']))
        self.norm_action.fit(sel(t['action']))
        self.norm_exo.fit(sel(t['exo']))
        tr, er = rows
        self.norm_context.fit(torch.cat((t['bps'][er], t['masscom'][tr, er]), -1))
        self.norm_delta.fit(one_step_deltas(t, rows, self.L))
        return self


def one_step_deltas(t, rows, L):
    """Logged one-step state changes (for the delta normaliser), within episodes."""
    tr, er = rows
    keep = tr > 0
    tr, er = tr[keep], er[keep]
    cont = ~t['done'][tr, er]
    tr, er = tr[cont], er[cont]
    x0, x1 = t['state'][tr-1, er], t['state'][tr, er]
    R0, R1 = t['rot'][tr-1, er], t['rot'][tr, er]
    H = L.hand
    return torch.cat((x1[:, :3]-x0[:, :3], so3_log(R0.transpose(-1, -2)@R1), x1[:, 9:15]-x0[:, 9:15],
                      x1[:, 15:15+H]-x0[:, 15:15+H], x1[:, 15+H:]-x0[:, 15+H:]), -1)


STATE_WEIGHTS = dict(pos=1., rot=1., linvel=.25, angvel=.25, hand_q=1., hand_qd=.25)


def state_error(model, x_hat, R_hat, x, R):
    """Per-sample state errors, normalised by the state std: dict of (B,) squared errors."""
    L, std = model.L, model.norm_state.std
    H = L.hand
    rot_std = std[3:9].mean()
    sq = lambda a, b, s: (((a-b)/s)**2).mean(-1)
    return dict(pos=sq(x_hat[:, :3], x[:, :3], std[:3]),
                # chordal distance |R_hat - R|_F^2 / 2 = 2 (1 - cos theta) ~ theta^2: smooth everywhere (the squared
                # rotation-vector norm has an unbounded gradient near theta = pi; job 1591 iteration 15000: |g| 8130)
                rot=((R_hat-R)**2).sum((-1, -2))/2/rot_std**2,
                linvel=sq(x_hat[:, 9:12], x[:, 9:12], std[9:12]), angvel=sq(x_hat[:, 12:15], x[:, 12:15], std[12:15]),
                hand_q=sq(x_hat[:, 15:15+H], x[:, 15:15+H], std[15:15+H]),
                hand_qd=sq(x_hat[:, 15+H:], x[:, 15+H:], std[15+H:]))


def window_loss(model, w, k, lost_pos_weight=1., weights=STATE_WEIGHTS):
    """k-step autoregressive loss on window tensors w; returns (loss, logs)."""
    xs, Rs, fs, ls = model.rollout(w, k)
    bce = nn.functional.binary_cross_entropy_with_logits
    total, logs = 0., {}
    n_state = n_alive = torch.zeros((), device=w['state'].device)
    for j in range(1, k+1):
        has_next, alive = w['has_next'][:, j-1].float(), w['alive'][:, j-1].float()
        err = state_error(model, xs[j-1], Rs[j-1], w['state'][:, j], w['rot'][:, j])
        state = sum(weights[name]*e for name, e in err.items())
        forces = (((fs[j-1]-w['forces'][:, j])/model.norm_forces.std)**2).mean(-1)
        logit = ls[j-1]
        contact = bce(logit[:, :8], torch.cat((w['contact'][:, j], (w['forces'][:, j, 6:8] > math.log1p(.5)).float()), -1),
                      reduction='none').mean(-1)
        support = bce(logit[:, 8], w['supported'][:, j], reduction='none')
        lost = bce(logit[:, 9], w['lost'][:, j], pos_weight=torch.tensor(lost_pos_weight, device=logit.device), reduction='none')
        total = total+((state+forces+contact+support)*has_next).sum()+(lost*alive).sum()
        n_state, n_alive = n_state+has_next.sum(), n_alive+alive.sum()
        with torch.no_grad():                         # accumulate on the device; one sync per call
            for name, e in err.items():
                logs[name] = logs.get(name, 0.)+(e*has_next).sum()
            logs['forces'] = logs.get('forces', 0.)+(forces*has_next).sum()
            logs['contact_bce'] = logs.get('contact_bce', 0.)+(contact*has_next).sum()
            logs['lost_bce'] = logs.get('lost_bce', 0.)+(lost*alive).sum()
    n_state, n_alive = n_state.clamp_min(1.), n_alive.clamp_min(1.)
    logs = {name: float(v/(n_alive if name == 'lost_bce' else n_state)) for name, v in logs.items()}
    return total/n_state, logs
