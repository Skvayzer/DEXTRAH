"""E18: train a POLICY with the structured model's gradients (no MuJoCo in training) and test it in MuJoCo on held-out tools.

Policy: MLP features(s) -> 12-D correction z (a = z * SCALE, |z| <= BOUND / SCALE via tanh), added to M's logged commands
(fingers) and palm motion for h = 2 control steps (as utility_model / robust_steering). States: M 1454 hold and lowering
states, split by episode (tool) 80 / 20.
Training: each update takes a minibatch of training states; workers return g_i = d J_model(s_i, a) / d z at
z_i = pi(s_i), divided by the state's model cost at z = 0 (relative objective); the policy loss is mean_i g_i . z_i(theta)
(chain rule). Arms with identical states / updates / seeds:
  expected       J_model = mean over K = 8 antithetic per-step noise draws (evaluation's noise model) - E17 step_K8
  deterministic  J_model without noise (the 'analytic policy gradient' through the mean dynamics)
Evaluation (MuJoCo, 16 noise draws): J(pi(s)) / J(0) on held-out and on training states.
"""
import argparse
import json
import os
import time
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

import gradient_census as G
from learned_model import features
from robust_steering import HZ, model_cost
from utility_model import T
from utility_test import BOUND, SCALE, objective
from dextrah_lab.neurosim.structured.contact import ContactModel
from dextrah_lab.neurosim.structured.dynamics import SimState, StructuredSim
from dextrah_lab.neurosim.structured.scenes import tool_testbed
from dextrah_lab.neurosim.structured.spec import extract

SIG = G.SIGMAS['mid']
SD = np.r_[[SIG[0]]*6, [SIG[1]]*3, [SIG[2]]*3]
_CACHE = OrderedDict()
EPS = {}                      # seed -> EpisodeLog, set before the worker pool forks (shared copy-on-write)


def scene(seed, ep):
    """Per-worker cache of (testbed, model) per episode / tool."""
    if seed not in _CACHE:
        tb = tool_testbed(ep.tool)
        sc = extract(tb)
        cm = ContactModel(sc)
        _CACHE[seed] = (tb, sc, cm)
        while len(_CACHE) > 2:
            _CACHE.popitem(last=False)
    return _CACHE[seed]


def grad_task(args):
    """d J_model / d z at z (relative to J_model(0) of the same noise draws), and J_model."""
    seed, k, task, z, mode, rseed = args
    ep = EPS[seed]
    torch.set_num_threads(1)
    tb, sc, cm = scene(seed, ep)
    palm, tg, st = ep.palm[k:k+9], ep.targets[k:k+8], ep.states[k]
    tb.set_state(st)
    s0 = SimState.from_testbed(tb)
    sim = StructuredSim(sc, contact_model=cm)
    mass = ep.tool['mass']
    rng = np.random.default_rng(rseed)
    if mode == 'det':
        nz = torch.zeros(1, HZ, 12, dtype=torch.float64)
    else:
        h = T(rng.normal(size=(4, HZ, 12))*SD)
        nz = torch.cat((h, -h), 0)
    tk = None
    if task == 'lowering':
        with torch.no_grad():
            model_cost(sim, s0, palm, tg, torch.zeros(12, dtype=torch.float64), nz[:1], 'hold', mass, None)
        tk = torch.tensor([kd == 'object_table' for kd in sim.last['contacts'].kind], dtype=torch.float64)
    try:
        with torch.no_grad():
            J0 = float(model_cost(sim, s0, palm, tg, torch.zeros(12, dtype=torch.float64), nz, task, mass, tk))
        zt = T(z).requires_grad_(True)
        f = model_cost(sim, s0, palm, tg, zt*T(SCALE), nz, task, mass, tk)
        f.backward()
        g = zt.grad.numpy()/max(J0, 1e-6)
        return dict(g=g, J=float(f)/max(J0, 1e-6), ok=bool(np.all(np.isfinite(g))))
    except Exception as e:
        return dict(g=np.zeros(12), J=np.nan, ok=False, error=repr(e))


def eval_task(args):
    seed, k, task, zs = args
    ep = EPS[seed]
    rng = np.random.default_rng(seed*7919+k)
    tb = tool_testbed(ep.tool)
    palm, tg, st = ep.palm[k:k+9], ep.targets[k:k+8], ep.states[k]
    noise_eval = [rng.normal(size=(8, 12))*SD for _ in range(16)]
    ev = lambda a: float(np.mean([objective(tb, st, palm, tg, np.clip(a, -BOUND, BOUND), nz, .8, task, ep.tool['mass'])
                                  for nz in noise_eval]))
    res = dict(seed=seed, k=k, task=task, none=ev(np.zeros(12)))
    for name, z in zs.items():
        res[name] = ev(z*SCALE)
    return res


def feature_task(args):
    seed, k, task = args
    ep = EPS[seed]
    tb = tool_testbed(ep.tool)
    return np.concatenate((features(tb, ep, k), [float(task == 'lowering')])).astype(np.float32)


class Policy(torch.nn.Module):
    def __init__(self, ds, mean, std, h=128):
        super().__init__()
        self.mean, self.std = mean, std
        self.f = torch.nn.Sequential(torch.nn.Linear(ds, h), torch.nn.SiLU(), torch.nn.Linear(h, h), torch.nn.SiLU(),
                                     torch.nn.Linear(h, 12))
        torch.nn.init.zeros_(self.f[-1].weight)
        torch.nn.init.zeros_(self.f[-1].bias)
        self.bd = torch.tensor(BOUND/SCALE, dtype=torch.float32)

    def forward(self, s):
        return self.bd*torch.tanh(self.f((s-self.mean)/self.std)/self.bd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=300)
    ap.add_argument('--per_episode', type=int, default=4)
    ap.add_argument('--updates', type=int, default=60)
    ap.add_argument('--batch', type=int, default=56)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--eval_states', type=int, default=80)
    ap.add_argument('--arms', default='expected,det')
    ap.add_argument('--workers', type=int, default=28)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=os.environ.get('OUT', 'outputs/neurosim/policy_from_model'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print('CONFIG', json.dumps(vars(a)), 'EPISODE_SOURCE', G.EPISODE_SOURCE, 'HORIZON', HZ, flush=True)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    with Pool(a.workers) as pool:
        eps = pool.map(G.episode_task, [(a.seed*100_000+i,) for i in range(a.episodes)])
    states = []
    for seed, ep, labs in eps:
        EPS[seed] = ep
        ks = [(k, 'hold' if l in ('hold', 'slip') else 'lowering') for k, l in enumerate(labs) if l in ('hold', 'slip', 'lowering')]
        if ks:
            for i in rng.choice(len(ks), min(a.per_episode, len(ks)), replace=False):
                states.append((seed, *ks[i]))
    del eps
    seeds = sorted({s[0] for s in states})
    test_seeds = set(rng.choice(seeds, len(seeds)//5, replace=False).tolist())
    tr = [s for s in states if s[0] not in test_seeds]
    te = [s for s in states if s[0] in test_seeds]
    with Pool(a.workers) as pool:                    # forked after EPS is filled
        F = np.stack(pool.map(feature_task, states, chunksize=4))
        idx = {(s[0], s[1]): i for i, s in enumerate(states)}
        Ftr = F[[idx[(s[0], s[1])] for s in tr]]
        mean, std = torch.tensor(Ftr.mean(0)), torch.tensor(Ftr.std(0)+1e-6)
        print('STATES train', len(tr), 'test', len(te), 'tools test', len(test_seeds), 'features', F.shape[1],
              '%.0fs' % (time.time()-t0), flush=True)
        policies = {}
        for arm in a.arms.split(','):
            torch.manual_seed(a.seed)
            pi = Policy(F.shape[1], mean, std)
            opt = torch.optim.Adam(pi.parameters(), lr=a.lr)
            brng = np.random.default_rng(a.seed+1)
            for u in range(a.updates):
                b = brng.choice(len(tr), a.batch, replace=False)
                S = torch.tensor(F[[idx[(tr[i][0], tr[i][1])] for i in b]])
                Z = pi(S)
                jobs = [(*tr[i], Z[j].detach().double().numpy(), 'det' if arm == 'det' else 'expected', a.seed*1_000_003+u*1000+j)
                        for j, i in enumerate(b)]
                R = pool.map(grad_task, jobs)
                ok = torch.tensor([r['ok'] for r in R])
                Gm = torch.tensor(np.stack([r['g'] for r in R]), dtype=torch.float32)
                Gm = torch.where(ok[:, None], Gm, torch.zeros_like(Gm))
                # clip per-state relative gradients (a few states have huge relative sensitivities)
                n = Gm.norm(dim=1, keepdim=True)
                Gm = Gm*torch.clamp(10./(n+1e-9), max=1.)
                loss = (Gm*Z).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if u % 5 == 0 or u == a.updates-1:
                    Js = np.array([r['J'] for r in R if r['ok']])
                    print(f'TRAIN {arm} update {u} model J/J0 (batch, before step) median {np.median(Js):.3f} mean {np.mean(Js):.3f} '
                          f'|z| median {Z.detach().norm(dim=1).median():.3f} failed {int((~ok).sum())} {time.time()-t0:.0f}s', flush=True)
            policies[arm] = pi
            torch.save(pi.state_dict(), out/f'policy_{arm}.pt')
        # MuJoCo evaluation on held-out tools and on training states
        ev_sets = {'heldout': [te[i] for i in rng.choice(len(te), min(a.eval_states, len(te)), replace=False)],
                   'train': [tr[i] for i in rng.choice(len(tr), min(a.eval_states//2, len(tr)), replace=False)]}
        results = {}
        for split, items in ev_sets.items():
            jobs = []
            for s in items:
                f = torch.tensor(F[idx[(s[0], s[1])]])[None]
                with torch.no_grad():
                    zs = {arm: pi(f)[0].double().numpy() for arm, pi in policies.items()}
                jobs.append((*s, zs))
            results[split] = pool.map(eval_task, jobs)
    (out/'eval.json').write_text(json.dumps(results, indent=1))
    for split, R in results.items():
        for task in ('hold', 'lowering'):
            S = [r for r in R if r['task'] == task]
            if not S:
                continue
            none = np.array([r['none'] for r in S])
            for arm in policies:
                rel = np.array([r[arm] for r in S])/np.maximum(none, 1e-12)
                print(f'RESULT {split} {task} n={len(S)} policy_{arm}: median J/J_none {np.median(rel):.3f} (IQR {np.quantile(rel, .25):.3f}-'
                      f'{np.quantile(rel, .75):.3f}), mean {np.mean(rel):.3f}, improved >50% in {np.mean(rel < .5):.2f}, worse in {np.mean(rel > 1):.2f}', flush=True)
    print('DONE %.0fs -> %s' % (time.time()-t0, out), flush=True)


if __name__ == '__main__':
    main()
