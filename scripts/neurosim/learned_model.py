"""E16: does supervising a LEARNED model with expected-gradient (Jacobian) targets make its gradient steer?

Data (MuJoCo testbed, M run 1454 holds, hold / lowering states; physics from TESTBED_* env as elsewhere): per state the
features s (tool geometry and mass, tool pose and velocity in the palm frame, finger angles / velocities / targets,
palm velocity), K random 12-D corrections a (constant over h control steps; noise sigma 'mid' in every rollout) with
their outcome y(a) at h (tool position and rotation change in the palm frame, table normal force), and the expected
Jacobian dE[y]/da at a = 0 (common-random-number central differences, sigma 'mid').

Models (MLP 3 x 256, inputs / outputs standardised): B0 regression of y on (s, a); B1 the same + Jacobian loss
(normalised error + (1 - cosine)) on dy/da at a = 0. Split by episode (80 / 20).

Evaluation on held-out states: cosine of the model's dy/da at 0 with the measured expected Jacobian (per outcome
group); utility as utility_model (E12): optimise a with Adam on the model's objective J(f(s, a)) (30 iterations, no
MuJoCo), evaluate in MuJoCo on 16 noise draws; baselines none, CEM with 24 MuJoCo rollouts, CEM-24 seeded at the
model's correction.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

import gradient_census as G
from utility_test import BOUND, SCALE, objective
from dextrah_lab.neurosim.reference.held_states import mat
from dextrah_lab.neurosim.structured.scenes import tool_testbed

HZ = int(os.environ.get('UTILITY_HORIZON', 2))
K_ACTIONS = 32
Y_DRAWS = int(os.environ.get('Y_DRAWS', 4))
GROUPS = dict(pos_finger=(slice(0, 3), slice(0, 6)), pos_palm=(slice(0, 3), slice(6, 12)),
              rot_finger=(slice(3, 6), slice(0, 6)), rot_palm=(slice(3, 6), slice(6, 12)))
OUT_KEYS = ('rel_pos', 'rel_rot', 'table_force')


def features(tb, ep, k):
    tb.set_state(ep.states[k])
    d = tb.d
    t = ep.tool
    geo = np.concatenate((np.asarray(t['size']).ravel(), np.asarray(t['pos']).ravel(), np.asarray(t['quat']).ravel(),
                          np.asarray(t['used'], float).ravel(), [t['mass']], np.asarray(t['inertia']).ravel()))
    p0, p1 = ep.palm[k], ep.palm[k+1]
    Rp = mat(p0[3:])
    o, v = tb.object_qadr, tb.object_dadr
    rel_p = Rp.T @ (d.qpos[o:o+3]-p0[:3])
    rel_R = Rp.T @ mat(d.qpos[o+3:o+7])
    vel = np.concatenate((Rp.T @ d.qvel[v:v+3], Rp.T @ (mat(d.qpos[o+3:o+7]) @ d.qvel[v+3:v+6])))
    palm_v = Rp.T @ ((p1[:3]-p0[:3])*60)
    fing = np.concatenate((d.qpos[tb.qadr], d.qvel[tb.dadr], ep.targets[k]))
    grav = Rp.T @ np.array([0., 0., -1.])
    return np.concatenate((geo, rel_p, rel_R[:, :2].ravel(), vel, palm_v, grav, fing, [float(ep.palm[k][2]-.8)])).astype(np.float32)


def outcome_vec(y):
    return np.concatenate([y[k] for k in OUT_KEYS])


def state_task(args):
    seed, ep, k, task = args
    rng = np.random.default_rng(seed*7919+k)
    tb = tool_testbed(ep.tool)
    palm, tg, st = ep.palm[k:k+9], ep.targets[k:k+8], ep.states[k]
    sig = G.SIGMAS['mid']
    sd = np.r_[[sig[0]]*6, [sig[1]]*3, [sig[2]]*3]
    s = features(tb, ep, k)
    A = np.clip(rng.normal(size=(K_ACTIONS, 12))*SCALE*1.5, -BOUND, BOUND)
    A[0] = 0.
    # outcome targets averaged over Y_DRAWS noise draws (the regression target approximates E[y | s, a])
    Y = np.stack([np.mean([outcome_vec(G.rollout(tb, st, palm, tg, a, rng.normal(size=(8, 12))*sd, .8)[HZ])
                           for _ in range(Y_DRAWS)], 0) for a in A])
    noises = [rng.normal(size=(8, 12))*sd for _ in range(8)]
    Jh = np.zeros((8, Y.shape[1], 12))
    for c in range(12):
        e = np.zeros(12)
        e[c] = .2*sd[c]
        for i, n in enumerate(noises):
            Jh[i, :, c] = (outcome_vec(G.rollout(tb, st, palm, tg, e, n, .8)[HZ])-outcome_vec(G.rollout(tb, st, palm, tg, -e, n, .8)[HZ]))/(2*e[c])
    return dict(seed=seed, k=k, task=task, s=s, A=A.astype(np.float32), Y=Y.astype(np.float32), J=Jh.mean(0).astype(np.float32),
                Ja=Jh[:4].mean(0).astype(np.float32), Jb=Jh[4:].mean(0).astype(np.float32))


class Net(torch.nn.Module):
    def __init__(self, ds, dy, h=256):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(ds+12, h), torch.nn.SiLU(), torch.nn.Linear(h, h), torch.nn.SiLU(),
                                     torch.nn.Linear(h, h), torch.nn.SiLU(), torch.nn.Linear(h, dy))

    def forward(self, s, a):
        return self.f(torch.cat((s, a), -1))


def train(data, jac_weight, epochs, seed):
    torch.manual_seed(seed)
    S = torch.tensor(np.stack([d['s'] for d in data]))
    A = torch.tensor(np.stack([d['A'] for d in data]))           # (N, K, 12)
    Y = torch.tensor(np.stack([d['Y'] for d in data]))           # (N, K, dy)
    Jt = torch.tensor(np.stack([d['J'] for d in data]))          # (N, dy, 12)
    sm, ss = S.mean(0), S.std(0)+1e-6
    am, as_ = torch.zeros(12), torch.tensor(SCALE, dtype=torch.float32)
    ym, ys = Y.reshape(-1, Y.shape[-1]).mean(0), Y.reshape(-1, Y.shape[-1]).std(0)+1e-6
    net = Net(S.shape[1], Y.shape[-1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    N, K = A.shape[:2]
    norm = dict(sm=sm, ss=ss, am=am, as_=as_, ym=ym, ys=ys)
    # Jacobian target in normalised units: d(y/ys)/d(a/as)
    Jn = Jt/ys[None, :, None]*as_[None, None, :]
    for ep in range(epochs):
        perm = torch.randperm(N)
        for b in perm.split(256):
            sn = ((S[b]-sm)/ss)
            an = A[b]/as_
            pred = net(sn[:, None].expand(-1, K, -1), an)
            loss = ((pred-(Y[b]-ym)/ys)**2).mean()
            if jac_weight > 0:
                a0 = torch.zeros(len(b), 12, requires_grad=True)
                out = net(sn, a0)
                rows = [torch.autograd.grad(out[:, i].sum(), a0, create_graph=True)[0] for i in range(out.shape[1])]
                Jm = torch.stack(rows, 1)                            # (B, dy, 12)
                err = ((Jm-Jn[b])**2).sum(-1)/((Jn[b]**2).sum(-1)+1e-3)
                cos = (Jm*Jn[b]).sum(-1)/(Jm.norm(dim=-1)*Jn[b].norm(dim=-1)+1e-8)
                loss = loss+jac_weight*(err.mean()+(1-cos).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net, norm


def model_jac(net, norm, s):
    sn = ((torch.tensor(s)-norm['sm'])/norm['ss'])[None]
    a0 = torch.zeros(1, 12, requires_grad=True)
    out = net(sn, a0)[0]
    rows = [torch.autograd.grad(out[i], a0, retain_graph=True)[0][0] for i in range(out.shape[0])]
    return (torch.stack(rows)*norm['ys'][:, None]/norm['as_'][None]).detach().numpy()


def model_objective(net, norm, s, a, task, mass):
    sn = ((torch.tensor(s)-norm['sm'])/norm['ss'])[None]
    y = net(sn, (a/norm['as_'])[None])[0]*norm['ys']+norm['ym']
    J = 1e4*(y[0:3]**2).sum()+1e2*(y[3:6]**2).sum()
    if task == 'lowering':
        J = J+1e-2*torch.relu(y[6]-2*mass*9.81)**2
    return J


def utility_task(args):
    seed, ep, k, task, models = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed*104729+k)
    tb = tool_testbed(ep.tool)
    mass = ep.tool['mass']
    palm, tg, st = ep.palm[k:k+9], ep.targets[k:k+8], ep.states[k]
    sig = G.SIGMAS['mid']
    sd = np.r_[[sig[0]]*6, [sig[1]]*3, [sig[2]]*3]
    noise_eval = [rng.normal(size=(8, 12))*sd for _ in range(16)]
    J = lambda a, nz: objective(tb, st, palm, tg, np.clip(a, -BOUND, BOUND), nz, .8, task, mass)
    evaluate = lambda a: float(np.mean([J(a, nz) for nz in noise_eval]))
    s = features(tb, ep, k)
    res = dict(seed=seed, k=k, task=task, none=evaluate(np.zeros(12)))

    def cem(mu0, B):
        mu, sdv, used, pop, elite = mu0.copy(), np.ones(12), 0, 12, 3
        while used+pop <= B:
            Z = np.clip(mu+sdv*rng.normal(size=(pop, 12)), -BOUND/SCALE, BOUND/SCALE)
            nz = rng.normal(size=(8, 12))*sd
            fs = np.array([J(z*SCALE, nz) for z in Z])
            used += pop
            idx = np.argsort(fs)[:elite]
            mu, sdv = Z[idx].mean(0), np.maximum(Z[idx].std(0), .05)
        return mu
    res['cem_24'] = evaluate(cem(np.zeros(12), 24)*SCALE)
    for name, (net, norm) in models.items():
        z = torch.zeros(12, requires_grad=True)
        opt = torch.optim.Adam([z], lr=.1)
        sc = torch.tensor(SCALE, dtype=torch.float32)
        bd = torch.tensor(BOUND/SCALE, dtype=torch.float32)
        for _ in range(30):
            opt.zero_grad()
            model_objective(net, norm, s, z*sc, task, mass).backward()
            opt.step()
            with torch.no_grad():
                z.copy_(torch.maximum(torch.minimum(z, bd), -bd))
        zm = z.detach().numpy()
        res[name] = evaluate(zm*SCALE)
        res[name+'_cem24'] = evaluate(cem(zm, 24)*SCALE)
    return res


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b/(na*nb)) if na > 1e-12 and nb > 1e-12 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=400)
    ap.add_argument('--per_episode', type=int, default=12)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--utility_states', type=int, default=40)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.environ.get('OUT', 'outputs/neurosim/learned_model'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print('CONFIG', json.dumps(vars(a)), 'EPISODE_SOURCE', G.EPISODE_SOURCE, 'HAND_SERVO',
          os.environ.get('TESTBED_HAND_SERVO', 'velocity'), 'HORIZON', HZ, flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        eps = pool.map(G.episode_task, [(a.seed*100_000+i,) for i in range(a.episodes)])
        rng = np.random.default_rng(a.seed)
        jobs, by_ep = [], {}
        for seed, ep, labs in eps:
            ks = [(k, 'hold' if l in ('hold', 'slip') else 'lowering') for k, l in enumerate(labs) if l in ('hold', 'slip', 'lowering')]
            if not ks:
                continue
            sel = rng.choice(len(ks), min(a.per_episode, len(ks)), replace=False)
            for i in sel:
                jobs.append((seed, ep, *ks[i]))
        print('STATES', len(jobs), 'from', len({j[0] for j in jobs}), 'episodes', flush=True)
        data = pool.map(state_task, jobs, chunksize=4)
        print('DATA %.0fs' % (time.time()-t0), flush=True)
        seeds = sorted({d['seed'] for d in data})
        test_seeds = set(rng.choice(seeds, max(1, len(seeds)//5), replace=False).tolist())
        tr = [d for d in data if d['seed'] not in test_seeds]
        te = [d for d in data if d['seed'] in test_seeds]
        print('SPLIT train', len(tr), 'test', len(te), flush=True)
        np.savez_compressed(out/'data.npz', **{key: np.stack([d[key] for d in data]) for key in ('seed', 'k', 's', 'A', 'Y', 'J', 'Ja', 'Jb')},
                            task=np.array([d['task'] for d in data]))
        # ceiling: split-half reliability of the expected-Jacobian targets (4 vs 4 noise draws)
        rel = {g: float(np.nanmedian([cosine(d['Ja'][r, c].ravel(), d['Jb'][r, c].ravel()) for d in te])) for g, (r, c) in GROUPS.items()}
        print('TARGET split-half cos median (4 vs 4 draws)', {g: round(v, 3) for g, v in rel.items()}, flush=True)
        torch.set_num_threads(min(16, a.workers))
        models = {}
        for name, w in (('learned_B0', 0.), ('learned_B1', 1.), ('learned_B10', 10.)):
            t1 = time.time()
            models[name] = train(tr, w, a.epochs, a.seed)
            # held-out one-step prediction error (normalised) and gradient alignment
            net, norm = models[name]
            with torch.no_grad():
                S = torch.tensor(np.stack([d['s'] for d in te]))
                A = torch.tensor(np.stack([d['A'] for d in te]))
                Y = torch.tensor(np.stack([d['Y'] for d in te]))
                pred = net(((S-norm['sm'])/norm['ss'])[:, None].expand(-1, A.shape[1], -1), A/norm['as_'])*norm['ys']+norm['ym']
                r2 = 1-((pred-Y)**2).mean((0, 1))/Y.reshape(-1, Y.shape[-1]).var(0)
            for split, items in (('train', tr[:len(te)]), ('heldout', te)):
                cos = {g: [] for g in GROUPS}
                for d in items:
                    Jm = model_jac(net, norm, d['s'])
                    for g, (r, c) in GROUPS.items():
                        cos[g].append(cosine(Jm[r, c].ravel(), d['J'][r, c].ravel()))
                print('MODEL', name, split, 'grad cos median', {g: round(float(np.nanmedian(v)), 3) for g, v in cos.items()}, flush=True)
            print('MODEL', name, 'train %.0fs' % (time.time()-t1), 'heldout R2 per outcome', np.round(r2.numpy(), 3).tolist(), flush=True)
        # utility on held-out states (40 hold + 40 lowering)
        ute = {}
        for d in te:
            ute.setdefault(d['task'], []).append(d)
        ep_of = {seed: ep for seed, ep, _ in eps}
        ujobs = []
        for task, items in ute.items():
            idx = rng.choice(len(items), min(a.utility_states, len(items)), replace=False)
            ujobs += [(items[i]['seed'], ep_of[items[i]['seed']], items[i]['k'], task, models) for i in idx]
        R = pool.map(utility_task, ujobs)
    (out/'utility.json').write_text(json.dumps(R, indent=1))
    methods = [m for name in ('learned_B0', 'learned_B1', 'learned_B10') for m in (name, name+'_cem24')]+['cem_24']
    for task in ('hold', 'lowering'):
        Sx = [r for r in R if r['task'] == task]
        if not Sx:
            continue
        none = np.array([r['none'] for r in Sx])
        print(f'TASK {task} n={len(Sx)} none median {np.median(none):.3g}')
        for m in methods:
            rel = np.array([r[m] for r in Sx])/np.maximum(none, 1e-12)
            print(f'RESULT {task} {m}: median J/J_none {np.median(rel):.3f} (IQR {np.quantile(rel, .25):.3f}-{np.quantile(rel, .75):.3f}), '
                  f'improved >50% in {np.mean(rel < .5):.2f}, worse in {np.mean(rel > 1):.2f}', flush=True)
    print('DONE %.0fs -> %s' % (time.time()-t0, out), flush=True)


if __name__ == '__main__':
    main()
