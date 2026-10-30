"""E8: does the structured model's gradient alone steer a correction that works in the true simulator?

Same states and objective as utility_test (hold / lowering states of M run 1454 or scripted episodes; constant 12-D
correction over h control steps; J = 1e4 |dp_rel|^2 + 1e2 |drot_rel|^2 of the tool in the palm frame at h), but the
correction is optimised with gradients of the STRUCTURED MODEL only (autograd through h model steps, Adam, no MuJoCo
rollout during optimisation), then evaluated in MuJoCo on 16 noise draws (sigma mid). Model variants: exact (band 0)
and backward-only soft servo band s (forward unchanged). Baselines: none; CEM with B MuJoCo rollouts.
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
from dextrah_lab.neurosim.structured.contact import ContactModel
from dextrah_lab.neurosim.structured.dynamics import SimState, StructuredSim
from dextrah_lab.neurosim.structured.rotations import axis_angle_quat, quat_mul, quat_to_mat
from dextrah_lab.neurosim.structured.scenes import tool_testbed
from dextrah_lab.neurosim.structured.spec import extract

HZ = int(os.environ.get('UTILITY_HORIZON', 2))
BANDS = (0., .005, .02)
DT = 1/60
T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)


def model_objective(sim, s0, palm, tg, a):
    """J of the model rollout for correction a (12, torch), same definition as utility_test.objective."""
    kfac = torch.tensor([G.COUPLE_K.get(j, 1.) for js in G.COUPLING for j in js], dtype=torch.float64)
    group = [c for c, js in enumerate(G.COUPLING) for _ in js]
    R0 = quat_to_mat(T(palm[0][3:]))
    rel_p0 = R0.T @ (s0.obj_pos[0]-T(palm[0][:3]))
    rel_R0 = R0.T @ quat_to_mat(s0.obj_quat[0])
    s, prev = s0, T(palm[0])[None]
    w = a[9:12]
    n = torch.sqrt((w*w).sum()+1e-30)
    for k in range(HZ):
        p = T(palm[k+1])[None]
        dq = axis_angle_quat(w/n, n*DT*(k+1))
        nxt = torch.cat((p[:, :3]+a[6:9]*DT*(k+1), quat_mul(dq, p[0, 3:])[None]), -1)
        s = sim.step(s, T(tg[k])[None]+(a[group]*kfac)[None], prev, nxt)
        prev = nxt
    Rp = quat_to_mat(prev[0, 3:])
    dp = Rp.T @ (s.obj_pos[0]-prev[0, :3])-rel_p0
    Rr = rel_R0.T @ (Rp.T @ quat_to_mat(s.obj_quat[0]))
    ang = torch.acos(((Rr.diagonal().sum()-1)/2).clamp(-1+1e-12, 1-1e-12))
    return 1e4*(dp*dp).sum()+1e2*ang*ang


def run_state(args):
    seed, ep, k, task, sigma_name, iters, budgets = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed*7919+k)
    tb = tool_testbed(ep.tool)
    sc = extract(tb)
    top, mass = .8, ep.tool['mass']
    palm, tg, st = ep.palm[k:k+9], ep.targets[k:k+8], ep.states[k]
    sig = G.SIGMAS[sigma_name]
    sd = np.r_[[sig[0]]*6, [sig[1]]*3, [sig[2]]*3]
    noise_eval = [rng.normal(size=(8, 12))*sd for _ in range(16)]
    J = lambda a, nz: objective(tb, st, palm, tg, np.clip(a, -BOUND, BOUND), nz, top, task, mass)
    evaluate = lambda a: float(np.mean([J(a, nz) for nz in noise_eval]))
    res = dict(seed=seed, k=k, task=task, none=evaluate(np.zeros(12)))
    tb.set_state(st)
    s0 = SimState.from_testbed(tb)
    cm = ContactModel(sc)
    t0 = time.time()
    for b in BANDS:
        sim = StructuredSim(sc, contact_model=cm, servo_band=b or None)
        z = torch.zeros(12, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([z], lr=.1)
        sc_t, bd = T(SCALE), T(BOUND/SCALE)
        try:
            for _ in range(iters):
                opt.zero_grad()
                f = model_objective(sim, s0, palm, tg, z*sc_t)
                f.backward()
                opt.step()
                with torch.no_grad():
                    z.copy_(torch.maximum(torch.minimum(z, bd), -bd))
            res[f'model_{b:g}'] = evaluate(z.detach().numpy()*SCALE)
        except Exception as e:
            res[f'model_{b:g}'] = np.nan
            res['error'] = repr(e)
    res['model_sec'] = time.time()-t0
    for B in budgets:                                  # CEM with B MuJoCo rollouts
        mu, sdv, used, pop, elite = np.zeros(12), np.ones(12), 0, min(12, B), max(2, min(12, B)//4)
        while used+pop <= B:
            Z = np.clip(mu+sdv*rng.normal(size=(pop, 12)), -BOUND/SCALE, BOUND/SCALE)
            nz = rng.normal(size=(8, 12))*sd
            fs = np.array([J(z*SCALE, nz) for z in Z])
            used += pop
            idx = np.argsort(fs)[:elite]
            mu, sdv = Z[idx].mean(0), np.maximum(Z[idx].std(0), .05)
        res[f'cem_{B}'] = evaluate(mu*SCALE)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=300)
    ap.add_argument('--per_task', type=int, default=40)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--budgets', default='24,96,600')
    ap.add_argument('--sigma', default='mid')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=os.environ.get('OUT', 'outputs/neurosim/utility_model'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in a.budgets.split(',')]
    print('CONFIG', json.dumps(vars(a)), 'EPISODE_SOURCE', G.EPISODE_SOURCE, 'HORIZON', HZ, 'BANDS', BANDS, flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        eps = pool.map(G.episode_task, [(a.seed*100_000+i,) for i in range(a.episodes)])
        cand = {'hold': [], 'lowering': []}
        for seed, ep, labs in eps:
            for k, lab in enumerate(labs):
                if lab in ('hold', 'slip'):
                    cand['hold'].append((seed, ep, k, 'hold'))
                elif lab == 'lowering':
                    cand['lowering'].append((seed, ep, k, 'lowering'))
        rng = np.random.default_rng(a.seed)
        jobs = []
        for task, items in cand.items():
            sel = rng.choice(len(items), min(a.per_task, len(items)), replace=False)
            jobs += [(*items[i][:3], task, a.sigma, a.iters, budgets) for i in sel]
        print('STATES', {t: len(v) for t, v in cand.items()}, 'jobs', len(jobs), flush=True)
        R = []
        for i, r in enumerate(pool.imap_unordered(run_state, jobs)):
            R.append(r)
            if (i+1) % 10 == 0:
                print('PROGRESS %d/%d %.0fs' % (i+1, len(jobs), time.time()-t0), flush=True)
    (out/'utility_model.json').write_text(json.dumps(R, indent=1))
    methods = [f'model_{b:g}' for b in BANDS]+[f'cem_{B}' for B in budgets]
    for task in ('hold', 'lowering'):
        S = [r for r in R if r['task'] == task]
        if not S:
            continue
        none = np.array([r['none'] for r in S])
        print(f'TASK {task} n={len(S)} objective none: median {np.median(none):.3g}')
        for m in methods:
            rel = np.array([r[m] for r in S])/np.maximum(none, 1e-12)
            print(f'RESULT {task} {m}: median J/J_none {np.nanmedian(rel):.3f} (IQR {np.nanquantile(rel, .25):.3f}-'
                  f'{np.nanquantile(rel, .75):.3f}), improved >50% in {np.nanmean(rel < .5):.2f}, worse in {np.nanmean(rel > 1):.2f}', flush=True)
    print('ERRORS', sum('error' in r for r in R), 'DONE %.0fs -> %s' % (time.time()-t0, out), flush=True)


if __name__ == '__main__':
    main()
