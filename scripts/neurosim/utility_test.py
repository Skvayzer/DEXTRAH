"""E3 utility test: does a gradient signal steer a short correction toward the right solution in the true simulator?

From held / slipping states (tool lifted) and lowering states (tool in hand, descending to the table) of the
scripted table episodes, a constant 12-D correction a = (6 finger command offsets, palm linear velocity offset,
palm angular velocity offset) over h = 8 control steps (0.13 s) is optimised on top of the logged controls, with an
equal budget of MuJoCo rollouts per state:

  none        a = 0
  grad_local  gradient descent with MuJoCo's local derivative (central difference, delta = 1e-6, noise-free
              objective): what an exact differentiable simulator / model gives
  grad_smooth gradient descent with the derivative of the noise-averaged objective (common-random-number central
              differences, noise sigma 'mid'): the randomized-smoothing signal
  cem         cross-entropy method (zeroth order)

Objectives (lower is better), always evaluated on 16 fresh noise draws (sigma 'mid'):
  hold      keep the tool fixed in the palm: 1e4 |dp_rel|^2 + 1e2 |drot_rel|^2 at h (m, rad)
  lowering  the same + soft touchdown: + 1e-2 * max(0, F_table - 2 m g)^2
Actions are box-bounded (finger ±0.2 rad, palm ±0.1 m/s, ±1 rad/s) and scaled per channel for the optimisers.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from gradient_census import FINGER_MODE, SIGMAS, episode_task, rollout
from dextrah_lab.neurosim.structured.scenes import tool_testbed

HZ = int(os.environ.get("UTILITY_HORIZON", 8))
SCALE = np.r_[[.05]*6, [.03]*3, [.2]*3]
BOUND = np.r_[[.2]*6, [.1]*3, [1.]*3]
if FINGER_MODE == 'intent':                  # finger channel = grip-intent fraction (gradient_census FINGER_MODE)
    SCALE[:6], BOUND[:6] = .3, 1.


def objective(tb, st, palm, tg, a, noise, top, task, mass):
    y = rollout(tb, st, palm, tg, a, noise, top)[HZ]
    J = 1e4*float(y['rel_pos'] @ y['rel_pos'])+1e2*float(y['rel_rot'] @ y['rel_rot'])
    if task == 'lowering':
        J += 1e-2*max(0., float(y['table_force'][0])-2*mass*9.81)**2
    return J


def run_state(args):
    seed, ep, k, task, budget, sigma_name = args
    rng = np.random.default_rng(seed*7919+k)
    tb = tool_testbed(ep.tool)
    top, mass = .8, ep.tool['mass']
    palm, tg, st = ep.palm[k:k+9], ep.targets[k:k+8], ep.states[k]          # rollout runs 8 steps; J at HZ
    sig = SIGMAS[sigma_name]
    sd = np.r_[[sig[0]]*6, [sig[1]]*3, [sig[2]]*3]
    noise_eval = [rng.normal(size=(8, 12))*sd for _ in range(16)]           # rollout uses the first HZ rows
    zero = np.zeros((8, 12))
    J = lambda a, nz: objective(tb, st, palm, tg, np.clip(a, -BOUND, BOUND), nz, top, task, mass)
    evaluate = lambda a: float(np.mean([J(a, nz) for nz in noise_eval]))
    res = dict(seed=seed, k=k, task=task, none=evaluate(np.zeros(12)))
    # gradient descent in scaled coordinates z = a / SCALE, backtracking on the optimiser's own objective
    for name in ('grad_local', 'grad_smooth'):
        used, z = 0, np.zeros(12)
        draws = [zero] if name == 'grad_local' else [rng.normal(size=(8, 12))*sd for _ in range(4)]
        f_opt = lambda zz: float(np.mean([J(zz*SCALE, nz) for nz in draws]))
        f = f_opt(z)
        used += len(draws)
        step = 1.
        while used+25*len(draws) <= budget:
            g = np.zeros(12)
            dz = 1e-6/SCALE if name == 'grad_local' else np.full(12, .2)*sd/SCALE
            for i in range(12):
                e = np.zeros(12)
                e[i] = dz[i]
                g[i] = (f_opt(z+e)-f_opt(z-e))/(2*dz[i])
            used += 24*len(draws)
            gn = np.linalg.norm(g)
            if not np.isfinite(gn) or gn == 0:
                break
            d = -g/gn
            accepted = False
            for _ in range(6):                       # backtracking line search on the step length
                z_new = np.clip(z+step*d, -BOUND/SCALE, BOUND/SCALE)
                f_new = f_opt(z_new)
                used += len(draws)
                if f_new < f:
                    z, f, accepted = z_new, f_new, True
                    step *= 1.5
                    break
                step *= .3
            if not accepted:
                step = max(step, 1e-3)
        res[name] = evaluate(z*SCALE)
        res[name+'_used'] = used
    # CEM on the noisy objective (1 draw per sample)
    mu, sdv, used = np.zeros(12), np.ones(12), 0
    pop, elite = 24, 6
    best = (np.inf, np.zeros(12))
    while used+pop <= budget:
        Z = np.clip(mu+sdv*rng.normal(size=(pop, 12)), -BOUND/SCALE, BOUND/SCALE)
        nz = rng.normal(size=(8, 12))*sd
        fs = np.array([J(z*SCALE, nz) for z in Z])
        used += pop
        idx = np.argsort(fs)[:elite]
        mu, sdv = Z[idx].mean(0), np.maximum(Z[idx].std(0), .05)
        if fs[idx[0]] < best[0]:
            best = (fs[idx[0]], Z[idx[0]])
    res['cem'] = evaluate(mu*SCALE)
    res['cem_used'] = used
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=150)
    ap.add_argument('--per_task', type=int, default=40)
    ap.add_argument('--budget', type=int, default=600)
    ap.add_argument('--sigma', default='mid')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.environ.get('OUT', 'outputs/neurosim/utility'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print('CONFIG', json.dumps(vars(a)), 'FINGER_MODE', FINGER_MODE, 'TESTBED_SUBSTEPS', os.environ.get('TESTBED_SUBSTEPS', 2),
          'EPISODE_SOURCE', os.environ.get('EPISODE_SOURCE', 'scripted'), 'HORIZON', HZ, flush=True)
    t0 = time.time()
    from gradient_census import label
    with Pool(a.workers) as pool:
        eps = pool.map(episode_task, [(a.seed*100_000+i,) for i in range(a.episodes)])
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
            jobs += [(*items[i][:3], task, a.budget, a.sigma) for i in sel]
        print('STATES', {t: len(v) for t, v in cand.items()}, 'jobs', len(jobs), flush=True)
        results = []
        for i, r in enumerate(pool.imap_unordered(run_state, jobs)):
            results.append(r)
            if (i+1) % 10 == 0:
                print('PROGRESS %d/%d %.0fs' % (i+1, len(jobs), time.time()-t0), flush=True)
    (out/'utility.json').write_text(json.dumps(results, indent=1))
    for task in ('hold', 'lowering'):
        R = [r for r in results if r['task'] == task]
        if not R:
            continue
        none = np.array([r['none'] for r in R])
        print(f'TASK {task} n={len(R)} objective none: median {np.median(none):.3g}')
        for m in ('grad_local', 'grad_smooth', 'cem'):
            v = np.array([r[m] for r in R])
            rel = v/np.maximum(none, 1e-12)
            print(f'RESULT {task} {m}: median J/J_none {np.median(rel):.3f} (IQR {np.quantile(rel, .25):.3f}-{np.quantile(rel, .75):.3f}), '
                  f'improved >50% in {np.mean(rel < .5):.2f}, worse in {np.mean(rel > 1):.2f}, '
                  f'rollouts {np.mean([r[m+"_used"] for r in R]):.0f}', flush=True)
    print('DONE %.0fs -> %s' % (time.time()-t0, out), flush=True)


if __name__ == '__main__':
    main()
