"""E7: can the structured model's backward pass produce the noise-averaged (expected) gradient?

For census states (scripted or M run 1454 episodes, labelled by situation), compare after h = 1 control step, for
outcomes object position and object velocity (world), over 12 action channels (6 finger commands, palm linear and
angular velocity offsets):

  mj_local   MuJoCo central difference, delta 1e-6, converged solver (= the structured model's exact gradient,
             verified separately: palm cosine 1.00)
  mj_smooth  MuJoCo noise-averaged derivative at sigma low / mid (common random numbers, 2 x 16 draws)
  model_s    the structured model's autograd gradient with the backward-only soft servo band s (forward unchanged;
             s = 0 is the exact model)

Metric per (situation, channel group, band s): median cosine to mj_smooth (low, mid) and to mj_local, median
magnitude ratio |model| / |mj_smooth|.
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
from dextrah_lab.neurosim.structured.contact import ContactModel
from dextrah_lab.neurosim.structured.dynamics import SimState, StructuredSim
from dextrah_lab.neurosim.structured.rotations import axis_angle_quat, quat_mul
from dextrah_lab.neurosim.structured.scenes import tool_testbed
from dextrah_lab.neurosim.structured.spec import extract

BANDS = (0., .002, .005, .01, .02)
DT = 1/60
T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)


def mj_outcome(tb, st, palm, tgt, a, noise=None):
    tb.set_state(st)
    nz = np.zeros(12) if noise is None else noise
    a = a+nz
    nxt = G.perturbed_palm(palm, 1, a[6:9], a[9:12], DT)
    tb.step(tgt+G.cmd_to_targets(a[:6]), palm[0], nxt)
    o, v = tb.object_qadr, tb.object_dadr
    return np.concatenate((tb.d.qpos[o:o+3], tb.d.qvel[v:v+3]))


def mj_jac(tb, st, palm, tgt, delta, noises):
    J = np.zeros((6, 12))
    for ch in range(12):
        e = np.zeros(12)
        e[ch] = delta[ch]
        J[:, ch] = np.mean([(mj_outcome(tb, st, palm, tgt, e, n)-mj_outcome(tb, st, palm, tgt, -e, n))/(2*delta[ch])
                            for n in noises], 0)
    return J


def model_jac(sim, s0, palm, tgt):
    kfac = torch.tensor([G.COUPLE_K.get(j, 1.) for js in G.COUPLING for j in js], dtype=torch.float64)
    group = [c for c, js in enumerate(G.COUPLING) for _ in js]

    def f(a):
        tg = T(tgt)[None]+(a[group]*kfac)[None]
        p1 = T(palm[1])[None]
        w = a[9:12]
        n = torch.sqrt((w*w).sum()+1e-30)
        dq = axis_angle_quat(w/n, n*DT)
        p1 = torch.cat((p1[:, :3]+a[6:9]*DT, quat_mul(dq, p1[0, 3:])[None]), -1)
        s = sim.step(s0, tg, T(palm[0])[None], p1)
        return torch.cat((s.obj_pos[0], s.obj_v[0]))
    return torch.autograd.functional.jacobian(f, torch.zeros(12, dtype=torch.float64)).numpy()


def task(args):
    seed, ep, k, lab, draws = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed*1000+k)
    tb = tool_testbed(ep.tool)
    sc = extract(tb)
    st, palm, tgt = ep.states[k], ep.palm[k:k+2], ep.targets[k]
    res = dict(seed=seed, k=k, label=lab)
    t0 = time.time()
    res['mj_local'] = mj_jac(tb, st, palm, tgt, np.full(12, 1e-6), [None])
    for name in ('low', 'mid'):
        sig = G.SIGMAS[name]
        sd = np.r_[[sig[0]]*6, [sig[1]]*3, [sig[2]]*3]
        noises = [rng.normal(size=12)*sd for _ in range(draws)]
        res['mj_'+name] = mj_jac(tb, st, palm, tgt, .2*sd, noises)
    tb.set_state(st)
    s0 = SimState.from_testbed(tb)
    cm = ContactModel(sc)
    for b in BANDS:
        sim = StructuredSim(sc, contact_model=cm, servo_band=b or None)
        try:
            res[f'model_{b:g}'] = model_jac(sim, s0, palm, tgt)
        except Exception as e:                      # report, keep the state
            res[f'model_{b:g}'] = np.full((6, 12), np.nan)
            res['error'] = repr(e)
    res['sec'] = time.time()-t0
    return res


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b/(na*nb)) if na > 1e-12 and nb > 1e-12 and np.isfinite(a).all() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=150)
    ap.add_argument('--per_situation', type=int, default=30)
    ap.add_argument('--draws', type=int, default=32)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.environ.get('OUT', 'outputs/neurosim/model_vs_sim'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print('CONFIG', json.dumps(vars(a)), 'EPISODE_SOURCE', G.EPISODE_SOURCE, 'BANDS', BANDS, flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        eps = pool.map(G.episode_task, [(a.seed*100_000+i,) for i in range(a.episodes)])
        cand = {}
        for seed, ep, labs in eps:
            for k, lab in enumerate(labs):
                if lab in ('closing', 'liftoff', 'hold', 'slip', 'lowering', 'touchdown', 'release'):
                    cand.setdefault(lab, []).append((seed, ep, k, lab))
        rng = np.random.default_rng(a.seed)
        jobs = []
        for lab, items in cand.items():
            sel = rng.choice(len(items), min(a.per_situation, len(items)), replace=False)
            jobs += [(*items[i], a.draws) for i in sel]
        print('STATES', {k: len(v) for k, v in cand.items()}, 'jobs', len(jobs), flush=True)
        R = []
        for i, r in enumerate(pool.imap_unordered(task, jobs)):
            R.append(r)
            if (i+1) % 20 == 0:
                print('PROGRESS %d/%d %.0fs' % (i+1, len(jobs), time.time()-t0), flush=True)
    np.save(out/'model_vs_sim_raw.npy', np.array(R, dtype=object), allow_pickle=True)
    groups = dict(finger=slice(0, 6), palm_lin=slice(6, 9), palm_ang=slice(9, 12))
    outs = dict(pos=slice(0, 3), vel=slice(3, 6))
    table = []
    for lab in sorted({r['label'] for r in R}):
        S = [r for r in R if r['label'] == lab]
        for oname, osl in outs.items():
            for g, gsl in groups.items():
                for b in BANDS:
                    key = f'model_{b:g}'
                    row = dict(situation=lab, outcome=oname, channel=g, band=b, n=len(S))
                    for ref in ('mj_local', 'mj_low', 'mj_mid'):
                        c = [cos(r[key][osl, gsl].ravel(), r[ref][osl, gsl].ravel()) for r in S]
                        rat = [np.linalg.norm(r[key][osl, gsl])/max(np.linalg.norm(r[ref][osl, gsl]), 1e-300) for r in S]
                        row[f'cos_{ref}'] = float(np.nanmedian(c)) if np.isfinite(c).any() else None
                        row[f'ratio_{ref}'] = float(np.nanmedian(rat))
                    table.append(row)
                    print('ROW', lab, oname, g, 'band %.3f' % b, 'n=%d' % len(S),
                          ' '.join(f"{ref}: cos {row[f'cos_{ref}'] if row[f'cos_{ref}'] is None else round(row[f'cos_{ref}'], 2)} "
                                   f"ratio {row[f'ratio_{ref}']:.2g}" for ref in ('mj_local', 'mj_low', 'mj_mid')), flush=True)
    (out/'model_vs_sim_table.json').write_text(json.dumps(table, indent=1))
    print('ERRORS', sum('error' in r for r in R), 'DONE %.0fs -> %s' % (time.time()-t0, out), flush=True)


if __name__ == '__main__':
    main()
