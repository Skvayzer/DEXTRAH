"""E17: why does the (aligned) model gradient not steer, and does optimising the model's EXPECTED objective fix it?

Hypothesis. The evaluated cost is E_eps J(a + eps) ~ J(a) + sum_i sigma_i^2 |dg/da_i|^2 (J = |g|^2): a bias part (the
deterministic drift) and a variance part (the grasp's sensitivity to action noise). If the variance part dominates in
holds, a deterministic model gradient - even an exact one - can only remove a small part of the cost; the missing
signal is the gradient of the sensitivity (how the correction makes the grasp robust), which the model supplies when
its objective is the noise expectation with the evaluation's noise model.

Per state (M 1454 hold / lowering states, as utility_model):
  MuJoCo diagnosis: J_noisy(0) (16 draws) and J_det(0) (no noise) -> bias fraction; for every correction a* also J_det(a*).
  Model variants (structured model, Adam 30 iterations on z = a / SCALE, no MuJoCo):
    det          deterministic objective, now with the lowering table-force term (utility_model omitted it)
    old_noisy    utility_model's model_pathband_mid (band 0.02, one constant 12-D noise draw per iteration)
    step_K8      per-control-step noise exactly as the evaluation (fingers + palm velocity, integrated), 8 samples per
                 iteration (4 antithetic pairs, batched in one model rollout), band 0
    step_K8_band the same with backward servo band 0.02
  Then CEM-24 seeded at step_K8 (and at old_noisy, as E12/E15) and plain CEM-24.
"""
import argparse
import json
import os
import time
from dataclasses import fields, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

import gradient_census as G
from utility_model import HZ, DT, T
from utility_test import BOUND, SCALE, objective
from dextrah_lab.neurosim.structured.contact import ContactModel
from dextrah_lab.neurosim.structured.dynamics import SimState, StructuredSim
from dextrah_lab.neurosim.structured.rotations import axis_angle_quat, quat_mul, quat_to_mat
from dextrah_lab.neurosim.structured.scenes import tool_testbed
from dextrah_lab.neurosim.structured.spec import extract

GROUP = [c for c, js in enumerate(G.COUPLING) for _ in js]
KFAC = torch.tensor([G.COUPLE_K.get(j, 1.) for js in G.COUPLING for j in js], dtype=torch.float64)


def _aa(w):
    n = torch.sqrt((w*w).sum(-1)+1e-30)
    return w/n[..., None], n


def model_cost(sim, s0, palm, tg, a, noise, task, mass, table_kind):
    """Mean J over the batch of noise draws noise (K, HZ, 12) for correction a (12,), same J as utility_test.objective."""
    K = noise.shape[0]
    s = replace(s0, **{f.name: getattr(s0, f.name).expand(K, *getattr(s0, f.name).shape[1:]) for f in fields(s0)})
    R0 = quat_to_mat(T(palm[0][3:]))
    rel_p0 = R0.T @ (s0.obj_pos[0]-T(palm[0][:3]))
    rel_R0 = R0.T @ quat_to_mat(s0.obj_quat[0])
    prev = T(palm[0])[None].expand(K, 7)
    acc_p = torch.zeros(K, 3, dtype=torch.float64)
    acc_q = torch.tensor([1., 0., 0., 0.], dtype=torch.float64).expand(K, 4)
    ax, ang = _aa(a[9:12])
    for k in range(HZ):
        p = T(palm[k+1])
        dq = axis_angle_quat(ax, ang*DT*(k+1))
        nq = quat_mul(dq, p[3:])
        nx, nn = _aa(noise[:, k, 9:12])
        acc_q = quat_mul(axis_angle_quat(nx, nn*DT), acc_q)
        acc_p = acc_p+noise[:, k, 6:9]*DT
        nxt = torch.cat((p[:3]+a[6:9]*DT*(k+1)+acc_p, quat_mul(acc_q, nq.expand(K, 4))), -1)
        u = a[:6]+noise[:, k, :6]
        s = sim.step(s, T(tg[k])[None]+u[:, GROUP]*KFAC, prev, nxt)
        prev = nxt
    Rp = quat_to_mat(prev[:, 3:])
    dp = (Rp.transpose(-1, -2) @ (s.obj_pos-prev[:, :3])[..., None]).squeeze(-1)-rel_p0
    Rr = rel_R0.T @ (Rp.transpose(-1, -2) @ quat_to_mat(s.obj_quat))
    tr = Rr.diagonal(dim1=-2, dim2=-1).sum(-1)
    th = torch.acos(((tr-1)/2).clamp(-1+1e-12, 1-1e-12))
    J = 1e4*(dp*dp).sum(-1)+1e2*th*th
    if task == 'lowering' and table_kind is not None and 'elliptic' in sim.last['forces']:
        fn = (sim.last['forces']['elliptic'][..., 0]*table_kind).sum(-1)
        J = J+1e-2*torch.relu(fn-2*mass*9.81)**2
    return J.mean()


def run_state(args):
    seed, ep, k, task, iters = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed*7919+k)
    tb = tool_testbed(ep.tool)
    sc = extract(tb)
    top, mass = .8, ep.tool['mass']
    palm, tg, st = ep.palm[k:k+9], ep.targets[k:k+8], ep.states[k]
    sig = G.SIGMAS['mid']
    sd = np.r_[[sig[0]]*6, [sig[1]]*3, [sig[2]]*3]
    noise_eval = [rng.normal(size=(8, 12))*sd for _ in range(16)]
    J = lambda a, nz: objective(tb, st, palm, tg, np.clip(a, -BOUND, BOUND), nz, top, task, mass)
    evaluate = lambda a: float(np.mean([J(a, nz) for nz in noise_eval]))
    det = lambda a: J(a, np.zeros((8, 12)))
    res = dict(seed=seed, k=k, task=task, none=evaluate(np.zeros(12)), none_det=det(np.zeros(12)))
    tb.set_state(st)
    s0 = SimState.from_testbed(tb)
    cm = ContactModel(sc)
    # fidelity of the model's EXPECTED cost and its gradient at a = 0, common random numbers with MuJoCo
    try:
        tk0 = None
        sim0 = StructuredSim(sc, contact_model=cm)
        if task == 'lowering':
            with torch.no_grad():
                model_cost(sim0, s0, palm, tg, torch.zeros(12, dtype=torch.float64), torch.zeros(1, HZ, 12, dtype=torch.float64), 'hold', mass, None)
            tk0 = torch.tensor([kd == 'object_table' for kd in sim0.last['contacts'].kind], dtype=torch.float64)
        with torch.no_grad():
            res['model_none'] = float(model_cost(sim0, s0, palm, tg, torch.zeros(12, dtype=torch.float64),
                                                 T(np.stack(noise_eval)[:, :HZ]), task, mass, tk0))
            res['model_none_det'] = float(model_cost(sim0, s0, palm, tg, torch.zeros(12, dtype=torch.float64),
                                                     torch.zeros(1, HZ, 12, dtype=torch.float64), task, mass, tk0))
        crn = noise_eval[:8]
        g_mj = np.zeros(12)
        for c in range(12):
            e = np.zeros(12)
            e[c] = .2*sd[c]
            g_mj[c] = np.mean([J(e, nz)-J(-e, nz) for nz in crn])/(2*e[c])*SCALE[c]
        for gname, band in (('grad_cos_band0', 0.), ('grad_cos_band02', .02)):
            simg = StructuredSim(sc, contact_model=cm, servo_band=band or None)
            z = torch.zeros(12, dtype=torch.float64, requires_grad=True)
            model_cost(simg, s0, palm, tg, z*T(SCALE), T(np.stack(crn)[:, :HZ]), task, mass, tk0).backward()
            gm = z.grad.numpy()
            for part, sl in (('', slice(0, 12)), ('_finger', slice(0, 6)), ('_palm', slice(6, 12))):
                na, nb = np.linalg.norm(gm[sl]), np.linalg.norm(g_mj[sl])
                res[gname+part] = float(gm[sl] @ g_mj[sl]/(na*nb)) if na > 0 and nb > 0 else np.nan
            res[gname.replace('cos', 'ratio')] = float(np.linalg.norm(gm)/max(np.linalg.norm(g_mj), 1e-12))
    except Exception as e:
        res['fid_error'] = repr(e)
    zs = {}
    sc_t, bd, sd_t = T(SCALE), T(BOUND/SCALE), T(sd)
    for name, band, mode in (('det', 0., 'det'), ('old_noisy', .02, 'const1'), ('step_K8', 0., 'step8'), ('step_K8_band', .02, 'step8')):
        t0 = time.time()
        sim = StructuredSim(sc, contact_model=cm, servo_band=band or None)
        z = torch.zeros(12, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([z], lr=.1)
        try:
            # identify object-table contact slots once (the slot layout is fixed per scene)
            table_kind = None
            for _ in range(iters):
                opt.zero_grad()
                if mode == 'det':
                    nz = torch.zeros(1, HZ, 12, dtype=torch.float64)
                elif mode == 'const1':                       # utility_model: one constant offset draw, fingers + palm
                    nz = (T(rng.normal(size=12))*sd_t).expand(1, HZ, 12)
                else:
                    h = T(rng.normal(size=(4, HZ, 12)))*sd_t
                    nz = torch.cat((h, -h), 0)
                if table_kind is None and task == 'lowering':
                    with torch.no_grad():
                        model_cost(sim, s0, palm, tg, z*sc_t, nz[:1], 'hold', mass, None)
                    c = sim.last['contacts']
                    table_kind = torch.tensor([kd == 'object_table' for kd in c.kind], dtype=torch.float64)
                f = model_cost(sim, s0, palm, tg, z*sc_t, nz, task, mass, table_kind)
                f.backward()
                opt.step()
                with torch.no_grad():
                    z.copy_(torch.maximum(torch.minimum(z, bd), -bd))
            zm = z.detach().numpy().copy()
            zs[name] = zm
            res[name] = evaluate(zm*SCALE)
            res[name+'_det'] = det(zm*SCALE)
            res[name+'_sec'] = time.time()-t0
        except Exception as e:
            res[name] = np.nan
            res['error'] = repr(e)

    def cem(mu0, B):
        mu, sdv, used, pop, elite = mu0.copy(), np.ones(12), 0, 12, 3
        while used+pop <= B:
            Z = np.clip(mu+sdv*rng.normal(size=(pop, 12)), -BOUND/SCALE, BOUND/SCALE)
            nz = rng.normal(size=(8, 12))*sd
            fs = np.array([J(zz*SCALE, nz) for zz in Z])
            used += pop
            idx = np.argsort(fs)[:elite]
            mu, sdv = Z[idx].mean(0), np.maximum(Z[idx].std(0), .05)
        return evaluate(mu*SCALE)
    for name in ('old_noisy', 'step_K8'):
        if name in zs:
            res[name+'_cem24'] = cem(zs[name], 24)
    res['cem_24'] = cem(np.zeros(12), 24)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=300)
    ap.add_argument('--per_task', type=int, default=40)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=os.environ.get('OUT', 'outputs/neurosim/robust_steering'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print('CONFIG', json.dumps(vars(a)), 'EPISODE_SOURCE', G.EPISODE_SOURCE, 'HORIZON', HZ, flush=True)
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
            jobs += [(*items[i][:3], task, a.iters) for i in sel]
        print('STATES', {t: len(v) for t, v in cand.items()}, 'jobs', len(jobs), flush=True)
        R = []
        for i, r in enumerate(pool.imap_unordered(run_state, jobs)):
            R.append(r)
            if (i+1) % 10 == 0:
                print('PROGRESS %d/%d %.0fs' % (i+1, len(jobs), time.time()-t0), flush=True)
    (out/'robust_steering.json').write_text(json.dumps(R, indent=1))
    variants = ('det', 'old_noisy', 'step_K8', 'step_K8_band')
    for task in ('hold', 'lowering'):
        S = [r for r in R if r['task'] == task]
        if not S:
            continue
        none = np.array([r['none'] for r in S])
        none_det = np.array([r['none_det'] for r in S])
        fid = lambda key: np.nanmedian([r.get(key, np.nan) for r in S])
        print(f'FIDELITY {task}: model/MuJoCo expected cost at 0 median {np.nanmedian([r.get("model_none", np.nan)/r["none"] for r in S]):.3f}, '
              f'deterministic {np.nanmedian([r.get("model_none_det", np.nan)/max(r["none_det"], 1e-12) for r in S]):.3f}; expected-cost gradient cos '
              f'band0 {fid("grad_cos_band0"):.3f} (finger {fid("grad_cos_band0_finger"):.3f}, palm {fid("grad_cos_band0_palm"):.3f}, |g| ratio {fid("grad_ratio_band0"):.2f}), '
              f'band.02 {fid("grad_cos_band02"):.3f} (finger {fid("grad_cos_band02_finger"):.3f}, palm {fid("grad_cos_band02_palm"):.3f}, |g| ratio {fid("grad_ratio_band02"):.2f})', flush=True)
        print(f'TASK {task} n={len(S)} J_none median {np.median(none):.3g}; bias fraction J_det(0)/J_noisy(0) median '
              f'{np.median(none_det/np.maximum(none, 1e-12)):.3f} (IQR {np.quantile(none_det/none, .25):.3f}-{np.quantile(none_det/none, .75):.3f})')
        for m in [*variants, 'old_noisy_cem24', 'step_K8_cem24', 'cem_24']:
            rel = np.array([r.get(m, np.nan) for r in S])/np.maximum(none, 1e-12)
            line = (f'RESULT {task} {m}: median J/J_none {np.nanmedian(rel):.3f} (IQR {np.nanquantile(rel, .25):.3f}-'
                    f'{np.nanquantile(rel, .75):.3f}), improved >50% in {np.nanmean(rel < .5):.2f}, worse in {np.nanmean(rel > 1):.2f}')
            if m in variants:
                reld = np.array([r.get(m+'_det', np.nan) for r in S])/np.maximum(none_det, 1e-12)
                line += f'; deterministic J_det/J_det_none median {np.nanmedian(reld):.3f}, model sec {np.nanmedian([r.get(m+"_sec", np.nan) for r in S]):.1f}'
            print(line, flush=True)
    print('ERRORS', sum('error' in r for r in R), [r['error'] for r in R if 'error' in r][:3], 'DONE %.0fs -> %s' % (time.time()-t0, out), flush=True)


if __name__ == '__main__':
    main()
