"""E20: can structured-model gradients stop an in-hand creep when used closed loop (MPC over the carry)?

E19: the tool's later slip in noise-free carrying is predicted by its current drift (Spearman +0.61, AUC lost 0.95), not
by its noise sensitivity; but the h = 2 drift is mostly inertial (a slide already in progress) and a single finger
offset cannot remove it. Here the fingers are corrected at every control step of the carry (receding horizon), which can
decelerate a slide over many steps.

Episodes: M 1454 holds (m_episode), carry phase k = 30 .. 70 (random palm motion, logged palm path kept). Each step: a
6-D finger command offset z (a = z * SCALE[:6], |a| <= BOUND[:6]) is optimised on the structured model over h = HZ steps
from the current MuJoCo state (Adam, ITERS iterations, warm start from the previous step's z), applied for ONE MuJoCo
step. Objectives (tool in the palm frame): 'disp' = 1e4 |dp_h|^2 + 1e2 |dth_h|^2 (E17's), 'vel' = the same on the last
step's motion (relative velocity at the end of the horizon x dt). Modes: 'det' no noise; 'exp' 8 antithetic per-step
noise draws (fingers only). Reference: per-step CEM on MuJoCo (24 deterministic rollouts, objective 'vel').
Outcome: the tool's motion in the palm frame over the carry (mm, deg), loss (> 2 cm), mean finger offset.
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
from utility_model import DT, T
from utility_test import BOUND, SCALE
from dextrah_lab.neurosim.reference.held_states import mat
from dextrah_lab.neurosim.structured.contact import ContactModel
from dextrah_lab.neurosim.structured.dynamics import SimState, StructuredSim
from dextrah_lab.neurosim.structured.rotations import quat_to_mat
from dextrah_lab.neurosim.structured.scenes import tool_testbed
from dextrah_lab.neurosim.structured.spec import extract

HZ = int(os.environ.get('UTILITY_HORIZON', 2))
ITERS = int(os.environ.get('MPC_ITERS', 4))
RATE = float(os.environ.get('MPC_RATE', 1.))       # penalty on |z_k - z_{k-1}|^2 (z = a / SCALE)
K0, CARRY = 30, 40
GROUP = [c for c, js in enumerate(G.COUPLING) for _ in js]
KFAC = torch.tensor([G.COUPLE_K.get(j, 1.) for js in G.COUPLING for j in js], dtype=torch.float64)
SDF = G.SIGMAS['mid'][0]


def rel_terms(Rp, pp, op, oq, rel_p0, rel_R0):
    dp = (Rp.transpose(-1, -2) @ (op-pp)[..., None]).squeeze(-1)-rel_p0
    Rr = rel_R0.T @ (Rp.transpose(-1, -2) @ quat_to_mat(oq))
    return dp, Rr


def model_J(sim, s0, palm, tg, a6, noise, objective, ref=None):
    """palm (HZ+1, 7) logged, tg (HZ, 11), a6 (6,), noise (K, HZ, 6)."""
    K = noise.shape[0]
    s = replace(s0, **{f.name: getattr(s0, f.name).expand(K, *getattr(s0, f.name).shape[1:]) for f in fields(s0)})
    R0 = quat_to_mat(T(palm[0][3:]))
    rel_p0 = R0.T @ (s0.obj_pos[0]-T(palm[0][:3]))
    rel_R0 = R0.T @ quat_to_mat(s0.obj_quat[0])
    prev = T(palm[0])[None].expand(K, 7)
    hist = []
    for k in range(HZ):
        nxt = T(palm[k+1])[None].expand(K, 7)
        u = a6+noise[:, k]
        s = sim.step(s, T(tg[k])[None]+u[:, GROUP]*KFAC, prev, nxt)
        prev = nxt
        hist.append(rel_terms(quat_to_mat(nxt[:, 3:]), nxt[:, :3], s.obj_pos, s.obj_quat, rel_p0, rel_R0))
    if objective == 'track':                 # pose error to the reference (palm frame), not motion in the window
        dp = hist[-1][0]+rel_p0-T(ref[0])
        Rr = T(ref[1]).T @ (rel_R0 @ hist[-1][1])
    elif objective == 'disp' or HZ == 1:
        dp, Rr = hist[-1]
    else:
        (p1, R1), (p2, R2) = hist[-2], hist[-1]
        dp, Rr = p2-p1, R1.transpose(-1, -2) @ R2
    tr = Rr.diagonal(dim1=-2, dim2=-1).sum(-1)
    th = torch.acos(((tr-1)/2).clamp(-1+1e-12, 1-1e-12))
    return (1e4*(dp*dp).sum(-1)+1e2*th*th).mean()


def mj_J(tb, st, palm, tg, a6, objective, ref=None):
    tb.set_state(st)
    o = tb.object_qadr
    R0 = mat(palm[0][3:])
    p0 = R0.T @ (tb.d.qpos[o:o+3]-palm[0][:3])
    Rr0 = R0.T @ mat(tb.d.qpos[o+3:o+7])
    hist = []
    for k in range(HZ):
        tb.step(tg[k]+G.cmd_to_targets(a6), palm[k], palm[k+1])
        R = mat(palm[k+1][3:])
        hist.append((R.T @ (tb.d.qpos[o:o+3]-palm[k+1][:3])-p0, Rr0.T @ (R.T @ mat(tb.d.qpos[o+3:o+7]))))
    if objective == 'track':
        dp, Rr = hist[-1][0]+p0-ref[0], ref[1].T @ (Rr0 @ hist[-1][1])
    elif objective == 'disp' or HZ == 1:
        dp, Rr = hist[-1]
    else:
        dp, Rr = hist[-1][0]-hist[-2][0], hist[-2][1].T @ hist[-1][1]
    th = np.arccos(np.clip((np.trace(Rr)-1)/2, -1, 1))
    return 1e4*float(dp @ dp)+1e2*th*th


def run_episode(args):
    seed, arm = args
    torch.set_num_threads(1)
    from dextrah_lab.neurosim.reference.m_holds import m_episode
    rng = np.random.default_rng(seed)
    ep, _ = m_episode(rng)
    if len(ep.phase) <= K0+CARRY+HZ or ep.phase[K0] != 'hold':
        return None
    tb = tool_testbed(ep.tool)
    sc = extract(tb)
    cm = ContactModel(sc)
    sim = StructuredSim(sc, contact_model=cm)
    o = tb.object_qadr
    tb.set_state(ep.states[K0])
    rel = lambda k: (mat(ep.palm[k][3:]).T @ (tb.d.qpos[o:o+3]-ep.palm[k][:3]), mat(ep.palm[k][3:]).T @ mat(tb.d.qpos[o+3:o+7]))
    p_start, R_start = rel(K0)
    ref = (p_start, R_start)
    z = np.zeros(6)
    bd, sc6 = BOUND[:6]/SCALE[:6], SCALE[:6]
    offsets, t0 = [], time.time()
    for k in range(K0, K0+CARRY):
        st = tb.get_state()
        palm, tg = ep.palm[k:k+HZ+1], ep.targets[k:k+HZ]
        if arm != 'none':
            if arm.startswith('mj_cem'):
                objective = arm.split('_')[2] if arm.count('_') == 2 else 'vel'
                mu, sd = z.copy(), np.full(6, .5)
                for _ in range(2):
                    Z = np.clip(mu+sd*rng.normal(size=(12, 6)), -bd, bd)
                    Z[0] = mu
                    fs = np.array([mj_J(tb, st, palm, tg, zz*sc6, objective, ref)+RATE*float((zz-z) @ (zz-z)) for zz in Z])
                    idx = np.argsort(fs)[:3]
                    mu, sd = Z[idx].mean(0), np.maximum(Z[idx].std(0), .05)
                z = Z[idx[0]]
            else:
                objective, mode = arm.split('_')
                tb.set_state(st)
                s0 = SimState.from_testbed(tb)
                zt = torch.tensor(z, dtype=torch.float64, requires_grad=True)
                opt = torch.optim.Adam([zt], lr=.1)
                for _ in range(ITERS):
                    if mode == 'exp':
                        h = T(rng.normal(size=(4, HZ, 6))*SDF)
                        nz = torch.cat((h, -h), 0)
                    else:
                        nz = torch.zeros(1, HZ, 6, dtype=torch.float64)
                    opt.zero_grad()
                    (model_J(sim, s0, palm, tg, zt*T(sc6), nz, objective, ref)+RATE*((zt-T(z))**2).sum()).backward()
                    opt.step()
                    with torch.no_grad():
                        zt.copy_(torch.maximum(torch.minimum(zt, T(bd)), -T(bd)))
                z = zt.detach().numpy().copy()
        tb.set_state(st)
        tb.step(ep.targets[k]+G.cmd_to_targets(z*sc6), ep.palm[k], ep.palm[k+1])
        offsets.append(z*sc6)
    p_end, R_end = rel(K0+CARRY)
    dp = float(np.linalg.norm(p_end-p_start)*1e3)
    dr = float(np.degrees(np.arccos(np.clip((np.trace(R_start.T @ R_end)-1)/2, -1, 1))))
    off = np.array(offsets)
    return dict(seed=seed, arm=arm, handle=ep.tool['handle'], slip_mm=dp, slip_deg=dr, lost=bool(dp > 20),
                mean_offset=off.mean(0).tolist(), mean_abs_offset=float(np.abs(off).mean()), sec=time.time()-t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=48)
    ap.add_argument('--arms', default='none,track_det,track_exp,mj_cem_track,mj_cem_vel')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--out', default=os.environ.get('OUT', 'outputs/neurosim/mpc_slip'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = a.arms.split(',')
    print('CONFIG', json.dumps(vars(a)), 'HORIZON', HZ, 'ITERS', ITERS, 'RATE', RATE, flush=True)
    jobs = [(a.seed*100_000+i, arm) for i in range(a.episodes) for arm in arms]
    R = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_episode, jobs)):
            if r:
                R.append(r)
            if (i+1) % 24 == 0:
                print('PROGRESS', i+1, len(jobs), flush=True)
    (out/'mpc_slip.json').write_text(json.dumps(R, indent=1))
    by = {(r['seed'], r['arm']): r for r in R}
    seeds = sorted({r['seed'] for r in R if all((r['seed'], m) in by for m in arms)})
    base_mm = np.array([by[(s, 'none')]['slip_mm'] for s in seeds])
    base_deg = np.array([by[(s, 'none')]['slip_deg'] for s in seeds])
    print('EPISODES', len(seeds), 'none: slip median %.2f mm %.2f deg' % (np.median(base_mm), np.median(base_deg)))
    for m in arms:
        mm = np.array([by[(s, m)]['slip_mm'] for s in seeds])
        dg = np.array([by[(s, m)]['slip_deg'] for s in seeds])
        print(f'RESULT {m:9s} slip median {np.median(mm):.2f} mm {np.median(dg):.2f} deg; ratio to none median mm '
              f'{np.median(mm/np.maximum(base_mm, 1e-9)):.3f} deg {np.median(dg/np.maximum(base_deg, 1e-9)):.3f}; '
              f'better in {np.mean(dg < base_deg):.2f}; lost {np.mean([by[(s, m)]["lost"] for s in seeds]):.3f}; '
              f'|offset| {np.mean([by[(s, m)]["mean_abs_offset"] for s in seeds]):.3f} rad; mean offset per group '
              f'{np.round(np.mean([by[(s, m)]["mean_offset"] for s in seeds], 0), 3).tolist()}; sec {np.median([by[(s, m)]["sec"] for s in seeds]):.0f}', flush=True)
    print('DONE ->', out)


if __name__ == '__main__':
    main()
