"""E19: does a grasp's sensitivity to action noise (the quantity the robustness gradient lowers) predict later slip of the
tool in the hand better than its deterministic drift or a classic grasp-quality proxy?

M 1454 holds rebuilt in the testbed (m_episode: settle 30 steps, carry 40 steps of random palm motion, lower, release).
At the first carry step k0 = 30: sensitivity S = mean J over 16 noise draws at h = 2 (sigma mid, as E17), drift
D = J without noise, capacity C = sum_contacts mu * f_n / (m g), grip force. Outcome (disjoint from the h = 2 window):
the tool's motion in the palm frame from step k0 + 2 to the end of the carry, position (mm) and rotation (deg), and loss
(tool more than 2 cm from its hold, or below the table top). Spearman rank correlations and AUC for loss.
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

import gradient_census as G
from utility_test import objective
from dextrah_lab.neurosim.reference.held_states import mat

K0, CARRY = 30, 40
SIG = G.SIGMAS['mid']
SD = np.r_[[SIG[0]]*6, [SIG[1]]*3, [SIG[2]]*3]


def rel(ep, k):
    R = mat(ep.palm[k][3:])
    return R.T @ (ep.obj[k][:3]-ep.palm[k][:3]), R.T @ mat(ep.obj[k][3:])


def task(seed):
    from dextrah_lab.neurosim.reference.m_holds import m_episode
    rng = np.random.default_rng(seed)
    ep, tb = m_episode(rng)
    if len(ep.phase) <= K0+CARRY or ep.phase[K0] != 'hold':
        return None
    palm, tg, st = ep.palm[K0:K0+9], ep.targets[K0:K0+8], ep.states[K0]
    mass = ep.tool['mass']
    S = float(np.mean([objective(tb, st, palm, tg, np.zeros(12), rng.normal(size=(8, 12))*SD, .8, 'hold', mass) for _ in range(16)]))
    D = objective(tb, st, palm, tg, np.zeros(12), np.zeros((8, 12)), .8, 'hold', mass)
    tb.set_state(st)
    mujoco.mj_forward(tb.m, tb.d)
    grip, margin, table, util, cap = G.contact_terms(tb)
    p1, R1 = rel(ep, K0+2)
    p2, R2 = rel(ep, K0+CARRY)
    dp = float(np.linalg.norm(p2-p1)*1e3)
    dr = float(np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2)-1)/2, -1, 1))))
    lost = bool(dp > 20 or ep.obj[K0+CARRY][2] < ep.obj[K0][2]-.05)
    return dict(seed=seed, handle=ep.tool['handle'], S=S, D=D, capacity=float(cap/(mass*9.81)), grip=float(grip),
                util=float(util), slip_mm=dp, slip_deg=dr, lost=lost)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(np.mean([(p > neg).mean()+.5*(p == neg).mean() for p in pos]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=400)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out', default=os.environ.get('OUT', 'outputs/neurosim/sensitivity_slip'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        R = [r for r in pool.map(task, [a.seed*100_000+i for i in range(a.episodes)]) if r]
    (out/'sensitivity_slip.json').write_text(json.dumps(R, indent=1))
    arr = lambda k: np.array([r[k] for r in R], float)
    lost = np.array([r['lost'] for r in R])
    print('EPISODES', len(R), 'lost during carry', int(lost.sum()), 'slip_mm median %.2f slip_deg median %.2f' % (np.median(arr('slip_mm')), np.median(arr('slip_deg'))))
    preds = dict(sensitivity=arr('S'), drift=arr('D'), excess_sensitivity=arr('S')-arr('D'), neg_capacity=-arr('capacity'),
                 neg_grip=-arr('grip'), utilization=arr('util'))
    for name, x in preds.items():
        print(f'PREDICT {name:20s} spearman slip_mm {spearman(x, arr("slip_mm")):+.3f}  slip_deg {spearman(x, arr("slip_deg")):+.3f}  '
              f'AUC lost {auc(x, lost):.3f}', flush=True)
    # within-family (tool type) correlations, to rule out a tool-type confound
    fams = sorted({r['handle'] for r in R})
    for name in ('sensitivity', 'drift', 'neg_capacity'):
        cs = []
        for f in fams:
            m = np.array([r['handle'] == f for r in R])
            if m.sum() >= 15:
                cs.append(spearman(preds[name][m], arr('slip_deg')[m]))
        print(f'WITHIN-FAMILY {name}: spearman slip_deg median over {len(cs)} families {np.median(cs):+.3f}', flush=True)
    print('DONE ->', out)


if __name__ == '__main__':
    main()
