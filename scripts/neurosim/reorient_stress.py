"""E21: does fast palm rotation (reorientation) reproduce M's in-task drop pattern by tool type in the testbed?

M's mjlab evaluation (E14b/E14d records): drops are 41% (trained servo) / 32% (deadbeat) of episodes, ~3% per goal
attempt, ordered marker > screwdriver > eraser > brush > spatula > hammer. The scripted carries used so far (E19/E20,
angular speed ~0.2 rad/s) almost never lose the tool. Here M's holds of each family are carried for 1 s with smoothed
random palm rotation at increasing rates (m_episode carry_rate r: stationary angular speed ~0.23 r rad/s).
Per episode: lost (tool > 2 cm from its hold in the palm frame, or contact with all fingers lost), rotation slip (deg),
both at the end of the carry. Per family and rate: loss fraction, slip median.
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from dextrah_lab.neurosim.reference.held_states import mat

K0 = 30


def task(args):
    seed, i, rate, steps = args
    from dextrah_lab.neurosim.reference.m_holds import cached_pool, m_episode
    rng = np.random.default_rng(seed)
    ep, tb = m_episode(rng, i=i, carry_steps=steps, carry_rate=rate)
    k1 = K0+steps
    if len(ep.phase) <= k1 or ep.phase[K0] != 'hold':
        return None
    rel = lambda k: (mat(ep.palm[k][3:]).T @ (ep.obj[k][:3]-ep.palm[k][:3]), mat(ep.palm[k][3:]).T @ mat(ep.obj[k][3:]))
    p0, R0 = rel(K0)
    p1, R1 = rel(k1)
    dp = float(np.linalg.norm(p1-p0)*1e3)
    dr = float(np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1)-1)/2, -1, 1))))
    nf = sum(1 for v in ep.forces[k1-1].values() if v > 1e-3)
    w = [np.linalg.norm(np.array(ep.palm[k+1][3:])-np.array(ep.palm[k][3:]))*2*60 for k in range(K0, k1)]
    return dict(seed=seed, i=i, family=ep.tool['handle'], rate=rate, slip_mm=dp, slip_deg=dr,
                lost=bool(dp > 20 or nf == 0), palm_w_mean=float(np.mean(w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_family', type=int, default=40)
    ap.add_argument('--rates', default='1,4,8,16')
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--out', default=os.environ.get('OUT', 'outputs/neurosim/reorient_stress'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from dextrah_lab.neurosim.reference.m_holds import cached_pool
    pool_, _ = cached_pool()
    fam = np.asarray(pool_.family)
    rng = np.random.default_rng(a.seed)
    jobs = []
    for f in sorted(set(fam.tolist())):
        idx = rng.choice(np.nonzero(fam == f)[0], a.per_family, replace=False)
        for j, i in enumerate(idx):
            for r in [float(x) for x in a.rates.split(',')]:
                jobs.append((a.seed*100_000+j, int(i), r, a.steps))      # same hold and seed across rates
    with Pool(a.workers) as p:
        R = [r for r in p.map(task, jobs, chunksize=2) if r]
    (out/'reorient_stress.json').write_text(json.dumps(R, indent=1))
    for r_ in sorted({r['rate'] for r in R}):
        S = [r for r in R if r['rate'] == r_]
        print(f'RATE {r_:g}: palm angular speed mean {np.mean([r["palm_w_mean"] for r in S]):.2f} rad/s; lost {np.mean([r["lost"] for r in S]):.3f}; '
              f'slip median {np.median([r["slip_deg"] for r in S]):.2f} deg', flush=True)
        line = {}
        for f in sorted({r['family'] for r in S}):
            Sf = [r for r in S if r['family'] == f]
            line[f] = (round(float(np.mean([r['lost'] for r in Sf])), 3), round(float(np.median([r['slip_deg'] for r in Sf])), 2))
        print('  by family (lost, slip deg median):', line, flush=True)
    print('DONE ->', out)


if __name__ == '__main__':
    main()
