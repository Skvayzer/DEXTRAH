"""E10: finger servo variants on M run 1454's holds: does removing the 120 Hz chatter change grip stability?

Variants (TestbedSpec): velocity servo at 2 substeps (training), deadbeat servo at 2 substeps (gain capped at 1/dt),
velocity servo at 10 substeps (fine-dt reference). Per variant, on the same random sample of M's held configurations:
tool_holds.accept outcome (held / shifted / dropped after a 0.5 s settle and a ±3 mm / ±3 deg palm shake), transient
and worst displacement / rotation, wall time per control step; plus the free-finger chatter (constant targets, last
1 s) and the closing-trajectory error against the 10-substep reference.
"""
import argparse
import collections
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from dextrah_lab.neurosim.reference import m_holds as M
from dextrah_lab.neurosim.reference.tool_holds import accept, tool_of
from dextrah_lab.neurosim.testbed import Testbed, TestbedSpec

VARIANTS = dict(velocity2=dict(hand_servo='velocity', substeps=2), deadbeat2=dict(hand_servo='deadbeat', substeps=2),
                velocity10=dict(hand_servo='velocity', substeps=10))


def job(args):
    i, name = args
    pool, table = M.cached_pool()
    tb = Testbed(TestbedSpec(object_kind='tool', tool=tool_of(table, int(pool.asset[i])), **VARIANTS[name]))
    t0 = time.time()
    r = accept(tb, pool, i, table, np.random.default_rng(i))
    return dict(i=i, variant=name, outcome=r['outcome'], worst_mm=r['worst_mm'], worst_deg=r['worst_deg'],
                transient_mm=r['transient_mm'], transient_deg=r['transient_deg'], sec_per_step=(time.time()-t0)/75,
                grip=float(sum(v for k, v in r['forces'].items() if k != 'palm')))


def free_fingers():
    palm = np.array([0, .8, 1.2, 1, 0, 0, 0.])
    tgt = np.array([.6, .5, .5, .7, .8, .7, .8, .7, .8, .7, .8])
    Q = {}
    for name, kw in VARIANTS.items():
        tb = Testbed(TestbedSpec(**kw))
        tb.reset(palm, np.zeros(11), np.r_[1., 1., 10., 1, 0, 0, 0])
        q = []
        for _ in range(120):
            tb.step(tgt, palm)
            q.append(tb.d.qpos[tb.qadr].copy())
        Q[name] = np.array(q)
    return {n: dict(gap_median_mrad=float(1e3*np.median(np.abs(tgt-q[60:]))), gap_max_mrad=float(1e3*np.abs(tgt-q[60:]).max()),
                    closing_err_vs_ref_mrad=float(1e3*np.abs(q[:30]-Q['velocity10'][:30]).max())) for n, q in Q.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--holds', type=int, default=400)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.environ.get('OUT', 'outputs/neurosim/servo_compare'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print('CONFIG', json.dumps(vars(a)), 'VARIANTS', json.dumps(VARIANTS), flush=True)
    ff = free_fingers()
    for n, v in ff.items():
        print('FREE', n, json.dumps({k: round(x, 3) for k, x in v.items()}), flush=True)
    pool, _ = M.cached_pool()
    idx = np.random.default_rng(a.seed).choice(len(pool.keys), a.holds, replace=False)
    with Pool(a.workers) as p:
        R = p.map(job, [(int(i), n) for n in VARIANTS for i in idx])
    (out/'servo_compare.json').write_text(json.dumps(dict(free=ff, holds=R), indent=1))
    for n in VARIANTS:
        S = [r for r in R if r['variant'] == n]
        c = collections.Counter(r['outcome'] for r in S)
        print('RESULT', n, 'n=%d' % len(S), ' '.join(f'{k} {c[k]/len(S):.3f}' for k in ('held', 'shifted', 'dropped')),
              'worst_deg median %.1f' % np.median([r['worst_deg'] for r in S]),
              'worst_mm median %.2f' % np.median([r['worst_mm'] for r in S]),
              'transient_deg median %.1f' % np.median([r['transient_deg'] for r in S]),
              'grip N median %.1f' % np.median([r['grip'] for r in S]),
              'ms/step %.2f' % (1e3*np.median([r['sec_per_step'] for r in S])), flush=True)
    # paired: same holds, deadbeat vs training servo
    by = {(r['i'], r['variant']): r for r in R}
    d = np.array([by[(int(i), 'deadbeat2')]['worst_deg']-by[(int(i), 'velocity2')]['worst_deg'] for i in idx])
    print('PAIRED deadbeat2 - velocity2 worst_deg: median %.2f, deadbeat better in %.2f of holds' % (np.median(d), np.mean(d < 0)))
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
