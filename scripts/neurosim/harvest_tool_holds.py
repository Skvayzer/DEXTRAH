#!/usr/bin/env python3
"""Harvest M's stored holds (tools) into the stage 3 testbed: the Phase 1 held-state bank (CPU, multiprocess).

For every held state of a run's success-guided sampler pool (dextrah_lab/neurosim/reference/tool_holds.py): rebuild
it in the testbed (mesh geometry = training hulls; one compiled testbed per tool), run the acceptance test and keep
the testbed state after the contact transient and settle. Outcome per hold: held (within 5 mm / 5 deg of its
post-transient pose through settle and a +-3 mm / 3 deg shake), shifted (still in the hand), dropped (no contact).
Phase 1 uses held and shifted (firm vs slipping); dropped are logged only.
Output: <out>/holds.pt (one row per hold), <out>/summary.json.
"""
import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np


def work(args):
    shard_keys, sgs, run, geometry, seed = args
    from dextrah_lab.mj.compat import install
    install()
    from dextrah_lab.neurosim.reference.tool_holds import decode_pool, run_acceptance, HoldPool
    pool = decode_pool(sgs, run)
    idx = np.nonzero(np.isin(pool.keys, shard_keys))[0]
    sub = HoldPool(keys=pool.keys[idx], asset=pool.asset[idx], family=[pool.family[i] for i in idx], palm=pool.palm[idx],
                   hand_q=pool.hand_q[idx], hand_target=pool.hand_target[idx], object_qpos=pool.object_qpos[idx],
                   table_top=pool.table_top[idx], arm_touch=pool.arm_touch[idx], success=pool.success[idx], meta=pool.meta)
    order = np.argsort(sub.asset, kind='stable')           # group by tool: one compiled testbed per tool
    sub = HoldPool(keys=sub.keys[order], asset=sub.asset[order], family=[sub.family[i] for i in order], palm=sub.palm[order],
                   hand_q=sub.hand_q[order], hand_target=sub.hand_target[order], object_qpos=sub.object_qpos[order],
                   table_top=sub.table_top[order], arm_touch=sub.arm_touch[order], success=sub.success[order], meta=sub.meta)
    rows = run_acceptance(sub, run, geometry=geometry, seed=seed)
    return [dict(key=int(sub.keys[i]), asset=int(sub.asset[i]), family=sub.family[i], m_success=float(sub.success[i]),
                 palm=sub.palm[i], hand_q=sub.hand_q[i], hand_target=sub.hand_target[i], object_qpos=sub.object_qpos[i],
                 table_top=float(sub.table_top[i]), **{k: v for k, v in r.items()}) for i, r in enumerate(rows)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sgs', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--geometry', default='mesh')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import torch
    from dextrah_lab.mj.compat import install
    install()
    from dextrah_lab.neurosim.reference.tool_holds import decode_pool, summarize, HoldPool
    t0 = time.time()
    pool = decode_pool(a.sgs, a.run, limit=a.limit, seed=a.seed)
    n = len(pool.keys)
    shards = np.array_split(pool.keys, a.workers)
    print(f'HARVEST decoded {n} holds in {time.time()-t0:.0f} s; {a.workers} workers', flush=True)
    with get_context('spawn').Pool(a.workers) as ex:
        parts = ex.map(work, [(s, str(a.sgs), str(a.run), a.geometry, a.seed+i) for i, s in enumerate(shards)])
    rows = [r for part in parts for r in part]
    torch.save(rows, a.out/'holds.pt')
    pool_rows = HoldPool(keys=np.array([r['key'] for r in rows]), asset=np.array([r['asset'] for r in rows]),
                         family=[r['family'] for r in rows], palm=None, hand_q=None, hand_target=None, object_qpos=None,
                         table_top=None, arm_touch=np.zeros(len(rows), dtype=bool), success=np.array([r['m_success'] for r in rows]))
    s = summarize(pool_rows, rows)
    s.update(sgs=str(a.sgs), run=str(a.run), geometry=a.geometry, wall_s=round(time.time()-t0, 1),
             usable=round(float(np.mean([r['outcome'] != 'dropped' for r in rows])), 4))
    (a.out/'summary.json').write_text(json.dumps(s, indent=2))
    print('HARVEST_DONE '+json.dumps({k: s[k] for k in ('n', 'accepted', 'usable', 'outcomes', 'by_family', 'wall_s')}), flush=True)


if __name__ == '__main__':
    main()
