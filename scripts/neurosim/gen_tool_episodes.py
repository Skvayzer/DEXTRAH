#!/usr/bin/env python3
"""Phase 1 episodes on tools (dextrah_lab/neurosim/reference/tool_episodes.py), CPU, multiprocess.

Episodes start from usable harvested holds (held or shifted; harvest_tool_holds.py) of one or more pools. Sampling:
object family uniform, then a hold, firm holds drawn with probability --firm-share. Tools with asset id % 10 == 0 are
held out (split 'test'). Each worker writes shards of --shard-episodes episodes:
<out>/shard_<worker>_<k>.pt = dict(meta per episode, offsets, concatenated step arrays, states, contacts).
"""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from dextrah_lab.neurosim.reference.tool_episodes import INPUTS

META = ('pool', 'key', 'asset', 'family', 'outcome', 'worst_mm', 'worst_deg', 'm_success', 'table_top')


def pack(eps):
    """Concatenate a list of (meta, arrays, labels) into one shard dict."""
    out = dict(meta=[dict(**m, **l) for m, _, l in eps])
    keys = [k for k in eps[0][1] if k not in ('state', 'contacts')]
    for name, ks in (('step', keys), ('state', ['state']), ('contact', ['contacts'])):
        lens = [len(a[ks[0]]) for _, a, _ in eps]
        out[f'{name}_offsets'] = np.concatenate(([0], np.cumsum(lens)))
        for k in ks:
            x = np.concatenate([a[k] for _, a, _ in eps])
            out[k] = (x if k in ('state', 'contacts') or k in INPUTS else
                      x.astype(np.int8) if k == 'phase' else x.astype(np.float32))
    return out


def work(args):
    jobs, holds_by_pool, runs, out, wid, shard_n = args
    import torch
    from dextrah_lab.mj.compat import install
    install()
    from dextrah_lab.neurosim.testbed import Testbed, TestbedSpec
    from dextrah_lab.neurosim.reference.tool_holds import object_bank_for_run, tool_of
    from dextrah_lab.neurosim.reference.tool_episodes import Episode
    tables = {p: object_bank_for_run(r)[2] for p, r in runs.items()}
    holds = {p: torch.load(f, weights_only=False) for p, f in holds_by_pool.items()}
    tb_key, tb, buf, k, stats = None, None, [], 0, Counter()
    t0 = time.time()
    for n, (pool, i, seed) in enumerate(jobs):
        h = holds[pool][i]
        if tb_key != (pool, h['asset']):
            tb_key = (pool, h['asset'])
            tb = Testbed(TestbedSpec(geometry='mesh', object_kind='tool', tool=tool_of(tables[pool], h['asset'])))
        arrays, labels = Episode(tb, h, np.random.default_rng(seed)).run()
        meta = {m: h[m] for m in META if m != 'pool'}
        meta.update(pool=pool, hold_index=int(i), seed=int(seed), split='test' if h['asset'] % 10 == 0 else 'train')
        buf.append((meta, arrays, labels))
        stats['episodes'] += 1
        stats['steps'] += labels['steps']
        stats['contact_rows'] += len(arrays['contacts'])
        stats[f"carry_{labels['carry']}"] += 1
        stats['touchdown'] += labels['touchdown'] is not None
        stats['fell'] += labels['fell_at'] is not None
        if len(buf) == shard_n or n+1 == len(jobs):
            torch.save(pack(buf), Path(out)/f'shard_{wid:02d}_{k:03d}.pt')
            buf, k = [], k+1
            print(f'GEN worker {wid}: {n+1}/{len(jobs)} episodes, {stats["steps"]} steps, '
                  f'{stats["steps"]/(time.time()-t0):.0f} steps/s', flush=True)
    return dict(stats)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--holds', type=Path, nargs='+', required=True, help='holds.pt per pool')
    p.add_argument('--runs', type=Path, nargs='+', required=True, help='run dir per pool (same order)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=3000)
    p.add_argument('--firm-share', type=float, default=1/3)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--shard-episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import torch
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    cand = []                                            # (pool, index, family, firm)
    for pool, f in enumerate(a.holds):
        for i, r in enumerate(torch.load(f, weights_only=False)):
            if r['outcome'] != 'dropped':
                cand.append((pool, i, r['family'], r['outcome'] == 'held', r['asset']))
    fams = sorted({c[2] for c in cand})
    groups = {(f, firm): [c for c in cand if c[2] == f and c[3] == firm] for f in fams for firm in (True, False)}
    jobs = []
    for e in range(a.episodes):
        f = fams[rng.integers(len(fams))]
        firm = rng.random() < a.firm_share and len(groups[(f, True)]) > 0
        g = groups[(f, firm)]
        c = g[rng.integers(len(g))]
        jobs.append((c[0], c[1], int(rng.integers(2**31)), c[4]))
    jobs.sort(key=lambda j: (j[0], j[3]))                  # by pool and tool: each tool compiled once per worker
    split = np.array_split(np.arange(len(jobs)), a.workers)
    holds_by_pool = {i: str(f) for i, f in enumerate(a.holds)}
    runs = {i: str(r) for i, r in enumerate(a.runs)}
    print(f'GEN {len(cand)} usable holds, {a.episodes} episodes, {a.workers} workers, setup {time.time()-t0:.0f} s', flush=True)
    with ProcessPoolExecutor(a.workers, mp_context=get_context('spawn')) as ex:
        parts = list(ex.map(work, [([jobs[j][:3] for j in idx], holds_by_pool, runs, str(a.out), w, a.shard_episodes)
                                   for w, idx in enumerate(split)]))
    tot = Counter()
    for s in parts:
        tot.update(s)
    s = dict(tot, wall_s=round(time.time()-t0, 1), holds=[str(f) for f in a.holds], runs=[str(r) for r in a.runs],
             episodes_requested=a.episodes, firm_share=a.firm_share, seed=a.seed)
    (a.out/'summary.json').write_text(json.dumps(s, indent=2))
    print('GEN_DONE '+json.dumps(s), flush=True)


if __name__ == '__main__':
    main()
