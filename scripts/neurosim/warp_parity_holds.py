#!/usr/bin/env python3
"""Phase 0.6 Warp parity on harvested tool holds (dextrah_lab/neurosim/reference/warp_parity.py).

Samples tools stratified by family from a harvest (holds.pt of harvest_tool_holds.py), takes up to --cap holds per
tool and runs the acceptance test on each hold x --replicas shake directions in C MuJoCo and in MuJoCo Warp on the
same compiled model. Reports outcome rates (overall, per family), the paired outcome table, per-hold held / dropped
probabilities over the shake directions, displacement quantiles, Warp replica spread over the (shake-free)
transient, NaN count and timing. Output: <out>/parity.pt (one row per world), <out>/summary.json.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

OUTCOMES = ('held', 'shifted', 'dropped')


def q(x, ps=(.1, .25, .5, .75, .9)):
    return np.round(np.quantile(np.asarray(x), ps), 2).tolist()


def summarize(rows, R):
    out = dict(worlds=len(rows), holds=len({(r['asset'], r['hold']) for r in rows}), tools=len({r['asset'] for r in rows}),
               replicas=R)
    for side in ('c', 'w'):
        o = np.array([r[side]['outcome'] for r in rows])
        out[f'rates_{side}'] = {k: round(float((o == k).mean()), 4) for k in OUTCOMES}
    fam = np.array([r['family'] for r in rows])
    out['by_family'] = {}
    for f in sorted(set(fam)):
        sel = [r for r, x in zip(rows, fam == f) if x]
        out['by_family'][f] = dict(worlds=len(sel), **{f'held_{s}': round(float(np.mean([r[s]['outcome'] == 'held' for r in sel])), 3)
                                                       for s in ('c', 'w')},
                                   **{f'dropped_{s}': round(float(np.mean([r[s]['outcome'] == 'dropped' for r in sel])), 3)
                                      for s in ('c', 'w')})
    out['paired_c_rows_w_cols'] = {a: {b: int(sum(r['c']['outcome'] == a and r['w']['outcome'] == b for r in rows))
                                       for b in OUTCOMES} for a in OUTCOMES}
    out['paired_agreement'] = round(float(np.mean([r['c']['outcome'] == r['w']['outcome'] for r in rows])), 4)
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for s in ('c', 'w'):
            per[(r['asset'], r['hold'])][s].append(r[s])
    for k in ('held', 'dropped'):
        pc = np.array([np.mean([x['outcome'] == k for x in v['c']]) for v in per.values()])
        pw = np.array([np.mean([x['outcome'] == k for x in v['w']]) for v in per.values()])
        corr = float(np.corrcoef(pc, pw)[0, 1]) if pc.std() > 0 and pw.std() > 0 else float('nan')
        out[f'per_hold_p_{k}'] = dict(mean_c=round(float(pc.mean()), 4), mean_w=round(float(pw.mean()), 4),
                                      mean_abs_diff=round(float(np.abs(pc-pw).mean()), 4), corr=round(corr, 3))
    for key in ('worst_mm', 'worst_deg', 'transient_mm', 'transient_deg'):
        out[key] = dict(c=q([r['c'][key] for r in rows]), w=q([r['w'][key] for r in rows if not r['w']['nan']]))
    spread = [np.ptp([x['transient_mm'] for x in v['w']]) for v in per.values()]
    out['warp_transient_spread_mm_over_replicas'] = q(spread, (.5, .9, .99, 1.))
    out['warp_nan_worlds'] = int(sum(r['w']['nan'] for r in rows))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--holds', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tools-per-family', type=int, default=12)
    p.add_argument('--min-holds', type=int, default=6)
    p.add_argument('--cap', type=int, default=24)
    p.add_argument('--replicas', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import torch
    from dextrah_lab.mj.compat import install
    install()
    from dextrah_lab.neurosim.testbed import Testbed, TestbedSpec
    from dextrah_lab.neurosim.reference.tool_holds import object_bank_for_run, tool_of
    from dextrah_lab.neurosim.reference.warp_parity import accept_c, accept_warp, shake_dirs
    t0 = time.time()
    harvest = torch.load(a.holds, weights_only=False)
    by_tool = defaultdict(list)
    for r in harvest:
        by_tool[r['asset']].append(r)
    rng = np.random.default_rng(a.seed)
    fam_tools = defaultdict(list)
    for t, rs in by_tool.items():
        if len(rs) >= a.min_holds:
            fam_tools[rs[0]['family']].append(t)
    tools = []
    for f in sorted(fam_tools):
        ts = sorted(fam_tools[f])
        tools += [int(t) for t in rng.choice(ts, min(a.tools_per_family, len(ts)), replace=False)]
    _, _, table = object_bank_for_run(a.run)
    print(f'PARITY {len(tools)} tools ({ {f: min(a.tools_per_family, len(v)) for f, v in sorted(fam_tools.items())} }), '
          f'setup {time.time()-t0:.0f} s', flush=True)
    rows, wall_w, wall_c, steps_w = [], 0., 0., 0
    for n, t in enumerate(tools):
        rs = by_tool[t]
        rs = [rs[i] for i in sorted(rng.choice(len(rs), min(a.cap, len(rs)), replace=False))]
        tb = Testbed(TestbedSpec(geometry='mesh', object_kind='tool', tool=tool_of(table, t)))
        worlds = [(h, r) for h in range(len(rs)) for r in range(a.replicas)]
        dirs = [(h, *shake_dirs(a.seed, t, h, r)[1:]) for h, r in worlds]
        res_w, w = accept_warp(tb, rs, dirs, device=a.device)
        wall_w += w
        steps_w += len(worlds)*60
        tc = time.time()
        res_c = accept_c(tb, rs, [(h, shake_dirs(a.seed, t, h, r)[0]) for h, r in worlds])
        wall_c += time.time()-tc
        for (h, r), c, wr in zip(worlds, res_c, res_w):
            rows.append(dict(asset=t, family=rs[h]['family'], key=rs[h]['key'], hold=h, replica=r,
                             harvest_outcome=rs[h]['outcome'], c=c, w=wr))
        if (n+1) % 6 == 0 or n+1 == len(tools):
            s = summarize(rows, a.replicas)
            print(f'PARITY {n+1}/{len(tools)} tools, {len(rows)} worlds: C {s["rates_c"]} | Warp {s["rates_w"]} | '
                  f'agree {s["paired_agreement"]} | nan {s["warp_nan_worlds"]}', flush=True)
    torch.save(rows, a.out/'parity.pt')
    s = summarize(rows, a.replicas)
    s.update(holds_file=str(a.holds), wall_s=round(time.time()-t0, 1), warp_step_wall_s=round(wall_w, 1),
             warp_world_steps_per_s=round(steps_w/max(wall_w, 1e-9)), c_wall_s=round(wall_c, 1),
             harvest_consistency=round(float(np.mean([r['c']['outcome'] == r['harvest_outcome'] for r in rows if r['replica'] == 0])), 4))
    (a.out/'summary.json').write_text(json.dumps(s, indent=2))
    print('PARITY_DONE '+json.dumps(s), flush=True)


if __name__ == '__main__':
    main()
