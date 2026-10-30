#!/usr/bin/env python3
"""Phase 1 reference gradients on the tool episodes (dextrah_lab/neurosim/reference/ref_grads.py), CPU, multiprocess.

Samples --per-stratum logged states per stratum (pre_contact, forming, holding, touchdown, release, incidental) from
the episodes of one split, computes the noise-averaged directional derivatives at each, and reports: exact replay of
the log, the G1 agreement of two independent halves of the draws (cosine over standardized outcomes, per stratum,
noise setting, horizon and channel), gradient magnitudes and clipping. Output: <out>/grads_<worker>.pt (one dict per
state), <out>/summary.json.
"""
import argparse
import glob
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

H = 8


def work(args):
    samples, shards, runs, out, wid, seed, spec = args
    import torch
    from dextrah_lab.mj.compat import install
    install()
    from dextrah_lab.neurosim.testbed import PREFIX, Testbed, TestbedSpec
    from dextrah_lab.neurosim.reference.held_states import COMMAND_JOINTS
    from dextrah_lab.neurosim.reference.tool_holds import object_bank_for_run, tool_of
    from dextrah_lab.neurosim.reference.tool_episodes import Recorder, set_state
    from dextrah_lab.neurosim.reference.ref_grads import GradSpec, Probe, state_grads
    tables = {}
    sh_path, sh, tb_key, tb = None, None, None, None
    rows, t0 = [], time.time()
    for n, (si, e, t, stratum) in enumerate(samples):
        if shards[si] != sh_path:
            sh_path, sh = shards[si], torch.load(shards[si], weights_only=False)
        m = sh['meta'][e]
        if (m['pool'], m['asset']) != tb_key:
            tb_key = (m['pool'], m['asset'])
            if m['pool'] not in tables:
                tables[m['pool']] = object_bank_for_run(runs[m['pool']])[2]
            tb = Testbed(TestbedSpec(geometry='mesh', object_kind='tool', tool=tool_of(tables[m['pool']], m['asset'])))
            jid = [tb.m.joint(PREFIX+j).id for j in COMMAND_JOINTS]
            lo, hi = tb.m.jnt_range[jid, 0], tb.m.jnt_range[jid, 1]
            probe, rec = Probe(tb), Recorder(tb)
        a, sa = sh['step_offsets'][e], sh['state_offsets'][e]
        state = sh['state'][sa+t]
        set_state(tb, state)
        gap = sh['targets'][a+t]-tb.d.qpos[tb.qadr]
        rng = np.random.default_rng([seed, si, e, t])
        dirs_f, dirs_p, exact, res = state_grads(tb, probe, state, sh['cmd'][a+t:a+t+H], sh['palm_next'][a+t:a+t+H],
                                                 lo, hi, rng, spec, next_states=sh['state'][sa+t+1:sa+t+1+H],
                                                 rec=rec)
        rows.append(dict(shard=shards[si], episode=int(e), t=int(t), stratum=stratum, family=m['family'],
                         asset=int(m['asset']), pool=int(m['pool']), phase=int(sh['phase'][a+t]), replay_exact=exact,
                         gap=gap, dirs_f=dirs_f, dirs_p=dirs_p, results=res))
        if (n+1) % 25 == 0 or n+1 == len(samples):
            print(f'GRAD worker {wid}: {n+1}/{len(samples)} states, {(time.time()-t0)/(n+1):.1f} s/state', flush=True)
    torch.save(rows, Path(out)/f'grads_{wid:02d}.pt')
    return len(rows)


def agreement(rows, spec_noise, horizons):
    """G1: cosine between the two halves' directional derivatives over standardized outcomes (all outcomes, and per
    vector outcome group)."""
    from dextrah_lab.neurosim.reference.ref_grads import OUTCOMES, STRATA
    groups, i0 = {}, 0
    for name, n in OUTCOMES:
        if n > 1:
            groups[name] = slice(i0, i0+n)
        i0 += n
    D = rows[0]['dirs_f'].shape[0]
    out = {}
    for ni in range(len(spec_noise)):
        if rows[0]['results'][ni]['grad_halves'] is None:
            continue
        allg = np.stack([r['results'][ni]['grad'] for r in rows])          # (S, 2D, horizons, NY)
        for ch, sl in (('finger', slice(0, D)), ('palm', slice(D, 2*D))):
            scale = np.sqrt((allg[:, sl]**2).mean((0, 1)))+1e-12           # per horizon, outcome
            for st in STRATA:
                rs = [r for r in rows if r['stratum'] == st]
                if not rs:
                    continue
                def cosine(sel):
                    cos = []
                    for r in rs:
                        hv = r['results'][ni]['grad_halves'][:, sl][..., sel]/scale[..., sel]   # (2, D, horizons, y)
                        num = (hv[0]*hv[1]).sum(-1)
                        den = np.linalg.norm(hv[0], axis=-1)*np.linalg.norm(hv[1], axis=-1)+1e-12
                        cos.append(num/den)                                 # (D, horizons)
                    return np.concatenate(cos)                              # (states*D, horizons)
                cos = cosine(slice(None))
                by_group = {g: {h: round(float(np.median(c[:, i])), 3) for i, h in enumerate(horizons)}
                            for g, c in ((g, cosine(sl_g)) for g, sl_g in groups.items())}
                mag = np.stack([np.linalg.norm(r['results'][ni]['grad'][sl]/scale, axis=-1) for r in rs])
                out[f'{spec_noise[ni]}|{ch}|{st}'] = dict(
                    n=len(rs), median_cos={h: round(float(np.median(cos[:, i])), 3) for i, h in enumerate(horizons)},
                    frac_cos_ge_0_9={h: round(float((cos[:, i] >= .9).mean()), 3) for i, h in enumerate(horizons)},
                    median_norm_mag={h: round(float(np.median(mag[..., i])), 3) for i, h in enumerate(horizons)},
                    median_cos_by_group=by_group)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--episodes', type=Path, required=True)
    p.add_argument('--runs', type=Path, nargs='+', required=True, help='run dir per pool (order of generation)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--per-stratum', type=int, default=100)
    p.add_argument('--split', default='train')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--draws', type=int, default=64)
    p.add_argument('--delta-f', type=float, default=.01, help='rad, finger commands')
    p.add_argument('--delta-p', type=float, default=1., help='palm twist units (1 = 1 cm/s and 0.05 rad/s)')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import torch
    from dextrah_lab.neurosim.reference.ref_grads import STRATA, GradSpec, strata
    spec = GradSpec(draws=a.draws, delta_f=a.delta_f, delta_p=a.delta_p)
    t0 = time.time()
    shards = sorted(glob.glob(str(a.episodes/'shard_*.pt')))
    cand = defaultdict(list)
    for si, f in enumerate(shards):
        sh = torch.load(f, weights_only=False)
        for e, m in enumerate(sh['meta']):
            if m['split'] != a.split:
                continue
            lab = strata(sh, e)
            for t, s in enumerate(lab[:max(0, len(lab)-H)]):
                if s is not None:
                    cand[s].append((si, e, t, s))
    rng = np.random.default_rng(a.seed)
    samples = []
    for s in STRATA:
        c = cand[s]
        samples += [c[i] for i in rng.choice(len(c), min(a.per_stratum, len(c)), replace=False)]
    samples.sort(key=lambda x: (x[0], x[1], x[2]))
    print(f'GRAD candidates per stratum { {s: len(cand[s]) for s in STRATA} }; {len(samples)} states; '
          f'setup {time.time()-t0:.0f} s', flush=True)
    runs = {i: str(r) for i, r in enumerate(a.runs)}
    split = np.array_split(np.arange(len(samples)), a.workers)
    with ProcessPoolExecutor(a.workers, mp_context=get_context('spawn')) as ex:
        list(ex.map(work, [([samples[i] for i in idx], shards, runs, str(a.out), w, a.seed, spec)
                           for w, idx in enumerate(split)]))
    rows = [r for f in sorted(glob.glob(str(a.out/'grads_*.pt'))) for r in torch.load(f, weights_only=False)]
    s = dict(states=len(rows), spec=dict(draws=spec.draws, delta_f=spec.delta_f, delta_p=spec.delta_p,
                                         palm_lin=spec.palm_lin, palm_ang=spec.palm_ang, noise=spec.noise), per_stratum={st: sum(r['stratum'] == st for r in rows) for st in STRATA},
             replay_exact=round(float(np.mean([bool(r['replay_exact']) for r in rows])), 4),
             clipped_per_command={str(spec.noise[ni]): np.round(np.mean([r['results'][ni]['clipped'] for r in rows], 0), 3).tolist()
                                  for ni in range(len(spec.noise))},
             agreement=agreement(rows, [str(x) for x in spec.noise], spec.horizons), wall_s=round(time.time()-t0, 1))
    (a.out/'summary.json').write_text(json.dumps(s, indent=1))
    print('GRAD_DONE '+json.dumps({k: s[k] for k in ('states', 'per_stratum', 'replay_exact', 'clipped_per_command', 'wall_s')}),
          flush=True)


if __name__ == '__main__':
    main()
