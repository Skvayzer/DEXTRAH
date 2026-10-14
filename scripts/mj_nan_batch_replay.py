#!/usr/bin/env python3
"""Is a captured mjlab NaN a property of the world, or of the batched solve?

The single-world replay of run 1213's failing world (scripts/mj_nan_replay.py)
stayed finite: same contacts, solver converged in 8 iterations, whereas in the
23,040-world batch every constraint force of that world came out NaN after 4.
This replays the failing step of a nan_capture dump
* as 1 world, R times (run-to-run determinism), and
* as N copies of the same world in one batch, R times, eagerly and through a
  captured CUDA graph (how mjlab steps),
reporting NaN copies, solver iterations and the spread of qacc across copies
and repeats. Also records the GPU's ECC / row-remapping state (nvidia-smi).
Writes batch_replay.json next to the dump.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mj_nan_replay import load_inputs, restore_model  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dump', type=Path)
    p.add_argument('--world', type=int)
    p.add_argument('--copies', type=int, nargs='+', default=[1, 1024, 23040])
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--nconmax', type=int, default=64, help='per world, as in the run (single world: 512)')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    import mujoco
    import torch
    import warp as wp
    import mujoco_warp as mjw
    wp.init()
    dump = args.dump
    meta = json.loads((dump/'model_scalars.json').read_text())
    info = json.loads((dump.parent/'capture_info.json').read_text())
    inputs = np.load(dump/'inputs.npz')
    world = args.world if args.world is not None else meta['worlds'][0]
    k_model, k_in = meta['worlds'].index(world), list(inputs['worlds']).index(world)
    slot = len(inputs['steps'])-1
    out = dict(dump=str(dump), world=world, step=int(inputs['steps'][slot]), njmax=info['njmax'], runs=[])
    try:
        out['nvidia_smi'] = subprocess.run(['nvidia-smi', '-q', '-d', 'ECC,ROW_REMAPPER,PAGE_RETIREMENT'],
                                           capture_output=True, text=True, timeout=60).stdout
    except Exception as e:
        out['nvidia_smi'] = repr(e)
    mjm = mujoco.MjModel.from_binary_path(str(dump/'model.mjb'))
    saved = np.load(dump/'model.npz')
    reference = None
    with wp.ScopedDevice(args.device):
        m, out['model_restore'] = restore_model(mjm, saved, meta, k_model)
        for n in args.copies:
            d = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=n, nconmax=512 if n == 1 else args.nconmax,
                             njmax=info['njmax'])
            for mode in ('eager', 'graph'):
                graph = None
                if mode == 'graph':
                    load_inputs(d, inputs, slot, k_in)
                    with wp.ScopedCapture() as capture:
                        mjw.step(m, d)
                    graph = capture.graph
                runs = []
                for r in range(args.repeats):
                    load_inputs(d, inputs, slot, k_in)
                    if graph is None:
                        mjw.step(m, d)
                    else:
                        wp.capture_launch(graph)
                    wp.synchronize()
                    qacc = wp.to_torch(d.qacc).double()
                    bad = ~torch.isfinite(qacc).all(1)
                    good = qacc[~bad]
                    niter = wp.to_torch(d.solver_niter)
                    if reference is None and len(good):
                        reference = good[0].clone()
                    spread = float((good-good[0]).abs().max()) if len(good) > 1 else 0.
                    vs_ref = float((good-reference).abs().max()) if len(good) else None
                    runs.append(dict(nan_copies=int(bad.sum()), niter=sorted(set(niter.tolist()))[:10],
                                     qacc_spread_across_copies=spread, qacc_max_diff_vs_first=vs_ref,
                                     nacon=int(wp.to_torch(d.nacon)[0]),
                                     overflow=sorted(set(wp.to_torch(d.overflow).tolist()))))
                summary = dict(copies=n, mode=mode, nan_runs=sum(x['nan_copies'] > 0 for x in runs),
                               nan_copies_total=sum(x['nan_copies'] for x in runs), runs=runs)
                out['runs'].append(summary)
                print(json.dumps({k: v for k, v in summary.items() if k != 'runs'}),
                      json.dumps(runs[0]), flush=True)
            del d
    (dump/'batch_replay.json').write_text(json.dumps(out, indent=1))
    print(out['nvidia_smi'][-2500:])


if __name__ == '__main__':
    main()
