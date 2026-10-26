#!/usr/bin/env python3
"""Counterfactuals for a captured non-finite world (mj_nan_replay.py rebuild): repeat the failing
step N times (MuJoCo Warp float32 solves differ by atomic order) and count non-finite results with
the captured model, and with the object's mass and inertia multiplied by given factors (e.g. the
per-episode mass randomisation undone). Writes probe_w<world>.json next to the dump."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dump', type=Path)
    p.add_argument('--world', type=int)
    p.add_argument('--repeats', type=int, default=200)
    p.add_argument('--mass-factors', type=float, nargs='*', default=[1.])
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    import mujoco
    import torch
    import warp as wp
    import mujoco_warp as mjw
    from mj_nan_replay import load_inputs, restore_model
    wp.init()
    dump = args.dump
    meta = json.loads((dump/'model_scalars.json').read_text())
    info = json.loads((dump.parent/'capture_info.json').read_text())
    inputs = np.load(dump/'inputs.npz')
    world = args.world if args.world is not None else meta['worlds'][0]
    k_model, k_in = meta['worlds'].index(world), list(inputs['worlds']).index(world)
    last = len(inputs['steps'])-1
    mjm = mujoco.MjModel.from_binary_path(str(dump/'model.mjb'))
    saved = np.load(dump/'model.npz')
    body = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, 'object/object_root')
    out = dict(world=world, repeats=args.repeats, object_body=body, results=[])
    with wp.ScopedDevice(args.device):
        m, _ = restore_model(mjm, saved, meta, k_model)
        mass0 = wp.to_torch(m.body_mass)[0, body].item()
        inertia0 = wp.to_torch(m.body_inertia)[0, body].clone()
        n = args.repeats
        d = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=n, nconmax=64*n, njmax=info['njmax'])
        for f in args.mass_factors:
            wp.to_torch(m.body_mass)[0, body] = mass0*f
            wp.to_torch(m.body_inertia)[0, body] = inertia0*f
            for name in [x for x in inputs.files if x not in ('worlds', 'steps')]:
                arr = getattr(d, name, None)
                if arr is None or arr.size == 0:
                    continue
                t = wp.to_torch(arr)
                t[:] = torch.as_tensor(inputs[name][last, k_in], device=t.device).to(t.dtype).reshape(t.shape[1:])
            mjw.step(m, d)
            wp.synchronize()
            q = wp.to_torch(d.qpos)
            bad = ~torch.isfinite(q).all(-1)
            out['results'].append(dict(mass_factor=f, object_mass_kg=mass0*f, nonfinite=int(bad.sum()), of=n))
            print(json.dumps(out['results'][-1]), flush=True)
    (dump/f'probe_w{world}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
