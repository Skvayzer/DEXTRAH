#!/usr/bin/env python3
"""Do objects fall off the table at spawn with nobody touching them?

Inference only, no policy: zero latent residual (frozen SONIC stands) and
fully open right fingers. Every environment keeps its assigned object
(env i -> object i % 1200), so 1200 environments cover the whole bank once
per reset. Unchanged reset rules (spawn XY box, drop height, random
orientation). A spawn counts as an untouched fall when the object goes below
the table top within the window after reset while no right fingertip has
sensed any contact since that reset.
"""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import traceback


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an allocated Slurm step')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=1200)
    p.add_argument('--seconds', type=float, default=40.)
    p.add_argument('--window', type=float, default=3., help='seconds after reset that count as spawn')
    p.add_argument('--backend', choices=['isaac', 'mjlab'], default='isaac')
    p.add_argument('--torch-memory-limit-gib', type=float, default=6.)
    p.add_argument('--device', default='cuda:0')
    if p.parse_known_args()[0].backend == 'isaac':
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    if args.output.exists() or args.num_envs % 6:
        raise ValueError('Fresh output and a multiple of six environments')
    args.output.mkdir(parents=True)
    if args.backend == 'isaac':
        args.headless = True
        AppLauncher(args)
    faulthandler.enable(file=(args.output/'stacks.log').open('w'))
    report = dict(completed=False, backend=args.backend, policy='none: zero latent residual, open right fingers')
    try:
        import numpy as np
        import torch
        from dextrah_lab.object_shape.evaluation import family_name
        torch.set_num_threads(4)
        total = torch.cuda.get_device_properties(args.device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., args.torch_memory_limit_gib*2**30/total), args.device)
        if args.backend == 'isaac':
            env = make_isaac_env(args)
        else:
            from dextrah_lab.mj.factory import make_mj_env
            env = make_mj_env(args.num_envs, args.device, seed=42)
        n, dt = env.num_envs, env.step_dt
        window = round(args.window/dt)
        action = torch.zeros(n, 70, device=env.device)
        action[:, 64:] = -1.   # open right fingers
        assets = env._object_asset_index_per_env.cpu().numpy()
        families = np.array([family_name(env._object_urdf_paths[int(a)]) for a in assets])
        since = torch.zeros(n, dtype=torch.long, device=env.device)
        touched = torch.zeros(n, dtype=torch.bool, device=env.device)
        counted = torch.zeros(n, dtype=torch.bool, device=env.device)
        spawns = np.zeros(n, dtype=np.int64)
        falls = np.zeros(n, dtype=np.int64)
        touched_falls = np.zeros(n, dtype=np.int64)
        settle_xy = [[] for _ in range(n)]
        env.reset()
        with torch.inference_mode():
            for step in range(round(args.seconds/dt)):
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                since += 1
                touched |= (env.touch_raw[..., 0] > 0).any(-1)
                rel = env.object.data.root_pos_w-env.scene.env_origins
                table_top = env._table_z_per_env+0.15
                fell = (rel[:, 2] < table_top-0.05) | done
                in_window = (since <= window) & ~counted
                event = in_window & fell
                if event.any():
                    ids = event.nonzero().flatten().cpu().numpy()
                    t = touched[event].cpu().numpy()
                    falls[ids] += ~t
                    touched_falls[ids] += t
                    counted[event] = True
                finished = (since == window) & ~counted
                if finished.any():
                    for i, xy in zip(finished.nonzero().flatten().tolist(), rel[finished, :2].tolist()):
                        settle_xy[i].append(xy)
                    counted[finished] = True
                spawns += (since == 1).cpu().numpy()
                since[done] = 0
                touched[done] = False
                counted[done] = False
        per_family = {}
        for fam in sorted(set(families)):
            m = families == fam
            per_family[fam] = dict(objects=int(len(set(assets[m]))), spawns=int(spawns[m].sum()),
                                   untouched_falls=int(falls[m].sum()), touched_falls=int(touched_falls[m].sum()),
                                   untouched_fall_rate=float(falls[m].sum()/max(spawns[m].sum(), 1)))
        per_object = {}
        for a in np.unique(assets):
            m = assets == a
            per_object[int(a)] = dict(family=str(families[m][0]), spawns=int(spawns[m].sum()),
                                      untouched_falls=int(falls[m].sum()))
        worst = sorted(per_object.items(), key=lambda kv: -kv[1]['untouched_falls']/max(kv[1]['spawns'], 1))[:30]
        report.update(completed=True, num_envs=n, seconds=args.seconds, window_s=args.window,
                      total_spawns=int(spawns.sum()), untouched_falls=int(falls.sum()),
                      untouched_fall_rate=float(falls.sum()/max(spawns.sum(), 1)),
                      per_family=per_family, worst_objects=[dict(asset=k, path=env._object_urdf_paths[k], **v)
                                                            for k, v in worst])
        (args.output/'per_object.json').write_text(json.dumps(per_object))
        print('SPAWN_CHECK '+json.dumps({k: report[k] for k in ('total_spawns', 'untouched_falls',
                                                                 'untouched_fall_rate', 'per_family')}), flush=True)
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        (args.output/'spawn_check.json').write_text(json.dumps(report, indent=2))
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)


def make_isaac_env(args):
    from dextrah_lab.wholebody.sonic import FrozenSonic
    from dextrah_lab.wholebody.source_config import source_task_config
    from dextrah_lab.tasks.g1_revo2_adept.g1_sonic_touch_env import G1SonicTouchEnv
    workspace = Path('/data1/users/konstantin.smirnov')
    cfg, _ = source_task_config(args.output, args.num_envs, args.device)
    cfg.seed = 42
    # Same body/table/object-physics settings as the last training run (1105).
    contract = json.loads((workspace/'DEXTRAH-G1-WholeBody/outputs/0_frozen_sonic_sapg_touch_dexterous_1105/task_contract.json').read_text())
    cfg.sonic_body.from_dict(contract['body_termination'])
    sonic = FrozenSonic(workspace/'GRAIL', workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', args.device)
    return G1SonicTouchEnv(cfg, sonic=sonic)


if __name__ == '__main__':
    main()
