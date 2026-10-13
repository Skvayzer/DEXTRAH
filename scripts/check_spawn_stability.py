#!/usr/bin/env python3
"""Do objects fall off the table at spawn with nobody touching them?

Inference only, no policy: zero latent residual (frozen SONIC stands) and
fully open right fingers. Every environment keeps its assigned object
(env i -> object i % 1200), so 1200 environments cover the whole bank once
per reset. Unchanged reset rules (spawn XY box, drop height, random
orientation). A spawn counts as an untouched fall when the object goes below
the table top within the window after reset while no right fingertip has
sensed any contact since that reset. Also reports the peak object speed in
each spawn window (ejections) and hand_far terminations.
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
    p.add_argument('--spawn-on-table', action='store_true', help='Spawn fix: no table intersection, footprint on the table')
    if p.parse_known_args()[0].backend == 'isaac':
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(p)   # adds --device
    else:
        p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.output.exists() or args.num_envs % 6:
        raise ValueError('Fresh output and a multiple of six environments')
    args.output.mkdir(parents=True)
    if args.backend == 'isaac':
        args.headless = True
        AppLauncher(args)
    faulthandler.enable(file=(args.output/'stacks.log').open('w'))
    report = dict(completed=False, backend=args.backend, spawn_on_table=args.spawn_on_table, policy='none: zero latent residual, open right fingers')
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
            env = make_mj_env(args.num_envs, args.device, seed=42, spawn_on_table=args.spawn_on_table)
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
        palm = env.robot.body_names.index('right_base_link')
        near = torch.full((n,), 9., device=env.device)   # min palm-object distance since reset
        robot_falls = np.zeros(n, dtype=np.int64)
        fall_xy, fall_near = [], []
        peak = torch.zeros(n, device=env.device)   # peak object COM speed in the window
        window_peaks = []
        hand_far = 0
        env.reset()
        with torch.inference_mode():
            for step in range(round(args.seconds/dt)):
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                reasons = getattr(env, '_termination_reasons', None) or {}
                if 'hand_far' in reasons:
                    hand_far += int((done & reasons['hand_far']).sum())
                since += 1
                peak = torch.where(since <= window, torch.maximum(peak, env.object.data.root_lin_vel_w.norm(dim=-1)), peak)
                touched |= (env.touch_raw[..., 0] > 0).any(-1)
                rel = env.object.data.root_pos_w-env.scene.env_origins
                near = torch.minimum(near, (env.robot.data.body_pos_w[:, palm]-env.object.data.root_pos_w).norm(dim=-1))
                table_top = env._table_z_per_env+0.15
                fell = (rel[:, 2] < table_top-0.05) | done
                in_window = (since <= window) & ~counted
                event = in_window & fell
                if event.any():
                    ids = event.nonzero().flatten().cpu().numpy()
                    t = touched[event].cpu().numpy()
                    falls[ids] += ~t
                    touched_falls[ids] += t
                    robot = env._body_fallen[event].cpu().numpy()
                    robot_falls[ids] += robot
                    fall_xy += rel[event, :2].tolist()
                    fall_near += near[event].tolist()
                    window_peaks += peak[event].tolist()
                    counted[event] = True
                finished = (since == window) & ~counted
                if finished.any():
                    for i, xy in zip(finished.nonzero().flatten().tolist(), rel[finished, :2].tolist()):
                        settle_xy[i].append(xy)
                    window_peaks += peak[finished].tolist()
                    counted[finished] = True
                spawns += (since == 1).cpu().numpy()
                since[done] = 0
                touched[done] = False
                near[done] = 9.
                peak[done] = 0.
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
        fx, fn = np.array(fall_xy).reshape(-1, 2), np.array(fall_near)
        hx, hy = 0.475/2, 0.4/2
        edge = dict(robot_side=int((fx[:, 1] > hy).sum()), far_side=int((fx[:, 1] < -hy).sum()),
                    left_right=int(((np.abs(fx[:, 0]) > hx) & (np.abs(fx[:, 1]) <= hy)).sum()),
                    over_table=int(((np.abs(fx[:, 0]) <= hx) & (np.abs(fx[:, 1]) <= hy)).sum()))
        report.update(fall_events=len(fn), fall_edges_at_crossing=edge,
                      robot_fall_events=int(robot_falls.sum()),
                      palm_min_dist_quantiles_m=np.quantile(fn, [.05, .25, .5, .75]).tolist() if len(fn) else None,
                      falls_with_palm_never_within_10cm=int((fn > .10).sum()))
        wp_ = np.array(window_peaks)
        report.update(hand_far_terminations=hand_far,
                      window_peak_object_speed=dict(spawns=len(wp_), quantiles_50_90_99=np.quantile(wp_, [.5, .9, .99]).tolist(),
                                                    max=float(wp_.max()), over_2=float((wp_ > 2).mean()),
                                                    over_5=float((wp_ > 5).mean()), over_10=float((wp_ > 10).mean())))
        report.update(completed=True, num_envs=n, seconds=args.seconds, window_s=args.window,
                      total_spawns=int(spawns.sum()), untouched_falls=int(falls.sum()),
                      untouched_fall_rate=float(falls.sum()/max(spawns.sum(), 1)),
                      per_family=per_family, worst_objects=[dict(asset=k, path=env._object_urdf_paths[k], **v)
                                                            for k, v in worst])
        (args.output/'per_object.json').write_text(json.dumps(per_object))
        print('SPAWN_CHECK '+json.dumps({k: report[k] for k in ('total_spawns', 'untouched_falls',
            'untouched_fall_rate', 'per_family', 'fall_events', 'fall_edges_at_crossing', 'robot_fall_events',
            'palm_min_dist_quantiles_m', 'falls_with_palm_never_within_10cm', 'hand_far_terminations',
            'window_peak_object_speed')}), flush=True)
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
    cfg.sonic_body.spawn_on_table = args.spawn_on_table
    sonic = FrozenSonic(workspace/'GRAIL', workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', args.device)
    return G1SonicTouchEnv(cfg, sonic=sonic)


if __name__ == '__main__':
    main()
