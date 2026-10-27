#!/usr/bin/env python3
"""Videos of the carry expert C in its own training environment (dextrah_lab/mj/carry_env.py).

The carry run's settings (contract): holding-state starts of the manipulation expert, grasp check
after the 0.5 s warm-up, table parked below the floor (or kept, with a back-off clip), settle, walking
clips of the run's curriculum level, final stand; mass/COM randomisation and wrist pushes as trained.
Recorded worlds are evaluation worlds (holding states drawn uniformly, not success-guided), the first
--per-type of each object type by env index, chosen before the rollout. Deterministic mean actions.
"""
import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np

TYPES = ('hammer', 'screwdriver', 'marker', 'spatula', 'eraser', 'brush')
PHASE_NAMES = ('warm-up', 'settle', 'walk', 'final stand')


def record(args):
    import torch
    import mujoco_warp
    from dextrah_lab.mj.carry_env import CONTEXT_DIM, G1CarryMjEnv
    from dextrah_lab.mj.env import load_run_cfg
    from dextrah_lab.mj.factory import DEFAULT_MUJOCO, WORKSPACE
    from dextrah_lab.sgs.mj_task import SgsSettings
    from dextrah_lab.wholebody.carry_compose import CarrySettings, ClipSequencer
    from dextrah_lab.wholebody.sonic import FrozenSonic
    from mj_table_to_table import build_policy_dims
    from record_wholebody_videos import NO_RANDOMIZATION  # noqa: F401  (task randomisation left as trained)
    device = 'cuda:0'
    if 'mujoco_warp_cone_fix' not in Path(mujoco_warp.__file__).parts:
        raise RuntimeError('Training used the vendored mujoco_warp cone Hessian fix; put it on PYTHONPATH')
    run = args.checkpoint.resolve().parent.parent
    contract = json.loads((run/'task_contract.json').read_text())
    cm = contract['carry_mjlab']
    settings = CarrySettings(**{k: tuple(v) if isinstance(v, list) else v for k, v in cm['settings'].items()})
    task_run = Path(contract['task_config_source']).parent.parent
    task_contract = json.loads((task_run/'task_contract.json').read_text())
    cfg = load_run_cfg(task_run)
    cfg.seed = args.seed
    cfg.sonic_body.spawn_on_table = task_contract['body_termination']['spawn_on_table']
    holding = Path(cm['holding_state'])
    held_contract = json.loads((holding.parent.parent/'task_contract.json').read_text())
    sonic = FrozenSonic(WORKSPACE/'GRAIL', WORKSPACE/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', device)
    env = G1CarryMjEnv(cfg, sonic=sonic, carry=settings, holding_state=holding,
                       held_settings=SgsSettings(**dict(held_contract['sgs']['settings'], eval_pool_size=0)),
                       clips_path=Path(cm['clips']), num_envs=args.num_envs, device=device, nconmax=args.nconmax,
                       njmax=args.njmax, mujoco_cfg=DEFAULT_MUJOCO, seed=args.seed)
    if args.level is not None:
        env._curriculum.level = args.level
        env._seq.set_level(args.level)
    obs_dim, hand_dim = env.cfg.observation_space-CONTEXT_DIM, env.hand_surface_dim-CONTEXT_DIM
    model, payload = build_policy_dims(env, args.checkpoint, contract, device, obs_dim+CONTEXT_DIM, hand_dim+CONTEXT_DIM)
    paths = env._object_urdf_paths
    asset = env._object_asset_index_per_env.cpu().numpy()
    kind = [Path(paths[a]).stem.split('_')[1] for a in asset]
    ev = env._sgs.eval_mask.cpu().numpy()
    selected = [i for t in TYPES for i in [j for j in range(env.num_envs) if kind[j] == t and ev[j]][:args.per_type]]
    ids = torch.tensor(selected, device=device)
    n = len(selected)
    obs, _ = env.reset()
    states = tuple(s.to(device) for s in model.get_default_rnn_state())
    leader = torch.zeros(env.num_envs, 1, device=device)
    stride = max(1, round(1/(args.fps*env.step_dt)))
    a = lambda x: x.detach().cpu().numpy().copy()
    frames, events = [], {i: [] for i in selected}
    episode = np.zeros(n, np.int64)
    successes = np.zeros(n, np.int64)
    with torch.inference_mode():
        for step in range(round(args.seconds/env.step_dt)):
            if step % stride == 0:
                t = env._episode_time()
                data = env.robot.data
                yaw = torch.atan2(2*(data.root_quat_w[:, 0]*data.root_quat_w[:, 3]+data.root_quat_w[:, 1]*data.root_quat_w[:, 2]),
                                  1-2*(data.root_quat_w[:, 2]**2+data.root_quat_w[:, 3]**2))
                c, s = yaw.cos(), yaw.sin()
                v = data.root_lin_vel_w
                vb = torch.stack((c*v[:, 0]+s*v[:, 1], -s*v[:, 0]+c*v[:, 1]), -1)
                frames.append(dict(time_s=step*env.step_dt, qpos=a(env.sim.data.qpos[ids]),
                                   mocap_pos=a(env.sim.data.mocap_pos[ids]) if env.sim.mj_model.nmocap else np.zeros((n, 0, 3)),
                                   mocap_quat=a(env.sim.data.mocap_quat[ids]) if env.sim.mj_model.nmocap else np.zeros((n, 0, 4)),
                                   table=a(env.table.data.root_pos_w[ids]-env.scene.env_origins[ids]),
                                   episode_t=a(t[ids]), phase=a(env._seq.phase(t)[ids]),
                                   clip=a(env._seq.clip[ids]), command=a(env._clip_command[ids]),
                                   planned=a(env._planned_velocity[ids]), velocity=a(vb[ids]),
                                   offset_cm=a(100*env._offset_error[ids]), parked=a(env._table_parked[ids]),
                                   episode=episode.copy(), successes=successes.copy()))
            out = model(dict(obs=torch.cat((obs['policy'], leader), -1), is_train=False, prev_actions=None, rnn_states=states))
            states = out['rnn_states']
            offset_before = a(env._offset_error[ids])
            obs, _, terminated, truncated, info = env.step(out['mus'])
            done = terminated | truncated
            for s_ in states:
                s_[:, done, :] = 0
            d = a(done[ids]).astype(bool)
            tr = a(truncated[ids]).astype(bool)
            fell = a(env._body_fallen[ids]).astype(bool)
            for k, i in enumerate(selected):
                if d[k]:
                    ok = bool(tr[k]) and offset_before[k] < settings.retain_m
                    why = ('success (20 s, still holding)' if ok else 'timeout, object moved in the hand') if tr[k] else (
                        'robot fell' if fell[k] else 'object dropped / start check failed')
                    events[i].append(dict(time_s=round((step+1)*env.step_dt, 2), episode=int(episode[k]), outcome=why))
                    successes[k] += int(ok)
                    episode[k] += 1
    stacked = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
    np.savez_compressed(args.output/'trajectories.npz', **stacked)
    meta = dict(checkpoint=str(args.checkpoint), run=str(run), epoch=int(payload['epoch']), fps=args.fps, step_dt=env.step_dt,
                seconds=args.seconds, level=int(env._curriculum.level), clips=list(env.clips.names), selected=selected,
                policy='carry expert C, deterministic mean actions; evaluation worlds (uniform holding states)',
                envs=[dict(env=i, asset=int(asset[i]), type=kind[i], events=events[i]) for i in selected])
    (args.output/'recording.json').write_text(json.dumps(meta, indent=2))
    from dextrah_lab.mj.scene import object_indices
    body, slot_geoms = object_indices(env.sim.mj_model)
    bank = env.object_bank
    np.savez(args.output/'objects.npz', size=bank.size, pos=bank.pos, quat=bank.quat, used=bank.used, com=bank.com,
             slot_geoms=slot_geoms, body=body)
    return meta, cfg


def render(args, mj_model, meta):
    import copy
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from dextrah_lab.mj.scene import TABLE_SIZE
    tr = np.load(args.output/'trajectories.npz')
    ob = np.load(args.output/'objects.npz')
    font = ImageFont.load_default()
    W, H = args.width, args.height
    out_dir = args.output/'videos'
    out_dir.mkdir(exist_ok=True)
    for k, e in enumerate(meta['envs']):
        m = copy.copy(mj_model)
        a = e['asset']
        m.vis.global_.offwidth, m.vis.global_.offheight = max(W, 640), max(H, 480)
        m.vis.headlight.ambient[:] = (.45, .45, .45)
        m.vis.headlight.diffuse[:] = (.55, .55, .55)
        m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'table/table')] = (0, 0, 0, 0)   # drawn at its pose
        for s, g in enumerate(ob['slot_geoms']):
            m.geom_size[g] = ob['size'][a, s]
            m.geom_pos[g] = ob['pos'][a, s]-ob['com'][a]
            m.geom_quat[g] = ob['quat'][a, s]
            m.geom_rgba[g] = (.95, .55, .1, 1.) if ob['used'][a, s] else (0, 0, 0, 0)
            m.geom_sameframe[g] = 0
        d = mujoco.MjData(m)
        renderer = mujoco.Renderer(m, H, W)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 1
        wide, close = mujoco.MjvCamera(), mujoco.MjvCamera()
        wide.azimuth, wide.elevation, wide.distance = 135., -18., 3.2
        close.azimuth, close.elevation, close.distance = 160., -12., 1.4
        name = f"{k:02d}_{e['type']}_asset{a}_env{e['env']}.mp4"
        writer = imageio.get_writer(out_dir/name, fps=meta['fps'], codec='libx264', quality=8, macro_block_size=1)
        title = (f"carry expert C  epoch {meta['epoch']}  level {meta['level']}  |  {e['type']} asset {a} env {e['env']}  "
                 f"|  deterministic, evaluation start (uniform holding state)")
        for f in range(len(tr['time_s'])):
            d.qpos[:] = tr['qpos'][f, k]
            if tr['mocap_pos'].shape[2]:
                d.mocap_pos[:] = tr['mocap_pos'][f, k]
                d.mocap_quat[:] = tr['mocap_quat'][f, k]
            mujoco.mj_forward(m, d)
            root = d.qpos[:3]
            wide.lookat[:] = (root[0], root[1], .7)
            close.lookat[:] = (root[0], root[1], .9)
            table = tr['table'][f, k]
            views = []
            for cam in (wide, close):
                renderer.update_scene(d, camera=cam, scene_option=opt)
                sc = renderer.scene
                if table[2] > 0 and sc.ngeom < sc.maxgeom:          # parked tables are 2 m below the floor
                    mujoco.mjv_initGeom(sc.geoms[sc.ngeom], int(mujoco.mjtGeom.mjGEOM_BOX), np.array(TABLE_SIZE)/2, table,
                                        np.eye(3).flatten(), np.array([.62, .48, .33, 1.], dtype=np.float32))
                    sc.ngeom += 1
                views.append(renderer.render())
            img = Image.fromarray(np.concatenate(views, 1))
            draw = ImageDraw.Draw(img)
            clip = meta['clips'][int(tr['clip'][f, k])]
            cmd, plan, vel = tr['command'][f, k], tr['planned'][f, k], tr['velocity'][f, k]
            status = (f"t={tr['time_s'][f]:5.1f}s  episode {int(tr['episode'][f, k])+1} (t={tr['episode_t'][f, k]:4.1f}s)  "
                      f"phase {PHASE_NAMES[min(int(tr['phase'][f, k]), 3)]}  clip {clip}  "
                      f"planned vx {plan[0]:+.2f} vy {plan[1]:+.2f}  measured vx {vel[0]:+.2f} vy {vel[1]:+.2f}  "
                      f"hold offset {tr['offset_cm'][f, k]:4.1f} cm  table {'removed' if tr['parked'][f, k] else 'present'}  "
                      f"successes {int(tr['successes'][f, k])}")
            draw.rectangle((0, 0, img.width, 30), fill=(0, 0, 0))
            draw.text((6, 2), title, fill=(255, 255, 255), font=font)
            draw.text((6, 16), status, fill=(255, 230, 120), font=font)
            draw.text((6, img.height-14), 'success = still holding (within 5 cm of the grasp position in the palm) and upright after 20 s;'
                      ' drop = 12 cm off for 0.1 s or below 25 cm', fill=(255, 255, 255), font=font)
            writer.append_data(np.asarray(img))
        writer.close()
        renderer.close()
        print(f"VIDEO {out_dir/name} outcomes={[x['outcome'] for x in e['events']]}", flush=True)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside a Slurm GPU allocation (EGL)')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=1200)
    p.add_argument('--per-type', type=int, default=2)
    p.add_argument('--seconds', type=float, default=40.)
    p.add_argument('--level', type=int, default=None, help='curriculum level (default: the checkpoint run\'s)')
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--nconmax', type=int, default=64)
    p.add_argument('--njmax', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = dict(completed=False)
    try:
        from dextrah_lab.mj.compat import install
        install()
        meta, cfg = record(args)
        import record_wholebody_videos as rwv
        render(args, rwv.compile_scene(cfg), meta)
        report.update(completed=True, events={e['env']: e['events'] for e in meta['envs']})
        print('CARRY_VIDEOS_COMPLETE', flush=True)
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (args.output/'result.json').write_text(json.dumps(report, indent=2, default=str))
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)


if __name__ == '__main__':
    main()
