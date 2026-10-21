#!/usr/bin/env python3
"""Videos of a fixed-body checkpoint on the mjlab task (setup sanity check).

The task is built exactly as scripts/train_g1_touch_continuation.py --backend
mjlab builds it (RUN_FIXED, per-world objects, 70 Hz touch, 1 cm tolerance;
compared against the run's saved env_resolved.yaml) with one env per asset and
the original reset distribution (no SGS). The SAPG zero-entropy leader
(coefficient 0) acts with deterministic mean actions; ordinary resets are kept.
Selection: the first ``--per-type`` envs of each object type by env index,
before the rollout (no outcome filtering).

``--starts validated`` (opt-in) starts every episode from a spawn-validated
reaching configuration of the env's own object, taken from the checkpoint's SGS
state (X.sgs.pt next to X.pth) and written with the training code path
(MjSgsTask._write_rho), so videos match what SGS training and its evaluation
envs use (no penetrating spawns). The configuration is drawn uniformly among
the object's validated ones (no outcome filtering); its training success rate
is recorded per episode.

Rendering shows what MuJoCo simulates: the compiled scene with each world's
object primitives, plus the goal pose as a translucent green copy of the object.
Two views (front, side) per frame, with episode/goal overlays.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import traceback

LEADER = 0.                                        # SAPG coefficient ids [50, 40, 30, 20, 10, 0]
COEF_IDS = (50., 40., 30., 20., 10., 0.)
TYPES = ('hammer', 'screwdriver', 'marker', 'spatula', 'eraser', 'brush')


def build_env(args, device):
    from dextrah_lab.mj.env import G1SonicTouchMjEnv, load_run_cfg, RUN_FIXED
    from dextrah_lab.mj.factory import DEFAULT_MUJOCO
    from dextrah_lab.mj.robot import ContactParams
    from dextrah_lab.mj.scene import SceneParams
    cfg = load_run_cfg(RUN_FIXED)
    cfg.scene.num_envs, cfg.sim.device = args.num_envs, device
    for key in ('success_tolerance', 'target_success_tolerance', 'eval_success_tolerance'):
        setattr(cfg.termination, key, .01)
    cfg.bps_artifact_dir = str(args.output/'bps')
    cfg.bps_cache_dir = '/data1/users/konstantin.smirnov/DEXTRAH-BPS128/outputs/bps128_cache'
    cfg.touch.material_override = False
    cfg.touch.arm_torques = False
    cfg.touch.sensor_hz = cfg.touch.publish_hz = 70.
    env = G1SonicTouchMjEnv(cfg, sonic=None, num_envs=args.num_envs, device=device, nconmax=args.nconmax,
                            njmax=args.njmax, mujoco_cfg=DEFAULT_MUJOCO, fixed_body=True,
                            scene_params=SceneParams(contact=ContactParams(g1_arm_armature=True)))
    return env, cfg


def check_against_run(cfg, run):
    """The task sections must match what the training run resolved."""
    from dextrah_lab.g1_adept.touch_continuation import load_yaml
    saved = load_yaml(Path(run)/'params/env_resolved.yaml')
    now = cfg.to_dict()
    diff = [k for k in ('reset', 'termination', 'reward', 'domain_randomization', 'touch', 'action', 'obs', 'sim',
                        'decimation', 'episode_length_s', 'assets') if json.dumps(now.get(k), sort_keys=True, default=str)
            != json.dumps(saved.get(k), sort_keys=True, default=str)]
    return diff


def build_policy(checkpoint, run, n, device):
    import torch
    from rl_games.algos_torch import model_builder
    from dextrah_lab.g1_adept.touch_policy import register_models
    from dextrah_lab.g1_adept.touch_continuation import load_yaml
    register_models()
    params = load_yaml(Path(run)/'params/agent.yaml')['params']
    model = model_builder.ModelBuilder().load(params).build(dict(
        actions_num=13, input_shape=(249+32,), num_seqs=n, value_size=1, normalize_input=True,
        normalize_value=True, type='extra_param', coef_ids=torch.tensor(COEF_IDS, device=device),
        coef_id_idx=249)).to(device)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    payload = payload[0] if 0 in payload else payload
    model.load_state_dict(payload['model'], strict=True)
    return model.eval().requires_grad_(False), int(payload['epoch']), int(payload['frame'])


class ValidatedStarts:
    """Recording-only reset hook (env._sgs interface): every reset of every env
    starts from a validated reaching configuration of that env's own object."""

    def __init__(self, env, state_path, selected, seed=0):
        import torch
        from dextrah_lab.sgs.mj_task import RHO_FIELDS
        self.env, self.device = env, env.device
        self.wholebody = False                                          # fixed-body task (MjSgsTask._write_rho)
        st = torch.load(state_path, map_location='cpu', weights_only=False)
        fams, outcomes = st['settings']['families'], st['sampler']['outcomes']
        N = outcomes.shape[0]
        sizes = [N//len(fams)+(i < N % len(fams)) for i in range(len(fams))]       # MjSgsTask family blocks
        a = sum(sizes[:fams.index('reaching')])
        b = a+sizes[fams.index('reaching')]
        keep = torch.arange(a, b)[st['validated'][a:b]]
        self.ids = keep
        self.cfg = {k: st['cfg'][k][keep].to(self.device) for k in RHO_FIELDS+('asset',)}
        self.p = outcomes[keep].float().mean(1).tolist()                          # SGS success rate (window mean)
        n_assets = len(env._object_urdf_paths)
        asset = self.cfg['asset']
        self.count = torch.bincount(asset, minlength=n_assets)
        self.order = torch.argsort(asset, stable=True)
        self.start = torch.cumsum(self.count, 0)-self.count
        if bool((self.count[env._object_asset_index_per_env] == 0).any()):
            raise RuntimeError('Some recorded objects have no validated reaching configuration')
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.selected = set(selected)
        self.log = {i: [] for i in selected}
        self.report = dict(state=str(state_path), validated_reaching_configs=int(keep.numel()),
                           reaching_configs=int(b-a))

    def before_reset(self, env_ids):
        pass

    def after_reset(self, env_ids):
        import torch
        from dextrah_lab.sgs.mj_task import MjSgsTask
        env_ids = env_ids.reshape(-1)
        assets = self.env._object_asset_index_per_env[env_ids]
        r = (torch.rand(len(env_ids), device=self.device, generator=self.generator)*self.count[assets]).long()
        k = self.order[self.start[assets]+r]
        MjSgsTask._write_rho(self, env_ids, self.cfg, k)
        for i, j in zip(env_ids.tolist(), k.tolist()):
            if i in self.selected:
                self.log[i].append(dict(config=int(self.ids[j]), train_success_rate=round(self.p[j], 3)))

    def after_step(self):
        pass


def record(args):
    import numpy as np
    import torch
    device = 'cuda:0'
    run = args.checkpoint.resolve().parent.parent
    env, cfg = build_env(args, device)
    diff = check_against_run(cfg, run)
    if diff:
        raise RuntimeError(f'Recording task differs from the training run in {diff}')
    model, epoch, frame = build_policy(args.checkpoint, run, env.num_envs, device)
    paths = env._object_urdf_paths
    asset = env._object_asset_index_per_env.cpu().numpy()
    kind = [Path(paths[a]).stem.split('_')[1] for a in asset]
    selected = [i for t in TYPES for i in [j for j in range(env.num_envs) if kind[j] == t][:args.per_type]]
    ids = torch.tensor(selected, device=device)
    starts = None
    if args.starts == 'validated':
        starts = ValidatedStarts(env, args.checkpoint.with_suffix('.sgs.pt'), selected)
        env._sgs = starts
        print('VALIDATED_STARTS '+json.dumps(starts.report), flush=True)
    obs, _ = env.reset()
    states = tuple(s.to(device) for s in model.get_default_rnn_state())
    stride = max(1, round(1/(args.fps*env.step_dt)))
    a = lambda x: x.detach().cpu().numpy().copy()
    frames, events = [], {i: [] for i in selected}
    ep_goals = np.zeros(len(selected), dtype=np.int64)
    episodes = np.zeros(len(selected), dtype=np.int64)
    total_goals = np.zeros(len(selected), dtype=np.int64)
    from dextrah_lab.mj.compat import p2p
    obs_utils = p2p('obs_utils')
    with torch.inference_mode():
        for step in range(round(args.seconds/env.step_dt)):
            if step % stride == 0:
                g = env.goal_viz.data
                offsets = env._keypoint_offsets[ids]
                err = (obs_utils._keypoints_world(env.object.data.root_pos_w[ids], env.object.data.root_quat_w[ids], offsets)
                       - obs_utils._keypoints_world(g.root_pos_w[ids], g.root_quat_w[ids], offsets)).norm(dim=-1).max(-1).values
                frames.append(dict(time_s=step*env.step_dt, qpos=a(env.sim.data.qpos[ids]),
                                   mocap_pos=a(env.sim.data.mocap_pos[ids]), mocap_quat=a(env.sim.data.mocap_quat[ids]),
                                   goal=a(torch.cat((g.root_pos_w[ids], g.root_quat_w[ids]), -1)), goal_error=a(err),
                                   lifted=a(env._lifted_object[ids]), touch=a(env.touch_raw[ids].norm(dim=-1)),
                                   episode_goals=ep_goals.copy(), episodes=episodes.copy(), total_goals=total_goals.copy()))
            wrapped = torch.cat((obs['policy'], obs['policy'].new_full((env.num_envs, 1), LEADER)), -1)
            out = model(dict(obs=wrapped, is_train=False, prev_actions=None, rnn_states=states))
            states = out['rnn_states']
            obs, reward, terminated, truncated, info = env.step(out['mus'])
            done = terminated | truncated
            for s in states:
                s[:, done, :] = 0
            d = a(done[ids]).astype(bool)
            reasons = {k: a(v[ids]) for k, v in env._termination_reasons.items()}
            final_successes = a(info['episode_final']['successes'][ids])
            now = a(env._successes[ids])
            for k, i in enumerate(selected):
                hit = (final_successes[k] if d[k] else now[k]) - ep_goals[k]
                if hit > 0:
                    total_goals[k] += hit
                    ep_goals[k] += hit
                    events[i].append(dict(time_s=(step+1)*env.step_dt, goal=int(ep_goals[k])))
                if d[k]:
                    events[i].append(dict(time_s=(step+1)*env.step_dt, reset=True, goals=int(ep_goals[k]),
                                          reason=[r for r, v in reasons.items() if v[k]]))
                    ep_goals[k], episodes[k] = 0, episodes[k]+1
    stacked = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
    np.savez_compressed(args.output/'trajectories.npz', **stacked)
    meta = dict(checkpoint=str(args.checkpoint), run=str(run), epoch=epoch, frame=frame, fps=args.fps,
                step_dt=env.step_dt, seconds=args.seconds, selected=selected, coefficient_id=LEADER,
                policy='SAPG zero-entropy leader, deterministic mean actions; '
                       +('validated SGS reaching starts' if starts else 'original resets (no SGS)'),
                starts=starts.report if starts else 'original',
                envs=[dict(env=i, asset=int(asset[i]), type=kind[i], urdf=paths[asset[i]], events=events[i],
                           **(dict(starts=starts.log[i]) if starts else {})) for i in selected])
    (args.output/'recording.json').write_text(json.dumps(meta, indent=2))
    from dextrah_lab.mj.scene import object_indices
    body, slot_geoms = object_indices(env.sim.mj_model)
    bank = env.object_bank
    np.savez(args.output/'objects.npz', size=bank.size, pos=bank.pos, quat=bank.quat, used=bank.used, com=bank.com,
             slot_geoms=slot_geoms, body=body)
    return env.sim.mj_model, meta


def compile_scene():
    """The single-world MuJoCo model of the fixed-body scene (CPU compile)."""
    from mjlab.scene import Scene
    from dextrah_lab.mj import env as E
    from dextrah_lab.mj.robot import ContactParams
    from dextrah_lab.mj.scene import SceneParams, build_scene_cfg
    ref = json.loads(E.REFERENCE_FIXED.read_text())
    cfg = E.load_run_cfg(E.RUN_FIXED)
    return Scene(build_scene_cfg(1, SceneParams(contact=ContactParams(g1_arm_armature=True)), fixed_body_reference=ref,
                                 robot_pose=(cfg.assets.robot_init_pos, cfg.assets.robot_init_rot)), 'cpu').compile()


def quat_mat(q):
    import numpy as np
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


FIXED_CAMERAS = (((0., .12, .95), 90., -20., .85), ((0., .12, .95), 180., -15., .85))   # (lookat, azimuth, elevation, distance)


def render(args, mj_model, meta, cameras=FIXED_CAMERAS):
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    tr = np.load(args.output/'trajectories.npz')
    ob = np.load(args.output/'objects.npz')
    slot_geoms = ob['slot_geoms']
    slot_type = [mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CAPSULE]*2
    font = ImageFont.load_default()
    W, H = args.width, args.height
    out_dir = args.output/'videos'
    out_dir.mkdir(exist_ok=True)
    for k, e in enumerate(meta['envs']):
        m = copy.copy(mj_model)
        a = e['asset']
        m.vis.global_.offwidth, m.vis.global_.offheight = max(W, 640), max(H, 480)
        # The scene has no lights: with the headlight alone the table top is
        # nearly as dark as the background. Brighter ambient + a coloured table.
        m.vis.headlight.ambient[:] = (.45, .45, .45)
        m.vis.headlight.diffuse[:] = (.55, .55, .55)
        m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'table/table')] = (.62, .48, .33, 1.)
        for s, g in enumerate(slot_geoms):
            m.geom_size[g] = ob['size'][a, s]
            m.geom_pos[g] = ob['pos'][a, s]-ob['com'][a]
            m.geom_quat[g] = ob['quat'][a, s]
            m.geom_rgba[g] = (.95, .55, .1, 1.) if ob['used'][a, s] else (0, 0, 0, 0)
            # The template compiled every slot at the body origin, so C MuJoCo
            # flagged it "same frame as body" and would ignore the pose written
            # here (collapsing the parts onto the COM). MuJoCo Warp has no such
            # shortcut; clear the flag so the render matches the simulation.
            m.geom_sameframe[g] = 0
        d = mujoco.MjData(m)
        renderer = mujoco.Renderer(m, H, W)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 1
        cams = []
        for lookat, az, el, dist in cameras:
            c = mujoco.MjvCamera()
            c.lookat[:] = lookat
            c.azimuth, c.elevation, c.distance = az, el, dist
            cams.append(c)
        title = f"{e['type']}  asset {a}  env {e['env']}  |  epoch {meta['epoch']}  leader, deterministic"
        writer = imageio.get_writer(out_dir/f"{k:02d}_{e['type']}_asset{a}.mp4", fps=meta['fps'], codec='libx264',
                                    quality=8, macro_block_size=1)
        for f in range(len(tr['time_s'])):
            d.qpos[:] = tr['qpos'][f, k]
            if 'mocap_pos' in tr:
                d.mocap_pos[:] = tr['mocap_pos'][f, k]
                d.mocap_quat[:] = tr['mocap_quat'][f, k]
            mujoco.mj_forward(m, d)
            gp, gq = tr['goal'][f, k, :3], tr['goal'][f, k, 3:]
            R = quat_mat(gq)
            views = []
            for c in cams:
                renderer.update_scene(d, camera=c, scene_option=opt)
                sc = renderer.scene
                for s in range(4):
                    if not ob['used'][a, s] or sc.ngeom >= sc.maxgeom:
                        continue
                    pos = gp+R @ ob['pos'][a, s]
                    mat = R @ quat_mat(ob['quat'][a, s])
                    mujoco.mjv_initGeom(sc.geoms[sc.ngeom], int(slot_type[s]), ob['size'][a, s], pos,
                                        mat.flatten(), np.array([.1, .9, .2, .35], dtype=np.float32))
                    sc.ngeom += 1
                views.append(renderer.render())
            img = Image.fromarray(np.concatenate(views, 1))
            draw = ImageDraw.Draw(img)
            status = (f"t={tr['time_s'][f]:5.2f}s  episode {tr['episodes'][f, k]+1}  goals this episode {tr['episode_goals'][f, k]}"
                      f"  total {tr['total_goals'][f, k]}  goal err {100*tr['goal_error'][f, k]:.1f} cm"
                      f"  lifted {'yes' if tr['lifted'][f, k] else 'no'}  touch N {np.round(tr['touch'][f, k], 1).tolist()}")
            draw.rectangle((0, 0, img.width, 30), fill=(0, 0, 0))
            draw.text((6, 2), title, fill=(255, 255, 255), font=font)
            draw.text((6, 16), status, fill=(255, 230, 120), font=font)
            draw.text((6, img.height-14), 'orange: object (simulated primitives)   green: goal pose   success = keypoints within 1 cm for 10 steps',
                      fill=(255, 255, 255), font=font)
            writer.append_data(np.asarray(img))
        writer.close()
        renderer.close()
        print(f"VIDEO {out_dir}/{k:02d}_{e['type']}_asset{a}.mp4 goals={int(tr['total_goals'][-1, k])} "
              f"episodes={int(tr['episodes'][-1, k])+1}", flush=True)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside a Slurm GPU allocation (EGL)')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, help='Required unless --render-only')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=1200, help='>= assets: one env per asset')
    p.add_argument('--per-type', type=int, default=2)
    p.add_argument('--seconds', type=float, default=20.)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--render-only', type=Path, help='Re-render an existing recording directory (no simulation)')
    p.add_argument('--nconmax', type=int, default=16)
    p.add_argument('--njmax', type=int, default=160)
    p.add_argument('--starts', choices=('original', 'validated'), default='original',
                   help='validated: spawn-validated SGS reaching starts from the checkpoint\'s .sgs.pt')
    args = p.parse_args()
    if not args.render_only and args.checkpoint is None:
        p.error('--checkpoint is required unless --render-only')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = dict(completed=False)
    try:
        from dextrah_lab.mj.compat import install
        install()
        if args.render_only:
            import shutil
            for name in ('trajectories.npz', 'objects.npz', 'recording.json'):
                shutil.copy2(args.render_only/name, args.output/name)
            meta = json.loads((args.output/'recording.json').read_text())
            mj_model = compile_scene()
        else:
            mj_model, meta = record(args)
        render(args, mj_model, meta)
        report.update(completed=True, videos=sorted(str(p) for p in (args.output/'videos').glob('*.mp4')))
        print('FIXED_VIDEOS_COMPLETE '+json.dumps(report), flush=True)
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (args.output/'result.json').write_text(json.dumps(report, indent=2))
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)


if __name__ == '__main__':
    main()
