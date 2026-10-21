#!/usr/bin/env python3
"""Videos of a whole-body (frozen SONIC) mjlab checkpoint.

The task is the checkpoint run's own (params/env_resolved.yaml, training simulator
settings). Every episode starts from scratch (object resting on the table, robot
standing) from a spawn-validated reaching configuration of the env's own object,
taken from the run's SGS state and written with the training code path; goals
follow as in training. The SAPG zero-entropy leader acts with mean actions.
Selection: the first ``--per-type`` envs of each object type by env index, before
the rollout (no outcome filtering).

``--no-randomization`` (opt-in) turns domain randomization off: observation,
action and object-state delays, object-pose noise, joint-velocity noise and the
random object pushes. Start configurations (object pose, goal, robot start pose)
are the task's own and still vary. Touch noise is already zero in this task.

Rendering reuses scripts/record_fixed_policy_videos.py (MuJoCo EGL, per-world
object primitives, translucent goal), with a full-body view plus two close-ups.
"""
import argparse
import json
import os
from pathlib import Path
import traceback

import record_fixed_policy_videos as fixed_videos

TYPES = fixed_videos.TYPES
CAMERAS = (((0., .25, .75), 135., -12., 2.3),      # full body
           ((0., .12, .95), 90., -20., .85),       # hand close-up, front
           ((0., .12, .95), 180., -15., .85))      # hand close-up, side
NO_RANDOMIZATION = dict(use_obs_delay=False, use_action_delay=False, use_object_state_delay_noise=False,
                        object_state_xyz_noise_std=0., object_state_rotation_noise_degrees=0.,
                        joint_velocity_obs_noise_std=0., force_scale=0., torque_scale=0.)


class ValidatedReaching:
    """env._sgs hook: each reset starts from a random validated reaching configuration of
    the env's own object (MjSgsTask._write_rho, whole-body standing reset included)."""

    def __init__(self, env, task, generator, selected):
        import torch
        self.env, self.task, self.g = env, task, generator
        a, b = task.blocks['reaching']
        keep = torch.arange(a, b, device=env.device)[task.validated[a:b]]
        asset = task.cfg['asset'][keep]
        self.order = keep[torch.argsort(asset, stable=True)]
        self.count = torch.bincount(asset, minlength=task.n_assets)
        self.start = torch.cumsum(self.count, 0)-self.count
        if bool((self.count[env._object_asset_index_per_env] == 0).any()):
            raise RuntimeError('Some objects have no validated reaching configuration')
        self.selected = set(selected)
        self.log = {i: [] for i in selected}

    def before_reset(self, env_ids):
        pass

    def after_reset(self, env_ids):
        import torch
        ids = env_ids.reshape(-1)
        assets = self.env._object_asset_index_per_env[ids]
        r = (torch.rand(len(ids), device=self.env.device, generator=self.g)*self.count[assets]).long()
        k = self.order[self.start[assets]+r]
        self.task._write_rho(ids, self.task.cfg, k)
        for i, j in zip(ids.tolist(), k.tolist()):
            if i in self.selected:
                self.log[i].append(dict(config=int(j), train_success_rate=round(float(self.task.sampler.outcomes[j].float().mean()), 3)))

    def after_step(self):
        pass


def build_env(args, run, contract, device):
    from dextrah_lab.mj.env import G1SonicTouchMjEnv, load_run_cfg
    from dextrah_lab.mj.factory import DEFAULT_MUJOCO, WORKSPACE
    from dextrah_lab.wholebody.sonic import FrozenSonic
    cfg = load_run_cfg(run)
    cfg.seed = args.seed
    cfg.sonic_body.spawn_on_table = contract['body_termination']['spawn_on_table']
    if args.no_randomization:
        for key, value in NO_RANDOMIZATION.items():
            if not hasattr(cfg.domain_randomization, key):
                raise KeyError(f'domain_randomization.{key} missing')
            setattr(cfg.domain_randomization, key, value)
    sonic = FrozenSonic(WORKSPACE/'GRAIL', WORKSPACE/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', device)
    return G1SonicTouchMjEnv(cfg, sonic=sonic, num_envs=args.num_envs, device=device, nconmax=args.nconmax,
                             njmax=args.njmax, mujoco_cfg=DEFAULT_MUJOCO, seed=args.seed), cfg


def record(args):
    import numpy as np
    import torch
    import mujoco_warp
    from dextrah_lab.sgs.mj_task import MjSgsTask, SgsSettings
    from eval_wholebody_checkpoint import build_policy
    device = 'cuda:0'
    run = args.checkpoint.resolve().parent.parent
    contract = json.loads((run/'task_contract.json').read_text())
    if 'mujoco_warp_cone_fix' not in Path(mujoco_warp.__file__).parts:
        raise RuntimeError('Training used the vendored mujoco_warp cone Hessian fix; put it on PYTHONPATH')
    env, cfg = build_env(args, run, contract, device)
    model, payload = build_policy(env, args.checkpoint, contract, device)
    paths = env._object_urdf_paths
    asset = env._object_asset_index_per_env.cpu().numpy()
    kind = [Path(paths[a]).stem.split('_')[1] for a in asset]
    selected = [i for t in TYPES for i in [j for j in range(env.num_envs) if kind[j] == t][:args.per_type]]
    ids = torch.tensor(selected, device=device)
    settings = SgsSettings(**dict(contract['sgs']['settings'], eval_pool_size=0))
    state = torch.load(args.checkpoint.with_suffix('.sgs.pt'), map_location='cpu', weights_only=False)
    g = torch.Generator(device=device).manual_seed(args.seed)
    task = MjSgsTask(env, settings, generator=g, state=state)
    starts = ValidatedReaching(env, task, g, selected)
    env._sgs = starts
    obs, _ = env.reset()
    states = tuple(s.to(device) for s in model.get_default_rnn_state())
    stride = max(1, round(1/(args.fps*env.step_dt)))
    a = lambda x: x.detach().cpu().numpy().copy()
    frames, events = [], {i: [] for i in selected}
    n = len(selected)
    ep_goals, episodes, total = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64)
    leader = torch.zeros(env.num_envs, 1, device=device)
    with torch.inference_mode():
        for step in range(round(args.seconds/env.step_dt)):
            if step % stride == 0:
                gv = env.goal_viz.data
                frames.append(dict(time_s=step*env.step_dt, qpos=a(env.sim.data.qpos[ids]),
                                   goal=a(torch.cat((gv.root_pos_w[ids], gv.root_quat_w[ids]), -1)),
                                   goal_error=a(env._keypoints_max_dist[ids]), lifted=a(env._lifted_object[ids]),
                                   touch=a(env.touch_raw[ids].norm(dim=-1)), episode_goals=ep_goals.copy(),
                                   episodes=episodes.copy(), total_goals=total.copy()))
            out = model(dict(obs=torch.cat((obs['policy'], leader), -1), is_train=False, prev_actions=None, rnn_states=states))
            states = out['rnn_states']
            obs, _, terminated, truncated, info = env.step(out['mus'])
            done = terminated | truncated
            for s in states:
                s[:, done, :] = 0
            d = a(done[ids]).astype(bool)
            reasons = {k: a(v[ids]) for k, v in env._termination_reasons.items()}
            robot_fall = a(info['episode_final']['robot_fall'][ids]).astype(bool)
            final = a(info['episode_final']['successes'][ids])
            now = a(env._successes[ids])
            for k, i in enumerate(selected):
                hit = (final[k] if d[k] else now[k])-ep_goals[k]
                if hit > 0:
                    total[k] += hit
                    ep_goals[k] += hit
                    events[i].append(dict(time_s=(step+1)*env.step_dt, goal=int(ep_goals[k])))
                if d[k]:
                    why = [r for r, v in reasons.items() if v[k]]+(['robot_fall'] if robot_fall[k] else [])
                    events[i].append(dict(time_s=(step+1)*env.step_dt, reset=True, goals=int(ep_goals[k]), reason=why))
                    ep_goals[k], episodes[k] = 0, episodes[k]+1
    stacked = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
    np.savez_compressed(args.output/'trajectories.npz', **stacked)
    meta = dict(checkpoint=str(args.checkpoint), run=str(run), epoch=int(payload['epoch']), frame=int(payload['frame']),
                fps=args.fps, step_dt=env.step_dt, seconds=args.seconds, selected=selected, coefficient_id=0.,
                policy='SAPG zero-entropy leader, deterministic mean actions; validated reaching starts (from scratch)',
                domain_randomization='off: '+', '.join(NO_RANDOMIZATION) if args.no_randomization else 'as in training',
                envs=[dict(env=i, asset=int(asset[i]), type=kind[i], urdf=paths[asset[i]], events=events[i],
                           starts=starts.log[i]) for i in selected])
    (args.output/'recording.json').write_text(json.dumps(meta, indent=2))
    from dextrah_lab.mj.scene import object_indices
    body, slot_geoms = object_indices(env.sim.mj_model)
    bank = env.object_bank
    np.savez(args.output/'objects.npz', size=bank.size, pos=bank.pos, quat=bank.quat, used=bank.used, com=bank.com,
             slot_geoms=slot_geoms, body=body)
    return meta, cfg


def compile_scene(cfg):
    """Single-world MuJoCo model of the whole-body scene (CPU compile)."""
    from mjlab.scene import Scene
    from dextrah_lab.mj.scene import SceneParams, build_scene_cfg
    return Scene(build_scene_cfg(1, SceneParams(), robot_pose=(cfg.assets.robot_init_pos, cfg.assets.robot_init_rot)),
                 'cpu').compile()


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside a Slurm GPU allocation (EGL)')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=1200, help='>= assets: one env per asset')
    p.add_argument('--per-type', type=int, default=2)
    p.add_argument('--seconds', type=float, default=30.)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--nconmax', type=int, default=64)
    p.add_argument('--njmax', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-randomization', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = dict(completed=False)
    try:
        from dextrah_lab.mj.compat import install
        install()
        meta, cfg = record(args)
        fixed_videos.render(args, compile_scene(cfg), meta, cameras=CAMERAS)
        report.update(completed=True, videos=sorted(str(v) for v in (args.output/'videos').glob('*.mp4')))
        print('WB_VIDEOS_COMPLETE '+json.dumps(report), flush=True)
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
