#!/usr/bin/env python3
"""Frozen manipulation expert (M) on flat-ended cylinders: grasp rate and hold survival per radius.

The checkpoint run's task and simulator (params/env_resolved.yaml; condim / nconmax / njmax from task_contract.json)
with the opt-in cylinder object set (dextrah_lab/mj/cylinder_objects.py: a fifth, true-cylinder object slot), or the
run's own tools through the same path (--object-set tools, the control). Starts: fresh rho draws validated by the SGS
spawn check (the run's SGS settings, reaching only, no SGS state: the stored configurations belong to the tool pool);
with --spawn upright every draw stands its cylinder upright (random yaw, resting on the table) before validation,
with --spawn env the task's own random-orientation drop is kept (validation then keeps cylinders that came to rest).

Per episode (first episode per env, deterministic leader): lifted (the env's lift flag), the first lift step, the
first step after the lift at which the object touches the table or the floor again ("hold lost"; M never sets
objects down on purpose), goals reached, object fall, timeouts. Summary per radius: grasp rate, Kaplan-Meier hold
survival 1 / 2 / 5 s after the lift (censored at the episode end), drop to the floor after a lift, goal success.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

NO_RANDOMIZATION = dict(use_obs_delay=False, use_action_delay=False, use_object_state_delay_noise=False,   # as
                        object_state_xyz_noise_std=0., object_state_rotation_noise_degrees=0.,            # record_wholebody_videos
                        joint_velocity_obs_noise_std=0., force_scale=0., torque_scale=0.)
FIELDS = ('env', 'asset', 'goals', 'lifted', 'lift_step', 'lost_step', 'object_drop', 'hand_far', 'timeout',
          'robot_fall', 'steps', 'censored')


class CylinderEval:
    """env._sgs hook (as eval_wholebody_checkpoint.EvalStarts) with lift / hold-loss tracking."""

    def __init__(self, env, task, generator):
        import torch
        self.env, self.task, self.g, d = env, task, generator, env.device
        n = env.num_envs
        self.pick = torch.zeros(n, dtype=torch.long, device=d)
        self.started = torch.zeros(n, dtype=torch.bool, device=d)
        self.length = torch.zeros(n, dtype=torch.long, device=d)
        self.lift_step = torch.full((n,), -1, dtype=torch.long, device=d)
        self.lost_step = torch.full((n,), -1, dtype=torch.long, device=d)
        self.records, self.recording = [], False

    def _row(self, ids, censored):
        import torch
        env, r = self.env, self.env._termination_reasons
        z = torch.zeros(ids.numel(), device=self.env.device)
        reasons = [r[k][ids].float() for k in ('fall', 'hand_far', 'timeout')] if not censored else [z, z, z]
        return torch.stack((ids.float(), env._object_asset_index_per_env[ids].float(), env._successes[ids].float(),
                            env._episode_ever_lifted[ids].float(), self.lift_step[ids].float(), self.lost_step[ids].float(),
                            *reasons, env._body_fallen[ids].float() if not censored else z, self.length[ids].float(),
                            torch.full_like(z, float(censored))), -1)

    def before_reset(self, env_ids):
        import torch
        ids = env_ids.reshape(-1)
        done = ids[self.started[ids]]
        if self.recording and done.numel():
            self.records.append(self._row(done, False))
        pool = self.task.eval_pool
        self.pick[ids] = torch.randint(0, len(pool['asset']), (ids.numel(),), device=self.env.device, generator=self.g)
        self.task.assign_objects(ids, pool['asset'][self.pick[ids]])
        self.started[ids] = True
        self.length[ids] = 0
        self.lift_step[ids] = -1
        self.lost_step[ids] = -1

    def after_reset(self, env_ids):
        self.task._write_rho(env_ids.reshape(-1), self.task.eval_pool, self.pick[env_ids.reshape(-1)])

    def after_step(self):
        env = self.env
        self.length += 1
        new = (self.lift_step < 0) & env._episode_ever_lifted
        self.lift_step[new] = self.length[new]
        c = getattr(env, '_last_contacts', None) or {}
        support = (c['object_table_force'] > 0) | (c['object_ground_force'] > 0)
        lost = (self.lift_step >= 0) & (self.lost_step < 0) & (self.length > self.lift_step) & support
        self.lost_step[lost] = self.length[lost]

    def flush_running(self):
        import torch
        ids = torch.nonzero(self.started).squeeze(-1)
        self.records.append(self._row(ids, True))


def km(durations, events, times):
    """Kaplan-Meier survival at times (durations in steps; events: 1 lost, 0 censored)."""
    order = np.argsort(durations, kind='stable')
    d, e = durations[order], events[order]
    s, at_risk, out, k = 1., len(d), [], 0
    for t in times:
        while k < len(d) and d[k] <= t:
            j = k
            while j < len(d) and d[j] == d[k]:
                j += 1
            lost = int(e[k:j].sum())
            if at_risk > 0:
                s *= 1-lost/at_risk
            at_risk -= j-k
            k = j
        out.append(round(s, 4))
    return out


def summarize(rows, step_dt):
    if len(rows) == 0:
        return dict(episodes=0)
    f = {k: rows[:, i] for i, k in enumerate(FIELDS)}
    lifted = f['lift_step'] >= 0
    out = dict(episodes=int(len(rows)), grasp_rate=round(float(lifted.mean()), 4),
               success=round(float((f['goals'] > 0).mean()), 4), goals_mean=round(float(f['goals'].mean()), 3),
               object_drop=round(float(f['object_drop'].mean()), 4), hand_far=round(float(f['hand_far'].mean()), 4),
               timeout=round(float(f['timeout'].mean()), 4), robot_fall=round(float(f['robot_fall'].mean()), 4),
               censored=int(f['censored'].sum()))
    if lifted.any():
        L = rows[lifted]
        lift, lost, steps = L[:, FIELDS.index('lift_step')], L[:, FIELDS.index('lost_step')], L[:, FIELDS.index('steps')]
        event = lost >= 0
        dur = np.where(event, lost-lift, steps-lift)
        secs = (1., 2., 5.)
        out['hold_survival_after_lift'] = dict(zip([f'{s:g}s' for s in secs], km(dur, event, [s/step_dt for s in secs])))
        out['hold_lost_after_lift'] = round(float(event.mean()), 4)
        out['floor_drop_after_lift'] = round(float(L[:, FIELDS.index('object_drop')].mean()), 4)
        out['lift_time_s_median'] = round(float(np.median(lift)*step_dt), 3)
    return out


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a Slurm GPU allocation')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--object-set', default='cylinders', choices=('cylinders', 'tools'))
    p.add_argument('--spawn', default='upright', choices=('upright', 'env'))
    p.add_argument('--radii', default='0.025,0.0275,0.03')
    p.add_argument('--per-radius', type=int, default=100)
    p.add_argument('--length', default='0.08,0.20')
    p.add_argument('--mass', default='0.1,0.5')
    p.add_argument('--num-envs', type=int, default=3000)
    p.add_argument('--max-steps', type=int, default=1500)
    p.add_argument('--eval-pool', type=int, default=4096)
    p.add_argument('--no-randomization', action='store_true', help='no observation / action delays, noise, pushes')
    p.add_argument('--bps-cache', type=Path, default=Path('outputs/bps_cache_cylinders'))
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.num_envs % 6:
        raise ValueError('num_envs must be a multiple of six (SAPG blocks)')
    args.output.mkdir(parents=True)
    report = dict(completed=False, checkpoint=str(args.checkpoint), args={k: str(v) for k, v in vars(args).items()})
    try:
        import hashlib
        import torch
        import mujoco_warp
        from dextrah_lab.mj.compat import install
        install()
        from dextrah_lab.mj.factory import make_mj_env
        from dextrah_lab.mj.robot import ContactParams
        from dextrah_lab.mj.scene import SceneParams
        from dextrah_lab.sgs.mj_task import MjSgsTask, SgsSettings
        from eval_wholebody_checkpoint import build_policy
        device = 'cuda:0'
        torch.manual_seed(args.seed)
        run = args.checkpoint.resolve().parent.parent
        contract = json.loads((run/'task_contract.json').read_text())
        if 'mujoco_warp_cone_fix' not in Path(mujoco_warp.__file__).parts:
            raise RuntimeError('Training used the vendored mujoco_warp cone Hessian fix; put it on PYTHONPATH')
        sim = contract['simulator']
        condim = int(sim['contact']['condim'])
        scene_params = SceneParams(contact=ContactParams(condim=condim))
        radii = tuple(float(x) for x in args.radii.split(','))
        cyl = dict(radii=radii, per_radius=args.per_radius, length=tuple(float(x) for x in args.length.split(',')),
                   mass=tuple(float(x) for x in args.mass.split(',')), seed=args.seed)

        def configure(cfg):
            if args.object_set == 'cylinders':
                cfg.assets.object_set = 'cylinders'
                cfg.assets.cylinder_params = cyl
                cfg.bps_cache_dir = str(args.bps_cache.resolve())
            if args.no_randomization:
                for k, v in NO_RANDOMIZATION.items():
                    if not hasattr(cfg.domain_randomization, k):
                        raise KeyError(f'domain_randomization.{k} missing')
                    setattr(cfg.domain_randomization, k, v)
        env = make_mj_env(args.num_envs, device, nconmax=int(sim['nconmax']), njmax=int(sim['njmax']), run=run,
                          spawn_on_table=contract['body_termination']['spawn_on_table'], scene_params=scene_params,
                          configure=configure)
        model, payload = build_policy(env, args.checkpoint, contract, device)
        with args.checkpoint.open('rb') as f:
            sha = hashlib.file_digest(f, 'sha256').hexdigest()
        g = torch.Generator(device=device).manual_seed(args.seed)
        s = dict(contract['sgs']['settings'], families=('reaching',), num_configs=max(600, 2*len(env._object_urdf_paths)),
                 reharvest_every=0, hard_object_share=0., held_max_body_action=0., eval_pool_size=args.eval_pool)
        settings = SgsSettings(**{k: v for k, v in s.items() if k in SgsSettings.__dataclass_fields__})
        if args.object_set == 'cylinders' and args.spawn == 'upright':
            half_len = torch.as_tensor(env.object_set_meta[:, 1]/2, device=device, dtype=torch.float32)
            orig = MjSgsTask._capture_reaching

            def upright(self, n, assets=None):
                """Cylinder axis (link x) vertical, random yaw, resting on the table top (+2 mm)."""
                cand = orig(self, n, assets)
                a = assets if assets is not None else cand['asset']
                yaw = torch.rand(len(a), device=device, generator=g)*2*np.pi
                qy = torch.stack((torch.cos(yaw/2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw/2)), -1)
                qx = torch.tensor([np.cos(-np.pi/4), 0., np.sin(-np.pi/4), 0.], device=device).expand_as(qy)  # R_y(-90): x -> z
                w1, x1, y1, z1 = qy.unbind(-1)
                w2, x2, y2, z2 = qx.unbind(-1)
                q = torch.stack((w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2,
                                 w1*z2+x1*y2-y1*x2+z1*w2), -1)
                pose = cand['object_pose'].clone()
                pose[:, 2] = cand['table_z']+.15+half_len[a]+.002
                pose[:, 3:] = q
                cand['object_pose'] = pose
                return cand
            MjSgsTask._capture_reaching = upright
        task = MjSgsTask(env, settings, generator=g, state=None)
        hook = CylinderEval(env, task, g)
        env._sgs = hook
        if not hasattr(env, 'contacts'):
            env._init_contacts()
        env.contacts.object_table = env.contacts.object_ground = True
        paths = env._object_urdf_paths
        if args.object_set == 'cylinders':
            meta = env.object_set_meta
            group = np.array([f'r{1000*r:.2f}mm' for r in meta[:, 0]])
        else:
            group = np.array([Path(x).stem.split('_')[1] for x in paths])
        report.update(checkpoint_sha256=sha, epoch=int(payload['epoch']), condim=condim, assets=len(paths),
                      spawn_check=dict(task.spawn_stats), eval_pool=len(task.eval_pool['asset']),
                      object_set_record=getattr(env, 'object_set_record', None))
        torch.cuda.empty_cache()
        print('EVAL_READY '+json.dumps({k: report[k] for k in ('epoch', 'assets', 'eval_pool', 'spawn_check')}), flush=True)
        hook.records, hook.recording = [], False
        hook.started[:] = False
        with torch.inference_mode():
            obs, _ = env.reset()
        hook.recording = True
        states = tuple(x.to(device) for x in model.get_default_rnn_state())
        c = torch.zeros(env.num_envs, device=device)
        completed = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        t0, steps = time.monotonic(), 0
        with torch.inference_mode():
            while steps < args.max_steps and int(completed.min()) < 1:
                out = model(dict(obs=torch.cat((obs['policy'], c[:, None]), -1), is_train=False, prev_actions=None,
                                 rnn_states=states))
                states = out['rnn_states']
                obs, _, terminated, truncated, _ = env.step(out['mus'])
                done = terminated | truncated
                completed += done.long()
                for x in states:
                    x[:, done, :] = 0
                steps += 1
        hook.flush_running()
        rows = torch.cat(hook.records).cpu().numpy()
        first = np.zeros(len(rows), bool)
        seen = set()
        for i, e in enumerate(rows[:, 0].astype(int)):
            if e not in seen:
                first[i] = True
                seen.add(e)
        rows = rows[first]
        np.savez_compressed(args.output/'episodes.npz', rows=rows, fields=np.array(FIELDS), group=group)
        grp = group[rows[:, 1].astype(int)]
        res = dict(all=summarize(rows, env.step_dt), steps=steps, wall_s=round(time.monotonic()-t0, 1))
        for k in sorted(set(group)):
            res[k] = summarize(rows[grp == k], env.step_dt)
        if args.object_set == 'cylinders':
            m = env.object_set_meta[rows[:, 1].astype(int)]
            for name, col, edges in (('length', 1, (.08, .12, .16, .2001)), ('mass', 2, (.1, .233, .367, .5001))):
                for lo, hi in zip(edges[:-1], edges[1:]):
                    res[f'{name}_{lo:g}-{hi:.3g}'] = summarize(rows[(m[:, col] >= lo) & (m[:, col] < hi)], env.step_dt)
        report.update(completed=True, results=res)
        print('EVAL_COMPLETE '+json.dumps(res['all']), flush=True)
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (args.output/'summary.json').write_text(json.dumps(report, indent=2, default=str))
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)


if __name__ == '__main__':
    main()
