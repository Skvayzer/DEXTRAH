#!/usr/bin/env python3
"""Evaluate a whole-body (frozen SONIC) mjlab checkpoint on fresh validated starts.

The task is the checkpoint run's own (params/env_resolved.yaml, same simulator
settings as training). Every episode starts from a fresh rho draw that passed the
SGS spawn check (the run's SGS state supplies the reaching set; the evaluation
pool is drawn and validated anew), with a uniformly random object of the 1200,
as the training evaluation envs (sgs/eval_*). No outcome filtering.

Policy modes (one pass each, same envs):
  det_leader    SAPG zero-entropy leader (coefficient 0), mean actions: the deployable policy
  stoch_leader  leader, actions sampled from its distribution
  stoch_blocks  each env keeps its training SAPG block coefficient, sampled actions
                (what the training evaluation envs measure)

Per env, the first ``--episodes-per-env`` completed episodes count (no bias towards
short episodes); episodes still running after ``--max-steps`` are reported as censored.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

COEF_IDS = (50., 40., 30., 20., 10., 0.)
TYPES = ('hammer', 'screwdriver', 'marker', 'spatula', 'eraser', 'brush')


class EvalStarts:
    """env._sgs hook: every reset draws a validated fresh start and its object; finished
    episodes are recorded before the task's reset clears them."""

    def __init__(self, env, task, generator):
        import torch
        self.env, self.task, self.g = env, task, generator
        self.device = env.device
        self.pick = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.started = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self.length = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.records = []
        self.recording = False

    def before_reset(self, env_ids):
        import torch
        env, ids = self.env, env_ids.reshape(-1)
        done = ids[self.started[ids]]
        if self.recording and done.numel():
            r = env._termination_reasons
            self.records.append(torch.stack((
                done.float(), env._object_asset_index_per_env[done].float(), env._successes[done].float(),
                env._episode_ever_lifted[done].float(), r['fall'][done].float(), r['hand_far'][done].float(),
                r['max_successes'][done].float(), r['timeout'][done].float(), env._body_fallen[done].float(),
                self.length[done].float()), -1))
        pool = self.task.eval_pool
        self.pick[ids] = torch.randint(0, len(pool['asset']), (ids.numel(),), device=self.device, generator=self.g)
        self.task.assign_objects(ids, pool['asset'][self.pick[ids]])
        self.started[ids] = True
        self.length[ids] = 0

    def after_reset(self, env_ids):
        ids = env_ids.reshape(-1)
        self.task._write_rho(ids, self.task.eval_pool, self.pick[ids])

    def after_step(self):
        self.length += 1


FIELDS = ('env', 'asset', 'goals', 'lifted', 'object_drop', 'hand_far', 'max_successes', 'timeout', 'robot_fall', 'steps')


def summarize(rows, types, step_dt):
    import numpy as np
    out = {}
    def stats(m):
        r = rows[m]
        if len(r) == 0:
            return dict(episodes=0)
        g = r[:, 2]
        return dict(episodes=int(len(r)), success=float((g > 0).mean()), goals_mean=float(g.mean()),
                    goals_median=float(np.median(g)), lifted=float(r[:, 3].mean()), object_drop=float(r[:, 4].mean()),
                    hand_far=float(r[:, 5].mean()), all_50_goals=float(r[:, 6].mean()), timeout=float(r[:, 7].mean()),
                    robot_fall=float(r[:, 8].mean()), episode_seconds=float(r[:, 9].mean()*step_dt))
    out['all'] = stats(np.ones(len(rows), bool))
    for t in TYPES:
        out[t] = stats(types == t)
    return out


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a Slurm GPU allocation')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=12000)
    p.add_argument('--episodes-per-env', type=int, default=2)
    p.add_argument('--max-steps', type=int, default=4000)
    p.add_argument('--modes', default='det_leader,stoch_leader,stoch_blocks')
    p.add_argument('--nconmax', type=int, default=64)
    p.add_argument('--njmax', type=int, default=300)
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
        import numpy as np
        import torch
        import mujoco_warp
        from dextrah_lab.mj.compat import install
        install()
        from rl_games.algos_torch import model_builder
        from dextrah_lab.g1_adept.touch_continuation import load_yaml
        from dextrah_lab.mj.factory import make_mj_env
        from dextrah_lab.sgs.mj_task import MjSgsTask, SgsSettings
        from dextrah_lab.wholebody.sapg_network import load_touch_sources, register_sonic_models
        device = 'cuda:0'
        torch.manual_seed(args.seed)
        run = args.checkpoint.resolve().parent.parent
        contract = json.loads((run/'task_contract.json').read_text())
        if 'mujoco_warp_cone_fix' not in Path(mujoco_warp.__file__).parts:
            raise RuntimeError('Training used the vendored mujoco_warp cone Hessian fix; put it on PYTHONPATH')
        env = make_mj_env(args.num_envs, device, nconmax=args.nconmax, njmax=args.njmax, run=run,
                          spawn_on_table=contract['body_termination']['spawn_on_table'])
        # Policy: architecture from the run's teacher, every weight from the checkpoint (strict).
        teacher = Path(contract['source_checkpoint'])
        source, critic = load_touch_sources(device, teacher, contract['source_sha256'])
        register_sonic_models(source, critic, env._sonic_reference, env._body_lower[0], env._body_upper[0], frozen=True)
        params = load_yaml(run/'params/agent.yaml')['params']
        model = model_builder.ModelBuilder().load(params).build(dict(
            actions_num=70, input_shape=(env.cfg.observation_space+32,), num_seqs=env.num_envs, value_size=1,
            normalize_value=True, normalize_input=True, type='extra_param',
            coef_ids=source.a2c_network.param_ids, coef_id_idx=env.cfg.observation_space)).to(device)
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        payload = payload[0] if 0 in payload else payload
        model.load_state_dict(payload['model'], strict=True)
        model.eval().requires_grad_(False)
        with args.checkpoint.open('rb') as f:
            sha = hashlib.file_digest(f, 'sha256').hexdigest()
        # Validated fresh starts: the run's SGS reaching set (+ a new validated pool).
        settings = SgsSettings(**contract['sgs']['settings'])
        state = torch.load(args.checkpoint.with_suffix('.sgs.pt'), map_location='cpu', weights_only=False)
        g = torch.Generator(device=device).manual_seed(args.seed)
        task = MjSgsTask(env, settings, generator=g, state=state)
        hook = EvalStarts(env, task, g)
        env._sgs = hook
        paths = env._object_urdf_paths
        asset_type = np.array([Path(x).stem.split('_')[1] for x in paths])
        block = torch.arange(env.num_envs, device=device)//(env.num_envs//6)
        coef = torch.tensor(COEF_IDS, device=device)[block]
        report.update(checkpoint_sha256=sha, epoch=int(payload['epoch']), frame=int(payload['frame']),
                      teacher=str(teacher), spawn_check=dict(task.spawn_stats), eval_pool=len(task.eval_pool['asset']))
        print('EVAL_READY '+json.dumps({k: report[k] for k in ('epoch', 'frame', 'eval_pool')}), flush=True)
        results = {}
        for mode in args.modes.split(','):
            hook.records, hook.recording = [], False
            hook.started[:] = False
            obs, _ = env.reset()
            hook.recording = True
            states = tuple(s.to(device) for s in model.get_default_rnn_state())
            ids = torch.arange(env.num_envs, device=device)
            c = coef if mode == 'stoch_blocks' else torch.zeros(env.num_envs, device=device)
            completed = torch.zeros(env.num_envs, dtype=torch.long, device=device)
            t0, steps = time.monotonic(), 0
            with torch.inference_mode():
                while steps < args.max_steps and int(completed.min()) < args.episodes_per_env:
                    out = model(dict(obs=torch.cat((obs['policy'], c[:, None]), -1), is_train=False,
                                     prev_actions=None, rnn_states=states))
                    states = out['rnn_states']
                    action = out['mus'] if mode == 'det_leader' else out['actions']
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = terminated | truncated
                    completed += done.long()
                    for s in states:
                        s[:, done, :] = 0
                    steps += 1
            rows = torch.cat(hook.records).cpu().numpy() if hook.records else np.zeros((0, len(FIELDS)))
            # First episodes_per_env completed episodes of each env (records are in completion order).
            keep = np.zeros(len(rows), bool)
            seen = np.zeros(env.num_envs, int)
            for i, e in enumerate(rows[:, 0].astype(int)):
                if seen[e] < args.episodes_per_env:
                    keep[i], seen[e] = True, seen[e]+1
            rows = rows[keep]
            np.savez_compressed(args.output/f'episodes_{mode}.npz', rows=rows, fields=np.array(FIELDS),
                                asset_type=asset_type)
            censored = int((seen < args.episodes_per_env).sum())
            types = asset_type[rows[:, 1].astype(int)]
            s = summarize(rows, types, env.step_dt)
            per_asset = np.full(len(paths), np.nan)
            for a in np.unique(rows[:, 1].astype(int)):
                per_asset[a] = float((rows[rows[:, 1] == a, 2] > 0).mean())
            s['assets_evaluated'] = int(np.isfinite(per_asset).sum())
            s['assets_never_succeeded'] = int((per_asset == 0).sum())
            s['envs_censored'] = censored
            s['steps'], s['wall_s'] = steps, time.monotonic()-t0
            results[mode] = s
            print(f'EVAL_MODE {mode} '+json.dumps(s['all'])+f' censored_envs={censored} steps={steps}', flush=True)
            (args.output/'summary.json').write_text(json.dumps(dict(report, results=results), indent=2))
        report.update(completed=True, results=results)
        print('EVAL_COMPLETE', flush=True)
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (args.output/'summary.json').write_text(json.dumps(report, indent=2))
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)


if __name__ == '__main__':
    main()
