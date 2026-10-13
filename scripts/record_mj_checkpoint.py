#!/usr/bin/env python3
"""Record a checkpoint on the mjlab backend in the format of the Isaac recorder.

Same selection rule (first env of each predeclared family, before rollout),
deterministic leader, ordinary resets kept, measured MuJoCo link poses. The
output directory renders with `scripts/render_g1_sonic_checkpoint.py`.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an allocated Slurm step')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=60)
    p.add_argument('--seconds', type=float, default=60.)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--families', nargs='+', default=['hammer', 'brush', 'spatula'])
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Fresh output directory required')
    args.output.mkdir(parents=True)
    report = dict(completed=False, backend='mjlab')
    try:
        import numpy as np
        import torch
        import yourdfpy
        from dextrah_lab.mj.factory import make_mj_env
        from dextrah_lab.mj.env import RUN_1105
        from dextrah_lab.object_shape.evaluation import ReposeStats, select_family_envs
        from dextrah_lab.mj.compat import p2p
        from mj_contact_capacity import build_policy
        obs_utils = p2p('obs_utils')
        device = 'cuda:0'
        env = make_mj_env(args.num_envs, device)
        model, epoch = build_policy(env, args.checkpoint, device)
        stride = round(1/(args.fps*env.step_dt))
        assignment = env._object_asset_index_per_env.cpu().numpy()
        selected = select_family_envs(env._object_urdf_paths, assignment, args.families)
        ids = torch.tensor(list(selected.values()), device=device)
        with args.checkpoint.open('rb') as f:
            sha = hashlib.file_digest(f, 'sha256').hexdigest()
        contract = json.loads((RUN_1105/'task_contract.json').read_text())
        metadata = dict(backend='mjlab', checkpoint=str(args.checkpoint), checkpoint_sha256=sha, checkpoint_epoch=epoch,
                        num_envs=env.num_envs, seed=42, fps=args.fps, seconds=args.seconds,
                        physics_hz=1/env.cfg.sim.dt, policy_hz=1/env.step_dt, goal_termination_config=env.cfg.termination.to_dict(),
                        policy='Deterministic zero-entropy leader; strict checkpoint load',
                        selection='First assigned object per predeclared family, before rollout; no outcome filtering',
                        robot_urdf=str(env.g1_urdf), body_names=env.robot.body_names, joint_names=env.robot.joint_names,
                        controller_mode='frozen_pretrained_latent', task_contract=contract,
                        body_poses='Measured MuJoCo link poses after forward(); no interpolation')
        for family, index in selected.items():
            d = args.output/family
            d.mkdir()
            path = Path(env._object_urdf_paths[int(assignment[index])])
            yourdfpy.URDF.load(str(path)).scene.export(d/'object.glb')
            (d/'object.urdf').write_text(path.read_text())
            table = Path('/data1/users/konstantin.smirnov/play2perfect/assets/urdf/table_narrow.urdf')
            yourdfpy.URDF.load(str(table)).scene.export(d/'table.glb')
            (d/'table.urdf').write_text(table.read_text())
            (d/'metadata.json').write_text(json.dumps(dict(metadata, object_family=family, env_id=index,
                                                           asset_index=int(assignment[index])), indent=2))
        obs, _ = env.reset()
        states = tuple(x.to(device) for x in model.get_default_rnn_state())
        stats = ReposeStats(env.num_envs, env.step_dt)
        falls = np.zeros(env.num_envs, dtype=np.int64)
        frames, events = [], {f: [] for f in selected}
        a = lambda x: x.detach().cpu().numpy().copy()
        with torch.inference_mode():
            for step in range(round(args.seconds/env.step_dt)):
                if step % stride == 0:
                    def pose(asset):
                        return a(torch.cat((asset.data.root_pos_w[ids], asset.data.root_quat_w[ids]), -1))
                    goal = env.goal_viz.data
                    offsets = env._keypoint_offsets_fixed[ids]
                    actual = obs_utils._keypoints_world(env.object.data.root_pos_w[ids], env.object.data.root_quat_w[ids], offsets)
                    target = obs_utils._keypoints_world(goal.root_pos_w[ids], goal.root_quat_w[ids], offsets)
                    frames.append(dict(time_s=np.full(len(ids), step*env.step_dt),
                        body_pos=a(env.robot.data.body_pos_w[ids]), body_quat=a(env.robot.data.body_quat_w[ids]),
                        robot=pose(env.robot), object=pose(env.object), table=pose(env.table),
                        goal=a(torch.cat((goal.root_pos_w[ids], goal.root_quat_w[ids]), -1)),
                        goal_error=a((actual-target).norm(dim=-1).max(dim=-1).values),
                        goals=stats.hits[list(selected.values())].copy(), resets=stats.episodes[list(selected.values())].copy(),
                        robot_falls=falls[list(selected.values())].copy(), lifted=a(env._lifted_object[ids]),
                        touch=a(env.touch_raw[ids])))
                wrapped = torch.cat((obs['policy'], obs['policy'].new_zeros(env.num_envs, 1)), -1)
                out = model(dict(obs=wrapped, is_train=False, prev_actions=None, rnn_states=states))
                states = out['rnn_states']
                obs, reward, terminated, truncated, info = env.step(out['mus'])
                done = terminated | truncated
                final = {k: a(v) for k, v in info['episode_final'].items() if torch.is_tensor(v) and v.shape == (env.num_envs,)}
                dones = a(done).astype(bool)
                hits = stats.update(final, dones, a(reward).reshape(-1))
                falls += dones & final['robot_fall'].astype(bool)
                for s in states:
                    s[:, done, :] = 0
                for family, index in selected.items():
                    if dones[index] or hits[index]:
                        events[family].append(dict(time_s=(step+1)*env.step_dt, reset=bool(dones[index]), goal_hit=int(hits[index]),
                                                   terminal={k: float(v[index]) for k, v in final.items()}))
        stacked = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
        for i, (family, index) in enumerate(selected.items()):
            d = args.output/family
            np.savez_compressed(d/'trajectory.npz', **{k: v[:, i] for k, v in stacked.items()})
            meta = json.loads((d/'metadata.json').read_text())
            meta.update(completed=True, frames=len(frames), metrics=stats.report([index]), robot_falls=int(falls[index]))
            (d/'metadata.json').write_text(json.dumps(meta, indent=2))
            (d/'events.json').write_text(json.dumps(events[family], indent=2))
        report.update(completed=True, metrics=stats.report(), robot_falls=int(falls.sum()), selected=selected)
        print('MJ_RECORDING_COMPLETE '+json.dumps(report), flush=True)
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        (args.output/'recording_result.json').write_text(json.dumps(report, indent=2))
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)


if __name__ == '__main__':
    main()
