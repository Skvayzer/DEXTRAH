#!/usr/bin/env python3
"""Record a strictly loaded whole-body checkpoint in its source task, without RL.

Predeclare object families before rollout. Keep ordinary resets and measured
PhysX link poses; never select successful attempts or synthesize finger motion.
"""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a Slurm GPU allocation')
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=1200)
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--families', nargs='+', default=['hammer', 'brush', 'spatula'])
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists() or not args.checkpoint.is_file():
        raise ValueError('Require an existing checkpoint and a fresh output path')
    if args.num_envs < 6 or args.num_envs % 6 or args.seconds <= 0 or args.fps <= 0:
        raise ValueError('Invalid environment count or recording duration/rate')
    args.output.mkdir(parents=True)
    args.headless = True
    app = AppLauncher(args).app
    import carb
    settings = carb.settings.get_settings()
    launch_settings = {key: settings.get(key) for key in (
        '/plugins/carb.tasking.plugin/threadCount', '/plugins/omni.tbb.globalcontrol/maxThreadCount',
        '/app/asyncRendering', '/app/asyncRenderingLowLatency')}
    print('RECORDING_LAUNCH_SETTINGS '+json.dumps(launch_settings), flush=True)
    env = None
    report = dict(completed=False, optimizer_updates=0)
    stack_log = (args.output/'stacks.log').open('w')
    faulthandler.enable(file=stack_log)
    faulthandler.dump_traceback_later(90, repeat=True, file=stack_log)
    try:
        import numpy as np
        import torch
        import yourdfpy
        from rl_games.algos_torch import model_builder
        from dextrah_lab.g1_adept.touch_continuation import load_yaml
        from dextrah_lab.wholebody.sonic import FrozenSonic
        from dextrah_lab.wholebody.source_config import source_task_config
        from dextrah_lab.wholebody.sapg_network import load_touch_sources, register_sonic_models
        from dextrah_lab.tasks.g1_revo2_adept.g1_sonic_touch_env import G1SonicTouchEnv
        from dextrah_lab.object_shape.evaluation import ReposeStats, select_family_envs
        from isaacsimenvs.tasks.play.pose_viewer import object_urdf_for_env, table_urdf_for_env
        from isaacsimenvs.tasks.play.utils.obs_utils import _keypoints_world
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        workspace = Path('/data1/users/konstantin.smirnov')
        run = args.checkpoint.parent.parent
        previous = json.loads((run/'task_contract.json').read_text())
        cfg, contract = source_task_config(args.output, args.num_envs, args.device)
        cfg.seed = args.seed
        cfg.sonic_body.numerical_failure_mode = previous['body_termination']['numerical_failure_mode']
        for key in ('source_sha256', 'bank_sha256', 'physics_hz', 'policy_hz', 'tactile_hz', 'self_collision'):
            if contract[key] != previous[key]:
                raise ValueError(f'Checkpoint task contract mismatch: {key}')
        sonic = FrozenSonic(workspace/'GRAIL', workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', args.device)
        env = G1SonicTouchEnv(cfg, sonic=sonic)
        stride = round(1/(args.fps*env.step_dt))
        if stride < 1 or not np.isclose(stride*args.fps*env.step_dt, 1):
            raise ValueError('FPS must divide policy frequency')
        source, critic = load_touch_sources(args.device)
        register_sonic_models(source, critic, sonic, env._body_lower[0], env._body_upper[0])
        params = load_yaml(run/'params/agent.yaml')['params']
        model = model_builder.ModelBuilder().load(params).build(dict(
            actions_num=35, input_shape=(1243+32,), num_seqs=args.num_envs, value_size=1,
            normalize_value=True, normalize_input=True, type='extra_param',
            coef_ids=source.a2c_network.param_ids, coef_id_idx=1243)).to(args.device)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        checkpoint = checkpoint[0] if 0 in checkpoint else checkpoint
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval().requires_grad_(False)
        with args.checkpoint.open('rb') as f:
            checkpoint_sha = hashlib.file_digest(f, 'sha256').hexdigest()
        metadata = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_sha,
            checkpoint_epoch=int(checkpoint['epoch']), checkpoint_frame=int(checkpoint['frame']),
            num_envs=env.num_envs, seed=args.seed, fps=args.fps, seconds=args.seconds,
            physics_hz=1/cfg.sim.dt, policy_hz=1/env.step_dt, tactile_hz=cfg.touch.sensor_hz,
            optimizer_updates=0, policy='Deterministic zero-entropy leader; strict checkpoint load',
            selection='First assigned object per predeclared family, before rollout; no outcome filtering',
            reset_semantics='Unchanged source task plus robot-fall/numerical termination; all resets retained',
            split='Training objects; diagnostic recording, not a generalization benchmark',
            source_commit=os.environ.get('FULLBODY_SOURCE_COMMIT'), robot_urdf=str(env.g1_urdf),
            cpu_launch=dict(slurm_cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
                pxr_worker_limit=os.environ.get('PXR_WORK_THREAD_LIMIT'),
                kit_args=getattr(args, 'kit_args', None), actual_settings=launch_settings),
            body_names=env.robot.body_names, joint_names=env.robot.joint_names,
            self_collision=cfg.assets.robot_self_collision, fabrics=False, pca=False,
            body_poses='Measured PhysX link poses; no FK substitution or pose interpolation',
            task_contract=contract)
        del checkpoint
        assignment = env._object_asset_index_per_env.cpu().numpy()
        selected = select_family_envs(env._object_urdf_paths, assignment, args.families)
        ids = torch.tensor(list(selected.values()), device=env.device)
        for family, index in selected.items():
            directory = args.output/family
            directory.mkdir()
            for name, resolver in (('object', object_urdf_for_env), ('table', table_urdf_for_env)):
                text, path = resolver(env, index)
                yourdfpy.URDF.load(str(path)).scene.export(directory/f'{name}.glb')
                (directory/f'{name}.urdf').write_text(text)
            (directory/'metadata.json').write_text(json.dumps(dict(metadata,
                object_family=family, env_id=index, asset_index=int(assignment[index])), indent=2))
        obs, _ = env.reset()
        states = tuple(x.to(args.device) for x in model.get_default_rnn_state())
        if any(x.shape[1] != args.num_envs for x in states):
            raise ValueError('Wrong recurrent state batch size')
        stats = ReposeStats(env.num_envs, env.step_dt)
        robot_falls = np.zeros(env.num_envs, dtype=np.int64)
        numerical = np.zeros_like(robot_falls)
        frames, events = [], {family: [] for family in selected}
        def array(value):
            return value.detach().cpu().numpy().copy()
        faulthandler.cancel_dump_traceback_later()
        start = time.monotonic()
        print('WHOLEBODY_RECORDING_START '+json.dumps(metadata), flush=True)
        with torch.inference_mode():
            for step in range(round(args.seconds/env.step_dt)):
                if step % stride == 0:
                    origin = env.scene.env_origins[ids]
                    def pose(asset):
                        return array(torch.cat((asset.data.root_pos_w[ids]-origin, asset.data.root_quat_w[ids]), -1))
                    offsets = env._keypoint_offsets_fixed[ids]
                    actual = _keypoints_world(env.object.data.root_pos_w[ids], env.object.data.root_quat_w[ids], offsets)
                    target = _keypoints_world(env.goal_viz.data.root_pos_w[ids], env.goal_viz.data.root_quat_w[ids], offsets)
                    frames.append(dict(time_s=np.full(len(ids), step*env.step_dt),
                        body_pos=array(env.robot.data.body_pos_w[ids]-origin[:, None]),
                        body_quat=array(env.robot.data.body_quat_w[ids]), robot=pose(env.robot),
                        object=pose(env.object), table=pose(env.table), goal=pose(env.goal_viz),
                        goal_error=array((actual-target).norm(dim=-1).max(dim=-1).values),
                        goals=stats.hits[list(selected.values())].copy(),
                        resets=stats.episodes[list(selected.values())].copy(),
                        robot_falls=robot_falls[list(selected.values())].copy(),
                        lifted=array(env._lifted_object[ids]), touch=array(env.touch_raw[ids])))
                wrapped_obs = torch.cat((obs['policy'], obs['policy'].new_zeros(env.num_envs, 1)), -1)
                output = model(dict(obs=wrapped_obs, is_train=False, prev_actions=None, rnn_states=states))
                action, states = output['mus'].clamp(-1, 1), output['rnn_states']
                if not torch.isfinite(action).all():
                    raise RuntimeError('Nonfinite checkpoint action')
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated | truncated
                if not torch.isfinite(obs['policy']).all():
                    raise RuntimeError('Nonfinite observation')
                final = {key: array(value) for key, value in info['episode_final'].items()
                         if isinstance(value, torch.Tensor) and value.shape == (env.num_envs,)}
                dones = array(done).astype(bool)
                hits = stats.update(final, dones, array(reward).reshape(-1))
                robot_falls += dones & final['robot_fall'].astype(bool)
                numerical += dones & final['numerical_failure'].astype(bool)
                for state in states:
                    state[:, done, :] = 0
                for family, index in selected.items():
                    if dones[index] or hits[index]:
                        events[family].append(dict(time_s=(step+1)*env.step_dt,
                            reset=bool(dones[index]), goal_hit=int(hits[index]),
                            terminal={key: float(value[index]) for key, value in final.items()}))
                if (step+1) % 600 == 0:
                    print('WHOLEBODY_RECORDING_PROGRESS '+json.dumps(dict(
                        sim_s=(step+1)*env.step_dt, wall_s=time.monotonic()-start,
                        completed_episodes=int(stats.episodes.sum()), goal_hits=int(stats.hits.sum()),
                        robot_falls=int(robot_falls.sum()))), flush=True)
        stacked = {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}
        for i, (family, index) in enumerate(selected.items()):
            directory = args.output/family
            np.savez_compressed(directory/'trajectory.npz', **{key: value[:, i] for key, value in stacked.items()})
            meta = json.loads((directory/'metadata.json').read_text())
            meta.update(completed=True, frames=len(frames), metrics=stats.report([index]),
                        robot_falls=int(robot_falls[index]), numerical_failures=int(numerical[index]))
            (directory/'metadata.json').write_text(json.dumps(meta, indent=2))
            (directory/'events.json').write_text(json.dumps(events[family], indent=2))
        report.update(completed=True, checkpoint_sha256=checkpoint_sha, metrics=stats.report(),
            robot_falls=int(robot_falls.sum()), numerical_failures=int(numerical.sum()),
            selected=selected, wall_seconds=time.monotonic()-start)
        print('WHOLEBODY_RECORDING_COMPLETE '+json.dumps(report), flush=True)
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        (args.output/'recording_result.json').write_text(json.dumps(report, indent=2))
        faulthandler.cancel_dump_traceback_later()
        if not report['completed']:
            import sys
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(1)
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
