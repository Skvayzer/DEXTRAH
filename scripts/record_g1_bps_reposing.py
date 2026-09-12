#!/usr/bin/env python3
"""Evaluate the final BPS SAPG actor and capture predeclared continuous clips.

No optimizer, training, W&B run, changed reward, or outcome-based clip selection.
"""
import argparse
import copy
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run capture inside a Slurm GPU allocation')
    # Report a real stack if Kit/PhysX initialization stops progressing.
    # Diagnostic only: no changes to environment or evaluation protocol.
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--play2perfect-root',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=1200)
    p.add_argument('--seconds',type=float,default=120.)
    p.add_argument('--video-seconds',type=float,default=60.)
    p.add_argument('--fps',type=int,default=30)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--metrics-only',action='store_true',help='Frozen-policy evaluation without recording assets or trajectories')
    p.add_argument('--object-seed',type=int,default=42)
    p.add_argument('--wrench-multiplier',type=float,default=1.)
    p.add_argument('--capture-reference',action='store_true',help='Also retain 60 Hz achieved motion and held joint targets for offline reference construction')
    p.add_argument('--families',nargs='+',default=['hammer','spatula','brush','eraser'])
    AppLauncher.add_app_launcher_args(p)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not (0 < args.video_seconds <= args.seconds) or args.num_envs%6:
        raise ValueError('Invalid duration or six-group environment count')
    if not math.isfinite(args.wrench_multiplier) or args.wrench_multiplier <= 0:
        raise ValueError('Invalid wrench multiplier')
    if args.capture_reference and args.metrics_only:
        raise ValueError('Reference capture requires selected recording environments')
    args.headless=True
    app=AppLauncher(args).app
    # We are a standalone headless process, not an interactive Kit extension.
    # In this workstation's IsaacLab that distinction disables the STOP-event
    # callback which otherwise waits forever for a UI "play" command.
    import builtins
    builtins.ISAAC_LAUNCHED_FROM_TERMINAL=True
    args.output.mkdir(parents=True)
    stack_log=(args.output/'stacks.log').open('w')
    faulthandler.enable(file=stack_log)
    faulthandler.dump_traceback_later(180,repeat=True,file=stack_log)
    try:
        import numpy as np
        import torch
        import yaml
        import yourdfpy
        from isaaclab.utils import replace_strings_with_slices
        from isaacsimenvs.tasks.play.pose_viewer import object_urdf_for_env, table_urdf_for_env
        from isaacsimenvs.tasks.play.utils import scene_utils
        from isaacsimenvs.tasks.play.utils.obs_utils import _keypoints_world
        from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
        from rl_games.torch_runner import Runner
        from dextrah_lab.tasks.g1_revo2_adept.g1_revo2_bps_env import G1Revo2BpsEnv, G1Revo2BpsEnvCfg
        from dextrah_lab.object_shape.bank import load_primitive_urdf
        from dextrah_lab.object_shape.evaluation import ReposeStats, family_name, select_family_envs, bps_display_geometry

        torch.set_num_threads(2)
        class TupleLoader(yaml.SafeLoader):
            pass
        TupleLoader.add_constructor('tag:yaml.org,2002:python/tuple',
            lambda loader,node:tuple(loader.construct_sequence(node)))
        saved={name:yaml.load((args.run/'params'/f'{name}.yaml').read_text(),Loader=TupleLoader)
               for name in ('env_resolved','agent')}
        # env_resolved.yaml stores the task's integer space dimensions, not
        # serialized Gym spaces. Preserve their types while restoring config.
        tactile=saved['agent']['params']['model']['name']=='continuous_a2c_logstd_touch'
        if tactile:
            from dextrah_lab.g1_adept.touch_policy import register_models
            from dextrah_lab.tasks.g1_revo2_adept.g1_revo2_touch_env import G1Revo2TouchEnv, G1Revo2TouchEnvCfg
            register_models()
        cfg=G1Revo2TouchEnvCfg() if tactile else G1Revo2BpsEnvCfg()
        cfg.seed=saved['env_resolved']['seed']
        # IsaacLab's updater validates against the current value's type rather
        # than the Optional annotation. Continuation explicitly pins this field.
        cfg.termination.eval_success_tolerance=saved['env_resolved']['termination']['eval_success_tolerance']
        cfg.from_dict(replace_strings_with_slices(copy.deepcopy(saved['env_resolved'])))
        cfg.seed=args.seed
        cfg.scene.num_envs=args.num_envs
        cfg.sim.device=args.device
        cfg.termination.eval_success_tolerance=.01
        cfg.bps_artifact_dir=str(args.output/'bps')
        cfg.domain_randomization.force_scale *= args.wrench_multiplier
        cfg.domain_randomization.torque_scale *= args.wrench_multiplier
        # Keep solver/action/reward/DR settings intact; lower only buffer capacity.
        for name,value in dict(gpu_found_lost_pairs_capacity=2**20,
            gpu_found_lost_aggregate_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**20,gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**18,gpu_collision_stack_size=2**26).items():
            setattr(cfg.sim.physx,name,value)
        assert not cfg.fabric.enabled and not cfg.fabric.pca_enabled
        dt=cfg.sim.dt*cfg.decimation
        stride=round(1/(dt*args.fps))
        if stride<1 or not math.isclose(stride*dt*args.fps,1.):
            raise ValueError('Video FPS must divide the policy rate')
        steps=round(args.seconds/dt)
        clip_steps=round(args.video_seconds/dt)
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        checkpoint=checkpoint[0] if 0 in checkpoint else checkpoint
        actor_dim,critic_dim=(249,271) if tactile else (224,246)
        if checkpoint['model']['running_mean_std.running_mean'].shape != (actor_dim,):
            raise ValueError('Checkpoint/config observation layout mismatch')
        from dextrah_lab.object_shape.teacher_suite import object_seed_override
        source_manifest=json.loads((args.run/'bps/manifest.json').read_text())
        with object_seed_override(scene_utils,args.object_seed,source_manifest) as shape_audit:
            env=(G1Revo2TouchEnv if tactile else G1Revo2BpsEnv)(cfg)
        try:
            if args.object_seed==42:
                assert env._bps_manifest['features_sha256']==source_manifest['features_sha256'], 'Wrong training shape bank'
            else:
                assert shape_audit['training_geometry_overlap']==0
            assert len(env._bps_bank)==1200 and env.cfg.action_space==13
            indices=env._object_asset_index_per_env.cpu().numpy()
            if args.num_envs==1200:
                assert len(np.unique(indices))==1200
            selected={} if args.metrics_only else select_family_envs(env._object_urdf_paths,indices,args.families)
            ids=torch.tensor(list(selected.values()),device=env.device)
            agent=copy.deepcopy(saved['agent'])
            ac=agent['params']['config']
            ac.update(num_actors=args.num_envs,expl_coef_block_size=args.num_envs//6,
                      device=args.device,device_name=args.device,multi_gpu=False)
            ac['player'].update(deterministic=True,print_stats=False,evaluation=False)
            wrapped=register_rlgames_env(env,rl_device=args.device,
                clip_obs=agent['params']['env']['clip_observations'],
                clip_actions=agent['params']['env']['clip_actions'])
            runner=Runner()
            runner.load(agent)
            player=runner.create_player()
            player.set_weights(checkpoint)
            player.has_batch_dimension=True
            player.intr_reward_coef_embd.fill_(0.)
            player.reset()
            obs=player.env_reset(wrapped)
            player.get_batch_size(obs,1)
            assert obs.shape==(args.num_envs,actor_dim+1)
            for state in player.states:
                assert state.shape[1]==args.num_envs

            def array(value):
                return value.detach().cpu().numpy().copy()

            metadata=dict(checkpoint=str(args.checkpoint),checkpoint_epoch=int(checkpoint['epoch']),
                checkpoint_training_frames=int(checkpoint['frame']),
                checkpoint_sha256=hashlib.file_digest(args.checkpoint.open('rb'),'sha256').hexdigest(),
                source_run=str(args.run),num_envs=env.num_envs,objects=len(env._bps_bank),
                bank_sha256=env._bps_manifest['features_sha256'],seed=args.seed,policy_dt=dt,
                evaluation_seconds=steps*dt,video_seconds=clip_steps*dt,fps=args.fps,
                actor_dim=actor_dim,critic_dim=critic_dim,actions=13,coefficient_id=0.,
                policy='Strictly loaded SAPG zero-entropy leader; deterministic mean actions',
                tolerance_parameter=.01,max_keypoint_error_m=.01*cfg.reward.keypoint_scale,
                success_steps=cfg.termination.success_steps,
                consecutive_success_steps=cfg.termination.force_consecutive_near_goal_steps,
                domain_randomization='Saved training delays/noise/random wrenches and reset variation retained',
                resets='Original fall, hand-distance, per-goal timeout and 50-goal cap; visible in videos',
                selection='First predeclared asset per family, before rollout; no outcome-based selection',
                split='Training objects, not held-out generalization evaluation',selected_envs=selected,
                robot_urdf=str(args.play2perfect_root/cfg.assets.robot_urdf),
                visual_static_joint_pos=scene_utils.G1_BODY_DEFAULT_JOINT_POS,
                joint_names=env.robot.joint_names,body_names=env.robot.body_names,
                optimizer_updates=0,fabrics=False,pca=False,tactile=tactile,metrics_only=args.metrics_only)
            metadata.update(object_seed=args.object_seed,shape_audit=shape_audit,
                wrench_multiplier=args.wrench_multiplier,capture_reference=args.capture_reference,
                split=('Training objects' if args.object_seed==42 else 'Held-out geometry from the same six procedural families'),
                domain_randomization='Saved delays/noise/reset variation; object wrench magnitudes multiplied by '+str(args.wrench_multiplier),
                reference_target_semantics='Previous applied position target at pre-step state; achieved positions/velocities recorded separately',
                reference_dt=dt if args.capture_reference else None)
            for family,e in selected.items():
                directory=args.output/'clips'/family
                directory.mkdir(parents=True)
                for name,resolver in (('object',object_urdf_for_env),('table',table_urdf_for_env)):
                    text,path=resolver(env,e)
                    yourdfpy.URDF.load(str(path)).scene.export(directory/f'{name}.glb')
                    (directory/f'{name}.urdf').write_text(text)
                    if name=='object':
                        geom=bps_display_geometry(load_primitive_urdf(path),array(env._bps_features[e]))
                        np.savez_compressed(directory/'bps_geometry.npz',**geom)
                clip_meta=dict(metadata,object_family=family,env_id=e,asset_index=int(indices[e]),
                    object_name=Path(env._object_urdf_paths[int(indices[e])]).name,
                    object_urdf_sha256=env._bps_manifest['entries'][int(indices[e])]['urdf_sha256'])
                (directory/'metadata.json').write_text(json.dumps(clip_meta,indent=2))
            (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
            print('BPS_REPOSING_CHECKPOINT_VALIDATED '+json.dumps(metadata),flush=True)
            stats=ReposeStats(env.num_envs,dt)
            frames=[]
            reference_frames=[]
            clip_reports=None
            selected_events={name:[] for name in selected}
            begin=time.monotonic()

            def snapshot(step):
                origin=env.scene.env_origins[ids]
                def pose(asset):
                    return array(torch.cat((asset.data.root_pos_w[ids]-origin,asset.data.root_quat_w[ids]),-1))
                # Recompute error for the CURRENT displayed goal. The task's
                # cached error may still refer to the just-completed goal.
                offset=env._keypoint_offsets_fixed[ids]
                a=_keypoints_world(env.object.data.root_pos_w[ids],env.object.data.root_quat_w[ids],offset)
                b=_keypoints_world(env.goal_viz.data.root_pos_w[ids],env.goal_viz.data.root_quat_w[ids],offset)
                return dict(step=np.full(len(ids),step),
                    joint_pos=array(env.robot.data.joint_pos[ids]),
                    joint_vel=array(env.robot.data.joint_vel[ids]),
                    commanded_joint_targets=array(env._cur_targets[ids]),
                    body_pos=array(env.robot.data.body_pos_w[ids]-origin[:,None,:]),
                    body_quat=array(env.robot.data.body_quat_w[ids]),robot=pose(env.robot),
                    object=pose(env.object),table=pose(env.table),goal=pose(env.goal_viz),
                    goal_error=array((a-b).norm(dim=-1).max(dim=-1).values),
                    goal_hits=stats.hits[list(selected.values())].copy(),
                    resets=stats.episodes[list(selected.values())].copy(),
                    lifted=array(env._lifted_object[ids]))

            with torch.inference_mode():
                for step in range(steps):
                    if not args.metrics_only and step<clip_steps:
                        if args.capture_reference or step%stride==0:
                            state=snapshot(step)
                            if args.capture_reference:
                                reference_frames.append(state)
                            if step%stride==0:
                                frames.append(state)
                    action=player.get_action(obs,is_deterministic=True)
                    if not torch.isfinite(action).all():
                        raise RuntimeError('Nonfinite policy action')
                    obs,reward,done,info=player.env_step(wrapped,action)
                    if not torch.isfinite(obs).all():
                        raise RuntimeError('Nonfinite observation')
                    terminal={k:array(v) for k,v in info['episode_final'].items()
                              if isinstance(v,torch.Tensor) and v.shape==(env.num_envs,)}
                    dones=array(done).reshape(-1).astype(bool)
                    hits=stats.update(terminal,dones,array(reward).reshape(-1))
                    ended=np.flatnonzero(dones)
                    if player.is_rnn and len(ended):
                        for state in player.states:
                            state[:,torch.as_tensor(ended,device=player.device),:]=0.
                    for family,e in selected.items():
                        if hits[e] or dones[e]:
                            selected_events[family].append(dict(step=step+1,time_s=(step+1)*dt,
                                goal_hit=int(hits[e]),reset=bool(dones[e]),
                                episode_final={k:float(v[e]) for k,v in terminal.items()}))
                    if step+1==clip_steps:
                        clip_reports={family:stats.report([e]) for family,e in selected.items()}
                    if (step+1)%600==0:
                        print('EVAL_PROGRESS '+json.dumps(dict(sim_s=(step+1)*dt,
                            wall_s=time.monotonic()-begin,hits=int(stats.hits.sum()),
                            episodes=int(stats.episodes.sum()),selected_hits={f:int(stats.hits[e]) for f,e in selected.items()})),flush=True)
            report=stats.report()
            report['per_family']={family:stats.report([e for e,asset in enumerate(indices)
                if family_name(env._object_urdf_paths[int(asset)])==family])
                for family in sorted({family_name(p) for p in env._object_urdf_paths})}
            report.update(checkpoint_sha256=metadata['checkpoint_sha256'],wall_seconds=time.monotonic()-begin,
                          zero_training_updates=True)
            (args.output/'evaluation.json').write_text(json.dumps(report,indent=2))
            np.savez_compressed(args.output/'per_object_counts.npz',asset_indices=indices,
                **{name:getattr(stats,name) for name in ('hits','failed_attempts','episodes','any_goal','lifted','completed_goals')})
            stacked={key:np.stack([f[key] for f in frames]) for key in frames[0]} if frames else {}
            references={key:np.stack([f[key] for f in reference_frames]) for key in reference_frames[0]} if reference_frames else {}
            for index,(family,e) in enumerate(selected.items()):
                directory=args.output/'clips'/family
                np.savez_compressed(directory/'trajectory.npz',**{key:value[:,index] for key,value in stacked.items()})
                if references:
                    np.savez_compressed(directory/'reference_trace.npz',**{key:value[:,index] for key,value in references.items()})
                m=json.loads((directory/'metadata.json').read_text())
                m.update(frames=len(frames),seconds=clip_steps*dt,clip_metrics=clip_reports[family],
                         full_horizon_metrics=stats.report([e]),completed=True)
                (directory/'metadata.json').write_text(json.dumps(m,indent=2))
                (directory/'events.json').write_text(json.dumps(selected_events[family],indent=2))
            print('BPS_REPOSING_EVALUATION_COMPLETE '+json.dumps(report),flush=True)
        finally:
            env.close()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        app.close()


if __name__=='__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
