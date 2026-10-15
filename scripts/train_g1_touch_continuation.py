#!/usr/bin/env python3
"""Checked SAPG continuation: BPS-128 + capacitive-compatible fingertip forces.

--backend mjlab runs the same fixed-body task on MuJoCo Warp (dextrah_lab/mj,
fixed_body=True), configured exactly as the Isaac fixed-body run RUN_FIXED
resolved it (per-env object physics there; per-world object geometry and
inertials in mjlab).
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Training must run inside a Slurm GPU allocation')
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--backend', choices=['isaac', 'mjlab'], default='isaac')
    backend = pre.parse_known_args()[0].backend
    if backend == 'isaac':
        from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['isaac', 'mjlab'], default='isaac')
    p.add_argument('--mj-nconmax', type=int, help='mjlab: contacts allocated per world (pooled)')
    p.add_argument('--mj-njmax', type=int, help='mjlab: constraint rows per world')
    p.add_argument('--mj-nan-capture', action='store_true', help='mjlab: record the first non-finite world')
    p.add_argument('--mj-arm-armature', action='store_true',
                   help='mjlab: G1 motor rotor armature on the 7 arm joints (Isaac profile has 0; MuJoCo arm blows up '
                        'without it, see dextrah_lab/mj/robot.py ContactParams.g1_arm_armature)')
    p.add_argument('--mj-nan-capture-max-qvel', type=float,
                   help='mjlab diagnostic: also capture (and abort) when any |qvel| exceeds this')
    p.add_argument('--mj-cone-hessian-fix', action='store_true',
                   help='mjlab: require the vendored mujoco_warp with the elliptic cone Hessian fix')
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--source-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=24576)
    p.add_argument('--max-frames', type=int, default=100000000)
    p.add_argument('--max-epochs', type=int, default=-1)
    p.add_argument('--calibration', type=Path)
    p.add_argument('--calibration-steps', type=int, default=1200)
    p.add_argument('--resume', type=Path)
    p.add_argument('--control', action='store_true')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--continuous', action='store_true', help='No evaluation orchestration or transition cap')
    p.add_argument('--touch-sensor-hz', type=float)
    p.add_argument('--touch-publish-hz', type=float)
    p.add_argument('--per-env-object-physics', action='store_true',
                   help='Each env simulates its own assigned object (fixes the replicated-template bug: all envs '
                        'simulated env 0\'s object); verified against URDF masses at startup')
    p.add_argument('--stop-after-seconds', type=float, default=0,
                   help='Continuous mode: checkpointed stop after this many training seconds (probes)')
    p.add_argument('--snapshot-epochs', type=int, default=1024,
                   help='Continuous mode: immutable snapshot every N epochs (0 = none; latest.pth and best stay)')
    p.add_argument('--minibatch-size', type=int,
                   help='PPO minibatch for actor and critic (default 4*num_envs: 4 minibatches per epoch). '
                        'More minibatches = more gradient steps per epoch on the same rollout')
    p.add_argument('--best-delay-epochs', type=int, default=0,
                   help='New continuous run: start best-checkpoint tracking this many epochs after the resume '
                        '(0 = from the first epoch)')
    if backend == 'isaac':
        AppLauncher.add_app_launcher_args(p)
    else:
        p.add_argument('--device', default='cuda:0')
        p.add_argument('--headless', action='store_true')
    args = p.parse_args()
    if args.num_envs % 6 or args.num_envs < 6 or not args.output.name.startswith('0_'):
        raise ValueError('Require six SAPG groups and a numeric policy-index run name')
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    if args.continuous:
        if not args.resume or args.control:
            raise ValueError('Continuous mode requires a tactile continuation checkpoint')
        args.max_frames = args.max_epochs = -1
    new_run = not args.output.exists()
    args.headless = True
    app = AppLauncher(args).app if backend == 'isaac' else None
    env = None
    try:
        import torch
        import wandb
        if backend == 'mjlab':
            from dextrah_lab.mj.compat import install
            install()
            import yaml
            import mujoco_warp
            from dextrah_lab.mj.rlgames import register_mj_rlgames_env as register_rlgames_env
            from dextrah_lab.mj.env import G1SonicTouchMjEnv, Cfg, load_run_cfg, RUN_FIXED
            from dextrah_lab.mj.factory import DEFAULT_MUJOCO, DEFAULT_NCONMAX, DEFAULT_NJMAX
            def dump_yaml(path, data):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(yaml.dump(data.to_dict() if hasattr(data, 'to_dict') else data))
            vendored = 'mujoco_warp_cone_fix' in Path(mujoco_warp.__file__).parts
            if args.mj_cone_hessian_fix != vendored:
                raise RuntimeError(f'--mj-cone-hessian-fix={args.mj_cone_hessian_fix} but mujoco_warp is {mujoco_warp.__file__}')
        else:
            from isaaclab.utils import replace_strings_with_slices
            from isaaclab.utils.io import dump_yaml
            from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
        from isaacsimenvs.utils.rlgames_utils import EnvStatsAlgoObserver, MultiObserver
        from rl_games.torch_runner import Runner
        from dextrah_lab.g1_adept.touch_policy import register_models
        from dextrah_lab.g1_adept.touch_continuation import (load_yaml, verify_source_mdp,
            TouchContinuationObserver, install_complete_checkpoint, resume_at_episode_boundary,
            install_atomic_checkpoint_saves, inherit_resume_artifacts,
            SOURCE_FRAMES, SOURCE_SHA, BANK_SHA)
        if backend == 'isaac':
            from dextrah_lab.tasks.g1_revo2_adept.g1_revo2_bps_env import G1Revo2BpsEnv, G1Revo2BpsEnvCfg
            from dextrah_lab.tasks.g1_revo2_adept.g1_revo2_touch_env import G1Revo2TouchEnv, G1Revo2TouchEnvCfg
        elif args.control:
            raise ValueError('The mjlab backend implements the tactile task only')
        torch.set_num_threads(4)
        register_models()
        saved = load_yaml(args.source_run/'params/env_resolved.yaml')
        agent = load_yaml(args.source_run/'params/agent.yaml')
        if backend == 'mjlab':
            # The Isaac fixed-body run's resolved task (per-env object physics).
            cfg = load_run_cfg(RUN_FIXED)
            mj_settings = dict(task_config=str(RUN_FIXED/'params/env_resolved.yaml'),
                               nconmax=args.mj_nconmax or DEFAULT_NCONMAX, njmax=args.mj_njmax or DEFAULT_NJMAX,
                               mujoco_options=DEFAULT_MUJOCO, mujoco_warp=str(Path(mujoco_warp.__file__).parent),
                               cone_hessian_fix=args.mj_cone_hessian_fix, nan_capture=args.mj_nan_capture,
                               g1_arm_armature=args.mj_arm_armature)
        else:
            cfg = G1Revo2BpsEnvCfg() if args.control else G1Revo2TouchEnvCfg()
            cfg.seed = saved['seed']
            cfg.from_dict(replace_strings_with_slices(copy.deepcopy(saved)))
        provenance = None
        if args.resume:
            provenance = inherit_resume_artifacts(args.output,args.resume,args.control)
            parent_cfg = load_yaml(Path(provenance['parent_run'])/'params/env_resolved.yaml')
            if not args.control:
                if backend == 'isaac':
                    cfg.touch.from_dict(parent_cfg['touch'])
                elif parent_cfg['touch'] != cfg.touch.to_dict():
                    raise ValueError('RUN_FIXED touch settings differ from the resumed run')
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        for key in ('success_tolerance','target_success_tolerance','eval_success_tolerance'):
            setattr(cfg.termination,key,.01)
        cfg.bps_artifact_dir = str(args.output/'bps')
        cfg.bps_cache_dir = '/data1/users/konstantin.smirnov/DEXTRAH-BPS128/outputs/bps128_cache'
        if not args.control:
            cfg.touch.material_override = False
            cfg.touch.arm_torques = False
            if args.touch_sensor_hz is not None:
                cfg.touch.sensor_hz = args.touch_sensor_hz
            if args.touch_publish_hz is not None:
                cfg.touch.publish_hz = args.touch_publish_hz
            if backend == 'isaac':
                cfg.touch.observation_config().validate(cfg.sim.dt)
        if args.num_envs <= 1200 and backend == 'isaac':
            for key,value in dict(gpu_found_lost_pairs_capacity=2**20,
                gpu_found_lost_aggregate_pairs_capacity=2**21,gpu_total_aggregate_pairs_capacity=2**20,
                gpu_max_rigid_contact_count=2**20,gpu_max_rigid_patch_count=2**18,
                gpu_collision_stack_size=2**26).items():
                setattr(cfg.sim.physx,key,value)
        assert not cfg.fabric.enabled and not cfg.fabric.pca_enabled
        if backend == 'mjlab':
            # RUN_FIXED differs from the source only by the per-env replication flags.
            data = cfg.to_dict()
            for key, value in saved['assets'].items():
                if key.startswith('replicate_'):
                    data['assets'][key] = value
            verify_source_mdp(saved, Cfg(data))
            if not args.per_env_object_physics:
                raise ValueError('mjlab always simulates per-env objects; pass --per-env-object-physics')
        else:
            verify_source_mdp(saved,cfg)
        if args.per_env_object_physics and backend == 'isaac':
            # Explicit scene change after the MDP check: per-env distinct object physics.
            from dextrah_lab.wholebody import per_env_objects
            per_env_objects.configure(cfg)
        args.output.mkdir(parents=True,exist_ok=bool(args.resume))
        if provenance:
            provenance.update(continuous=args.continuous, fresh_physics_at_initial_resume=True,
                per_env_object_physics=args.per_env_object_physics, snapshot_epochs=args.snapshot_epochs,
                sensor_hz=None if args.control else cfg.touch.sensor_hz,
                publish_hz=None if args.control else cfg.touch.publish_hz,
                parent_sensor_hz=parent_cfg.get('touch',{}).get('sensor_hz'),
                parent_publish_hz=parent_cfg.get('touch',{}).get('publish_hz'), backend=backend,
                simulator=mj_settings if backend == 'mjlab' else 'isaac')
            (args.output/'resume_provenance.json').write_text(json.dumps(provenance,indent=2))
        if args.wandb:
            # Make startup visible without implying that optimizer updates have
            # begun. TensorBoard sync still starts before the summary writer.
            wandb.init(project='adept',entity='skvayzer',
                group='g1-bps128-tactile-continuation',name=args.output.name,
                id='unique_id_'+args.output.name,resume='allow',dir=str(args.output),
                sync_tensorboard=True,tags=['sapg','g1','revo2','bps128',
                    'control' if args.control else 'tactile-normal-shear','warmstart-8B',
                    'strict-tolerance','no-fabrics','no-pca','no-arm-torques']
                    +(['per-env-object-physics'] if args.per_env_object_physics else [])+['backend-'+backend])
            if wandb.run is None or wandb.run.settings.mode != 'online':
                raise RuntimeError('Requested online W&B logging is not active')
            wandb.run.summary.update(dict(experiment_status='initializing_simulation',
                optimizer_updates_started=False,num_envs=args.num_envs))
            (args.output/'wandb.json').write_text(json.dumps(dict(url=wandb.run.url,id=wandb.run.id),indent=2))
            print('WANDB_INITIALIZING_SIMULATION '+wandb.run.url,flush=True)
        if backend == 'mjlab':
            from dextrah_lab.mj.robot import ContactParams
            from dextrah_lab.mj.scene import SceneParams
            env = G1SonicTouchMjEnv(cfg, sonic=None, num_envs=args.num_envs, device=args.device,
                                    nconmax=mj_settings['nconmax'], njmax=mj_settings['njmax'],
                                    mujoco_cfg=mj_settings['mujoco_options'], fixed_body=True,
                                    scene_params=SceneParams(contact=ContactParams(g1_arm_armature=args.mj_arm_armature)))
            if args.mj_nan_capture:
                env.enable_nan_capture(args.output/'nan_capture', max_abs_qvel=args.mj_nan_capture_max_qvel)
            masses = env.sim.model.body_mass[:, env.object.entity.indexing.root_body_id]
            object_physics = dict(backend='mjlab per-world object geometry and inertials', envs=env.num_envs,
                                  unique_masses=int(torch.unique(masses).numel()))
        else:
            env_class = G1Revo2BpsEnv if args.control else G1Revo2TouchEnv
            if args.per_env_object_physics:
                env_class = per_env_objects.with_per_env_objects(env_class)
            env = env_class(cfg)
            object_physics = per_env_objects.verify(env) if args.per_env_object_physics else None
        if object_physics:
            (args.output/'per_env_object_physics.json').write_text(json.dumps(object_physics,indent=2))
        ac = agent['params']['config']
        minibatch_size = args.minibatch_size or 4*args.num_envs
        if (ac['horizon_length']*args.num_envs) % minibatch_size:
            raise ValueError(f'--minibatch-size {minibatch_size} must divide the rollout '
                             f'{ac["horizon_length"]}x{args.num_envs}')
        ac.update(name='g1_sapg_bps128_control' if args.control else 'g1_sapg_bps128_touch',
            num_actors=args.num_envs,expl_coef_block_size=args.num_envs//6,
            device=args.device,device_name=args.device,multi_gpu=False,
            minibatch_size=minibatch_size,max_frames=args.max_frames,max_epochs=args.max_epochs,
            learning_rate=1e-5,lr_schedule='constant',save_frequency=0,
            train_dir=str(args.output.parent),full_experiment_name=args.output.name)
        ac['central_value_config'].update(minibatch_size=minibatch_size,learning_rate=1e-4)
        optimization = dict(minibatch_size=minibatch_size,
            minibatches_per_epoch=ac['horizon_length']*args.num_envs//minibatch_size,
            actor_gradient_steps_per_epoch=ac['mini_epochs']*ac['horizon_length']*args.num_envs//minibatch_size,
            critic_gradient_steps_per_epoch=ac['central_value_config']['mini_epochs']*ac['horizon_length']*args.num_envs//minibatch_size)
        print('OPTIMIZATION '+json.dumps(optimization),flush=True)
        if args.continuous:
            ac.pop('score_to_win',None)
        if not args.control:
            agent['params']['model']['name'] = 'continuous_a2c_logstd_touch'
            ac['central_value_config']['model'] = dict(name='central_value_touch')
        agent.update(wandb_activate=args.wandb,wandb_entity='skvayzer',wandb_project='adept',
            wandb_group='g1-bps128-tactile-continuation',wandb_name='',
            wandb_tags=['sapg','g1','revo2','bps128','control' if args.control else 'tactile-normal-shear',
                        'warmstart-8B','strict-tolerance','no-fabrics','no-pca','no-arm-torques'])
        agent['hydra']['run']['dir'] = str(args.output)
        dump_yaml(str(args.output/'params/env_resolved.yaml'),cfg)
        dump_yaml(str(args.output/'params/agent.yaml'),agent)
        wrapped = register_rlgames_env(env,rl_device=args.device,
            clip_obs=agent['params']['env']['clip_observations'],clip_actions=agent['params']['env']['clip_actions'])
        observer = TouchContinuationObserver(env,args.source_run,args.source_checkpoint,args.output,
            args.calibration,args.calibration_steps,bool(args.resume),args.control,args.continuous)
        observer.snapshot_epochs = args.snapshot_epochs
        observers = [EnvStatsAlgoObserver(),observer]
        if args.wandb:
            class ExistingWandbObserver:
                def before_init(self,base_name,config,experiment_name):
                    if wandb.run.id != 'unique_id_'+experiment_name:
                        raise RuntimeError('W&B experiment identity changed during initialization')
                    wandb.config.update(config,allow_val_change=True)
            observers.append(ExistingWandbObserver())
        runner = Runner(MultiObserver(observers))
        runner.load(agent)
        algo = runner.algo_factory.create(runner.algo_name,base_name='run',params=runner.params)
        install_complete_checkpoint(algo)
        install_atomic_checkpoint_saves(algo)
        resume_info = resume_at_episode_boundary(algo,args.resume) if args.resume else None
        observer.run_start_frame = int(algo.frame)
        if args.continuous and new_run:
            # Fresh physics must not mix old rolling scores with newly reset
            # observer buffers. Preserve all learned statistics and optimizers.
            for meter in (algo.game_rewards,algo.game_shaped_rewards,algo.game_lengths):
                meter.clear()
            algo.last_mean_rewards = -1e9
            if args.best_delay_epochs:
                # Fresh episodes finish short (failures) first; skip that transient.
                from dextrah_lab.wholebody.carry_training import install_delayed_best
                optimization['best_tracking'] = install_delayed_best(algo, args.best_delay_epochs)
        if args.continuous:
            def request_checkpointed_stop(signum, frame):
                observer.stop_requested = signal.Signals(signum).name
                print('CHECKPOINTED_STOP_REQUESTED '+observer.stop_requested,flush=True)
            signal.signal(signal.SIGUSR1,request_checkpointed_stop)
            signal.signal(signal.SIGTERM,request_checkpointed_stop)
        if provenance:
            provenance['optimization'] = optimization
            (args.output/'resume_provenance.json').write_text(json.dumps(provenance,indent=2))
        if resume_info:
            (args.output/'resume_validation.json').write_text(json.dumps(resume_info,indent=2))
            print('CONTINUATION_RESUMED '+json.dumps(resume_info),flush=True)
        if args.wandb:
            if wandb.run is None or wandb.run.settings.mode != 'online':
                raise RuntimeError('Requested online W&B logging is not active')
            wandb.config.update(dict(source_frames=SOURCE_FRAMES,source_sha256=SOURCE_SHA,
                parent_run='unique_id_0_g1_bps128_warmstart_seed_42_355',bank_sha256=BANK_SHA,
                actor_dim=cfg.observation_space,critic_dim=cfg.state_space,
                additional_frame_budget=None if args.continuous else args.max_frames,
                continuous=args.continuous,periodic_evaluation=False,
                resume_checkpoint=str(args.resume) if args.resume else None,
                run_start_frame=int(algo.frame),
                tactile_sensor_hz=None if args.control else cfg.touch.sensor_hz,
                tactile_publish_hz=None if args.control else cfg.touch.publish_hz,
                best_checkpoint_metric='SAPG leader rolling mean training episode reward; not evaluation success',
                per_env_object_physics=object_physics,snapshot_epochs=args.snapshot_epochs,
                material_override=False,arm_torques=False,optimization=optimization),allow_val_change=True)
            wandb.run.summary.update(dict(warmstart_verified=True,experiment_status='training'))
            (args.output/'wandb.json').write_text(json.dumps(dict(url=wandb.run.url,id=wandb.run.id),indent=2))
        start_frame,start_time = int(algo.frame),time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        if args.stop_after_seconds and args.continuous:
            import threading
            def timed_stop():
                observer.stop_requested = 'TIME_LIMIT'
                print('CHECKPOINTED_STOP_REQUESTED TIME_LIMIT',flush=True)
            timer = threading.Timer(args.stop_after_seconds,timed_stop)
            timer.daemon = True
            timer.start()
        algo.train()
        torch.cuda.synchronize()
        checkpoint = args.output/'nn'/f'complete_{algo.frame}.pth'
        algo.save(str(checkpoint.with_suffix('')))
        summary = dict(checkpoint=str(checkpoint),frame=int(algo.frame),epoch=int(algo.epoch_num),
            segment_start_frame=start_frame,segment_wall_seconds=time.monotonic()-start_time,
            transitions_per_second=(algo.frame-start_frame)/(time.monotonic()-start_time),
            num_envs=env.num_envs,resume=resume_info,control=args.control,backend=backend,
            cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            device_free_gib=torch.cuda.mem_get_info()[0]/2**30)
        (args.output/'segment_complete.json').write_text(json.dumps(summary,indent=2))
        print('CONTINUATION_SEGMENT_COMPLETE '+json.dumps(summary),flush=True)
        algo.writer.flush()
        algo.writer.close()
        if args.wandb:
            status = ('checkpointed_stop' if observer.stop_requested else 'training_complete') if args.continuous else 'awaiting_evaluation'
            wandb.run.summary.update(dict(summary,experiment_status=status))
            wandb.finish(exit_code=0)
        if hasattr(env, 'close'):
            env.close()
        env = None
    except Exception:
        traceback.print_exc()
        try:
            import wandb
            if wandb.run:
                wandb.run.summary['experiment_status'] = 'failed'
                wandb.finish(exit_code=1)
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(1)
    finally:
        if env is not None and hasattr(env, 'close'):
            env.close()
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()
