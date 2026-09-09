#!/usr/bin/env python3
"""Checked SAPG continuation: BPS-128 + capacitive-compatible fingertip forces."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Training must run inside a Slurm GPU allocation')
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
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
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    if args.num_envs % 6 or args.num_envs < 6 or not args.output.name.startswith('0_'):
        raise ValueError('Require six SAPG groups and a numeric policy-index run name')
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.headless = True
    app = AppLauncher(args).app
    env = None
    try:
        import torch
        import wandb
        from isaaclab.utils import replace_strings_with_slices
        from isaaclab.utils.io import dump_yaml
        from isaacsimenvs.utils.rlgames_utils import register_rlgames_env, EnvStatsAlgoObserver, MultiObserver
        from rl_games.torch_runner import Runner
        from dextrah_lab.g1_adept.touch_policy import register_models
        from dextrah_lab.g1_adept.touch_continuation import (load_yaml, verify_source_mdp,
            TouchContinuationObserver, install_complete_checkpoint, resume_at_episode_boundary,
            SOURCE_FRAMES, SOURCE_SHA, BANK_SHA)
        from dextrah_lab.tasks.g1_revo2_adept.g1_revo2_bps_env import G1Revo2BpsEnv, G1Revo2BpsEnvCfg
        from dextrah_lab.tasks.g1_revo2_adept.g1_revo2_touch_env import G1Revo2TouchEnv, G1Revo2TouchEnvCfg
        from dextrah_lab.rl_games.wandb_utils import WandbAlgoObserver
        torch.set_num_threads(4)
        register_models()
        saved = load_yaml(args.source_run/'params/env_resolved.yaml')
        agent = load_yaml(args.source_run/'params/agent.yaml')
        cfg = G1Revo2BpsEnvCfg() if args.control else G1Revo2TouchEnvCfg()
        cfg.seed = saved['seed']
        cfg.from_dict(replace_strings_with_slices(copy.deepcopy(saved)))
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        for key in ('success_tolerance','target_success_tolerance','eval_success_tolerance'):
            setattr(cfg.termination,key,.01)
        cfg.bps_artifact_dir = str(args.output/'bps')
        cfg.bps_cache_dir = '/data1/users/konstantin.smirnov/DEXTRAH-BPS128/outputs/bps128_cache'
        if not args.control:
            cfg.touch.material_override = False
            cfg.touch.arm_torques = False
        if args.num_envs <= 1200:
            for key,value in dict(gpu_found_lost_pairs_capacity=2**20,
                gpu_found_lost_aggregate_pairs_capacity=2**21,gpu_total_aggregate_pairs_capacity=2**20,
                gpu_max_rigid_contact_count=2**20,gpu_max_rigid_patch_count=2**18,
                gpu_collision_stack_size=2**26).items():
                setattr(cfg.sim.physx,key,value)
        assert not cfg.fabric.enabled and not cfg.fabric.pca_enabled
        verify_source_mdp(saved,cfg)
        args.output.mkdir(parents=True,exist_ok=bool(args.resume))
        env = (G1Revo2BpsEnv if args.control else G1Revo2TouchEnv)(cfg)
        ac = agent['params']['config']
        ac.update(name='g1_sapg_bps128_control' if args.control else 'g1_sapg_bps128_touch',
            num_actors=args.num_envs,expl_coef_block_size=args.num_envs//6,
            device=args.device,device_name=args.device,multi_gpu=False,
            minibatch_size=4*args.num_envs,max_frames=args.max_frames,max_epochs=args.max_epochs,
            learning_rate=1e-5,lr_schedule='constant',save_frequency=0,
            train_dir=str(args.output.parent),full_experiment_name=args.output.name)
        ac['central_value_config'].update(minibatch_size=4*args.num_envs,learning_rate=1e-4)
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
            args.calibration,args.calibration_steps,bool(args.resume),args.control)
        observers = [EnvStatsAlgoObserver(),observer]
        if args.wandb:
            observers.append(WandbAlgoObserver(agent))
        runner = Runner(MultiObserver(observers))
        runner.load(agent)
        algo = runner.algo_factory.create(runner.algo_name,base_name='run',params=runner.params)
        install_complete_checkpoint(algo)
        resume_info = resume_at_episode_boundary(algo,args.resume) if args.resume else None
        if args.wandb:
            if wandb.run is None or wandb.run.settings.mode != 'online':
                raise RuntimeError('Requested online W&B logging is not active')
            wandb.config.update(dict(source_frames=SOURCE_FRAMES,source_sha256=SOURCE_SHA,
                parent_run='unique_id_0_g1_bps128_warmstart_seed_42_355',bank_sha256=BANK_SHA,
                actor_dim=cfg.observation_space,critic_dim=cfg.state_space,additional_frame_budget=500000000,
                material_override=False,arm_torques=False),allow_val_change=True)
            wandb.run.summary.update(dict(warmstart_verified=True,experiment_status='training'))
            (args.output/'wandb.json').write_text(json.dumps(dict(url=wandb.run.url,id=wandb.run.id),indent=2))
        start_frame,start_time = int(algo.frame),time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        algo.train()
        torch.cuda.synchronize()
        checkpoint = args.output/'nn'/f'complete_{algo.frame}.pth'
        algo.save(str(checkpoint.with_suffix('')))
        summary = dict(checkpoint=str(checkpoint),frame=int(algo.frame),epoch=int(algo.epoch_num),
            segment_start_frame=start_frame,segment_wall_seconds=time.monotonic()-start_time,
            transitions_per_second=(algo.frame-start_frame)/(time.monotonic()-start_time),
            num_envs=env.num_envs,resume=resume_info,control=args.control,
            cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            device_free_gib=torch.cuda.mem_get_info()[0]/2**30)
        (args.output/'segment_complete.json').write_text(json.dumps(summary,indent=2))
        print('CONTINUATION_SEGMENT_COMPLETE '+json.dumps(summary),flush=True)
        algo.writer.flush()
        algo.writer.close()
        if args.wandb:
            wandb.run.summary.update(dict(summary,experiment_status='awaiting_evaluation'))
            wandb.finish(exit_code=0)
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
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
