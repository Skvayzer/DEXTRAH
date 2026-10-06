#!/usr/bin/env python3
"""Continue BPS+touch SAPG with a trainable or frozen-pretrained SONIC body.

Original Play2Perfect task + original six-group SAPG Runner. No new manipulation
reward, no physics freezing, no fabrics/PCA, no periodic evaluation orchestrator.
"""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a Slurm GPU allocation')
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap', type=Path)
    p.add_argument('--frozen-pretrained-sonic', action='store_true',
                   help='Original frozen encoder/FSQ/decoder; SAPG learns 64 latent + 6 finger meta-actions')
    p.add_argument('--adapter-learning-rate', type=float, default=3e-4)
    p.add_argument('--num-envs', type=int, default=768)
    p.add_argument('--max-epochs', type=int, default=120, help='Development smoke limit; --continuous removes it')
    p.add_argument('--continuous', action='store_true')
    p.add_argument('--resume', type=Path)
    p.add_argument('--memory-trace', action='store_true')
    p.add_argument('--boundary-gc-interval', type=int, default=0,
                   help='Full cyclic GC between completed updates; zero disables; never flushes CUDA cache')
    p.add_argument('--memory-gc-probe-epoch', type=int, default=0,
                   help='One diagnostic garbage collection after this absolute epoch; zero disables')
    p.add_argument('--wandb', choices=['online', 'disabled'], default='online')
    p.add_argument('--numerical-failure-mode', choices=['reset', 'abort'], default='reset',
                   help='Finite extreme joint speeds reset that environment; non-finite states always abort')
    p.add_argument('--diagnostic-capture-seconds', type=float, default=0.,
                   help='Opt-in physics-state ring buffer for reproducing a numerical failure')
    p.add_argument('--reset-best', action='store_true',
                   help='After --resume, forget the parent run best return so this run saves its own best checkpoint')
    p.add_argument('--no-table-body-support', action='store_true',
                   help='Table collides only with right forearm/hand/fingers, so the body cannot lean on it')
    p.add_argument('--carry', action='store_true',
                   help='Carry expert: restored grasps, no object goal, walking clips, carry reward')
    p.add_argument('--carry-clips', type=Path, help='Precomputed planner walking clips (.npz)')
    p.add_argument('--grasp-bank', type=Path, help='Grasp snapshots from build_grasp_bank.py')
    p.add_argument('--carry-episode-seconds', type=float, default=20.)
    p.add_argument('--init-weights', type=Path,
                   help='Load actor/critic weights only (fresh optimizers, epoch 0) from a same-architecture checkpoint')
    p.add_argument('--critic-warmup-epochs', type=int, default=0,
                   help='Skip actor optimizer steps for this many epochs while the critic adapts')
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    if args.num_envs < 6 or args.num_envs % 6 or not args.output.name.startswith('0_'):
        raise ValueError('Require six groups and a policy-index-prefixed output name')
    if args.output.exists():
        raise FileExistsError('Use a new output directory, including for a checkpointed continuation')
    if args.frozen_pretrained_sonic:
        if args.bootstrap is not None:
            raise ValueError('A fine-tuned decoder bootstrap is incompatible with frozen pretrained SONIC')
    elif args.bootstrap is None or not args.bootstrap.is_file():
        raise ValueError('Direct decoder training requires a valid --bootstrap')
    if args.carry and (not args.frozen_pretrained_sonic or not args.carry_clips or not args.grasp_bank
                       or not args.carry_clips.is_file() or not args.grasp_bank.is_file()):
        raise ValueError('Carry training needs --frozen-pretrained-sonic, --carry-clips and --grasp-bank')
    if args.init_weights is not None and (args.resume is not None or not args.init_weights.is_file()):
        raise ValueError('--init-weights needs an existing checkpoint and excludes --resume')
    args.output.mkdir(parents=True)
    args.headless = True
    # Kit monkey-patches asyncio.run globally. W&B must start its background
    # asyncio thread BEFORE that patch, or it can stop Kit's loop from another
    # thread while a local USD availability check is running (smoke 556).
    import wandb
    if args.wandb == 'online':
        wandb.init(project='adept', entity='skvayzer', group='g1-carry-expert' if args.carry else 'g1-sonic-bps128-touch',
            name=args.output.name, id='unique_id_'+args.output.name, resume='allow',
            dir=str(args.output), sync_tensorboard=True, mode='online',
            config=dict(source_commit=os.environ.get('FULLBODY_SOURCE_COMMIT'), num_envs=args.num_envs),
            tags=['sapg', 'sonic', 'g1', 'revo2', 'bps128', 'touch70', 'floating-body',
                  'frozen-pretrained-sonic' if args.frozen_pretrained_sonic else 'trainable-sonic-decoder',
                  'continuous' if args.continuous else 'training-smoke'] + (['carry-expert'] if args.carry else []))
        if wandb.run is None or wandb.run.settings.mode != 'online':
            raise RuntimeError('Requested online logging did not initialize')
        wandb.run.summary.update(dict(experiment_status='initializing_simulation', optimizer_updates_started=False))
        (args.output/'wandb.json').write_text(json.dumps(dict(id=wandb.run.id, url=wandb.run.url), indent=2))
        print('WANDB_INITIALIZING_SIMULATION '+wandb.run.url, flush=True)
    try:
        app = AppLauncher(args).app
    except BaseException as error:
        failure = dict(completed=False, experiment_status='failed', error=f'{type(error).__name__}: {error}')
        (args.output/'training_result.json').write_text(json.dumps(failure, indent=2))
        if wandb.run:
            wandb.run.summary.update(failure)
            wandb.finish(exit_code=1)
        raise
    env, algo, diagnostic = None, None, None
    stack_log = (args.output/'stacks.log').open('w')
    faulthandler.enable(file=stack_log)
    faulthandler.dump_traceback_later(60, repeat=True, file=stack_log)
    report = dict(completed=False, source_commit=os.environ.get('FULLBODY_SOURCE_COMMIT'))
    try:
        import torch
        import wandb
        from isaaclab.utils.io import dump_yaml
        from isaacsimenvs.utils.rlgames_utils import register_rlgames_env, EnvStatsAlgoObserver, MultiObserver
        from rl_games.torch_runner import Runner
        from dextrah_lab.g1_adept.touch_continuation import (load_yaml, install_complete_checkpoint,
            install_atomic_checkpoint_saves, resume_at_episode_boundary)
        from dextrah_lab.wholebody.sonic import FrozenSonic, WEIGHTS_SHA256
        from dextrah_lab.wholebody.frozen_sapg import (FROZEN_ARCHITECTURE,
            install_latent_action_bounds, install_latent_optimizer)
        from dextrah_lab.wholebody.student_checkpoint import load_student_checkpoint
        from dextrah_lab.wholebody.sapg_network import load_touch_sources, register_sonic_models
        from dextrah_lab.wholebody.source_config import source_task_config
        from dextrah_lab.wholebody.task_contract import TOUCH_RUN, TOUCH_SHA256, P2P_REVISION
        from dextrah_lab.wholebody.training_observer import SonicTransferObserver
        from dextrah_lab.tasks.g1_revo2_adept.g1_sonic_touch_env import G1SonicTouchEnv
        torch.set_num_threads(4)
        workspace = Path('/data1/users/konstantin.smirnov')
        revision = subprocess.check_output(['git', '-C', str(workspace/'play2perfect'), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != P2P_REVISION:
            raise ValueError('Pinned Play2Perfect revision changed')
        cfg, contract = source_task_config(args.output, args.num_envs, args.device)
        cfg.sonic_body.numerical_failure_mode = args.numerical_failure_mode
        cfg.sonic_body.table_supports_body = not args.no_table_body_support
        frozen = args.frozen_pretrained_sonic
        cfg.sonic_body.controller_mode = 'frozen_pretrained_latent' if frozen else 'trainable_decoder'
        agent = load_yaml(Path(TOUCH_RUN)/'params/agent.yaml')
        bootstrap_sha = None
        if args.bootstrap is not None:
            with args.bootstrap.open('rb') as f:
                bootstrap_sha = hashlib.file_digest(f, 'sha256').hexdigest()
        contract.update(bootstrap=str(args.bootstrap) if args.bootstrap else None, bootstrap_sha256=bootstrap_sha,
            action_units=('64 unbounded latent residuals scaled 0.1 before FSQ + 6 original absolute finger commands'
                          if frozen else '29 body joint-limit-normalized absolute targets + 6 original absolute finger commands'),
            original_task_observation_clip=10., sonic_body_observation_clip=None,
            fresh_optimizers_for_architecture_migration=not bool(args.resume),
            continuous=args.continuous, periodic_evaluation=False,
            source_training_transitions=17364025344, source_commit=report['source_commit'])
        contract.update(controller_mode=cfg.sonic_body.controller_mode,
            pretrained_sonic_sha256=WEIGHTS_SHA256, pretrained_sonic_frozen=frozen,
            action_dim=70 if frozen else 35)
        if frozen:
            contract.update(architecture=FROZEN_ARCHITECTURE, latent_residual_scale=.1,
                initial_latent_std_leader=1., adapter_learning_rate=args.adapter_learning_rate,
                body_joint_exploration_noise=False, latent_clipping=False,
                action_delay='source delay on latent/finger meta-actions; SONIC uses current body feedback',
                initialization='completed touch-383 SAPG features/fingers/critic; zero-initialized new latent output',
                old_arm_policy_exactly_preserved=False, pretrained_sonic_audit='all tensors bitwise after every update',
                online_teacher_loss=False, new_balance_reward=False)
            contract['intentional_changes'] = [x for x in contract['intentional_changes']
                if x != '29_body_plus_6_right_finger_actions'] + [
                '64_latent_plus_6_right_finger_meta_actions', 'frozen_pretrained_SONIC_in_environment',
                'exploration_and_delay_before_body_decoding', 'latent_adapter_optimizer_group']
        else:
            contract['new_body_exploration_std_rad'] = .025
        contract['body_termination'] = cfg.sonic_body.to_dict()
        if args.no_table_body_support:
            contract['intentional_changes'] = contract['intentional_changes'] + [
                'table_collides_only_with_right_forearm_hand_fingers']
        contract['nonfinite_state_handling'] = 'abort before rewards and terminal observations'
        contract['finite_numerical_failure_reward'] = 'unchanged source reward; separately logged true termination'
        contract['touch_global_partners'] = 'enabled static collision shapes under /World/ground'
        contract['memory_trace'] = args.memory_trace
        contract['memory_gc_probe_epoch'] = args.memory_gc_probe_epoch
        contract['boundary_gc_interval'] = args.boundary_gc_interval
        contract['cuda_allocator_config'] = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
        contract['periodic_stack_dumps_during_training'] = False
        contract['wandb_initialization'] = 'before_Isaac_Kit_asyncio_patch'
        import carb
        settings = carb.settings.get_settings()
        contract['runtime_worker_settings'] = {key: settings.get(key) for key in (
            '/plugins/carb.tasking.plugin/threadCount', '/plugins/omni.tbb.globalcontrol/maxThreadCount',
            '/app/asyncRendering', '/app/asyncRenderingLowLatency')}
        (args.output/'task_contract.json').write_text(json.dumps(contract, indent=2))
        if args.wandb == 'online':
            wandb.config.update(contract, allow_val_change=True)
        sonic = FrozenSonic(workspace/'GRAIL', workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', args.device)
        actor, critic = load_touch_sources(args.device)
        student, bootstrap_report = None, None
        if not frozen:
            student, bootstrap_report = load_student_checkpoint(args.bootstrap, sonic, args.device)
            bootstrap_config = load_yaml(args.bootstrap.parent/'config.json')
            if (bootstrap_report['source_sapg_sha256'] != TOUCH_SHA256 or student.task_dim != 249
                    or bootstrap_config.get('policy_hz') != 60):
                raise ValueError('Require the completed touch teacher distilled on the 60 Hz task clock')
        if args.carry:
            from dextrah_lab.wholebody.carry_env import G1CarryEnv
            cfg.episode_length_s = args.carry_episode_seconds
            cfg.termination.episode_length = round(args.carry_episode_seconds*60)
            cfg.termination.max_consecutive_successes = 0
            env = G1CarryEnv(cfg, sonic=sonic, clips_path=args.carry_clips, grasp_bank_path=args.grasp_bank)
            def sha(path):
                with Path(path).open('rb') as f:
                    return hashlib.file_digest(f, 'sha256').hexdigest()
            contract['carry'] = dict(env.carry_contract(), clips=str(args.carry_clips), clips_sha256=sha(args.carry_clips),
                grasp_bank=str(args.grasp_bank), grasp_bank_sha256=sha(args.grasp_bank),
                episode_seconds=args.carry_episode_seconds, critic_warmup_epochs=args.critic_warmup_epochs)
            contract['unchanged'] = [x for x in contract['unchanged']
                                     if x not in ('reward', 'goals', 'success_counters', 'task_terminations')]
            contract['intentional_changes'] = contract['intentional_changes'] + [
                'carry_reward_replaces_reposing_reward', 'no_object_goal_goal_error_zeroed',
                'grasp_bank_resets_with_table_parked_below_floor', 'walking_clip_references',
                'moving_body_frame_palm_object_observations', 'drop_termination', 'value_normalizers_reset']
            (args.output/'task_contract.json').write_text(json.dumps(contract, indent=2))
            if args.wandb == 'online':
                wandb.config.update(dict(carry=contract['carry'], intentional_changes=contract['intentional_changes']),
                                    allow_val_change=True)
        else:
            env = G1SonicTouchEnv(cfg, sonic=sonic)
        if args.diagnostic_capture_seconds > 0:
            from dextrah_lab.wholebody.failure_recording import FailureRecorder
            diagnostic = FailureRecorder(env, args.output/'failure_clip', args.diagnostic_capture_seconds,
                                         epoch=lambda: 0 if algo is None else int(algo.epoch_num))
        register_sonic_models(actor, critic, sonic, env._body_lower[0], env._body_upper[0], student, frozen=frozen)
        params = agent['params']
        params['model']['name'] = 'sonic_sapg_logstd'
        params['network']['name'] = 'sonic_sapg_actor'
        # Per-task clipping now occurs inside TaskBodyNormalizer, so body qd
        # and executed-action histories are not clipped to legacy +/-10.
        params['env']['clip_observations'] = float('inf')
        params['env']['clip_actions'] = float('inf') if frozen else 1.
        params['load_checkpoint'], params['load_path'] = False, ''
        ac = params['config']
        ac.update(name='g1_sonic_sapg_bps128_touch', num_actors=args.num_envs,
            expl_coef_block_size=args.num_envs//6, minibatch_size=4*args.num_envs,
            device=args.device, device_name=args.device, multi_gpu=False,
            max_frames=-1, max_epochs=-1 if args.continuous else args.max_epochs,
            train_dir=str(args.output.parent), full_experiment_name=args.output.name)
        ac.pop('score_to_win', None)
        if frozen:
            ac['clip_actions'] = False
            ac['name'] = 'g1_frozen_sonic_sapg_bps128_touch'
            ac['adapter_learning_rate'] = args.adapter_learning_rate
        ac['central_value_config']['minibatch_size'] = 4*args.num_envs
        ac['central_value_config']['model'] = dict(name='sonic_sapg_value')
        ac['central_value_config']['network']['name'] = 'sonic_sapg_critic'
        agent['hydra']['run']['dir'] = str(args.output)
        agent['wandb_activate'] = args.wandb == 'online'
        agent['wandb_group'] = 'g1-sonic-bps128-touch'
        dump_yaml(str(args.output/'params/env_resolved.yaml'), cfg)
        dump_yaml(str(args.output/'params/agent.yaml'), agent)
        wrapped = register_rlgames_env(env, rl_device=args.device, clip_obs=float('inf'),
                                      clip_actions=float('inf') if frozen else 1.)
        observer = SonicTransferObserver(env, args.output)
        observers = [EnvStatsAlgoObserver(), observer]
        if args.wandb == 'online':
            class ExistingWandbObserver:
                def before_init(self, base_name, config, experiment_name):
                    if wandb.run.id != 'unique_id_'+experiment_name:
                        raise RuntimeError('W&B identity changed while constructing SAPG')
                    wandb.config.update(config, allow_val_change=True)
            observers.append(ExistingWandbObserver())
        runner = Runner(MultiObserver(observers))
        runner.load(agent)
        algo = runner.algo_factory.create(runner.algo_name, base_name='run', params=runner.params)
        if frozen:
            if algo.bound_loss_type != 'bound':
                raise ValueError('Expected the original finger bounds loss')
            install_latent_action_bounds(algo)
            install_latent_optimizer(algo, args.adapter_learning_rate)
            report['frozen_controller_validation'] = observer.controller_audit.check(algo.optimizer)
        install_complete_checkpoint(algo)
        install_atomic_checkpoint_saves(algo)
        if args.resume:
            previous = json.loads((args.resume.parent.parent/'task_contract.json').read_text())
            for key in ('source_sha256', 'bank_sha256', 'bootstrap_sha256', 'action_units'):
                if previous[key] != contract[key]:
                    raise ValueError(f'Incompatible continuation contract: {key}')
            if frozen:
                for key in ('architecture', 'pretrained_sonic_sha256', 'controller_mode', 'adapter_learning_rate'):
                    if previous.get(key) != contract[key]:
                        raise ValueError(f'Incompatible frozen-controller continuation: {key}')
            report['resume'] = resume_at_episode_boundary(algo, args.resume)
            if args.reset_best:
                # The parent's best return belongs to a different MDP; keeping
                # it would block every best-checkpoint save in this run.
                report['resume']['parent_best_return_discarded'] = float(algo.last_mean_rewards)
                algo.last_mean_rewards = -1000000000
        if args.init_weights is not None:
            from dextrah_lab.wholebody.carry_training import load_initial_weights, install_critic_warmup
            report['init_weights'] = load_initial_weights(algo, args.init_weights, reset_values=args.carry)
            if args.critic_warmup_epochs:
                install_critic_warmup(algo, args.critic_warmup_epochs)
        if args.memory_trace:
            from dextrah_lab.wholebody.memory_diagnostics import install_memory_trace
            install_memory_trace(algo, args.output, args.memory_gc_probe_epoch)
        if args.boundary_gc_interval:
            from dextrah_lab.wholebody.memory_cleanup import install_boundary_gc
            install_boundary_gc(algo, args.output, args.boundary_gc_interval)
        def request_stop(signum, frame):
            observer.stop_requested = signal.Signals(signum).name
            print('CHECKPOINTED_STOP_REQUESTED '+observer.stop_requested, flush=True)
        signal.signal(signal.SIGUSR1, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        report.update(bootstrap=bootstrap_report, task_contract=contract, num_envs=args.num_envs)
        (args.output/'warmstart_validation.json').write_text(json.dumps(report, indent=2))
        start_frame, start_time = int(algo.frame), time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        print('SONIC_SAPG_TRAINING_START '+json.dumps(dict(envs=args.num_envs, groups=6,
            actions=70 if frozen else 35, task_reward='unchanged', touch_hz=70,
            body_decoder_trainable=not frozen, pretrained_sonic_sha256=WEIGHTS_SHA256)), flush=True)
        # 528 died midway through its 26th watchdog traceback ("File ???").
        # Background frame walking is a suspected trigger, not a proven cause.
        # Keep startup diagnostics and fatal-signal reporting, but do not walk
        # all live Python stacks asynchronously during the hot training loop.
        # This changes diagnostics only, not task/optimizer/checkpoint behavior.
        faulthandler.cancel_dump_traceback_later()
        print('STARTUP_STACK_WATCHDOG_DISABLED_FOR_TRAINING', flush=True)
        algo.train()
        if diagnostic is not None:
            report['diagnostic_capture'] = diagnostic.save()
        torch.cuda.synchronize()
        checkpoint = args.output/'nn'/f'complete_{algo.frame}'
        algo.save(str(checkpoint))
        wall = time.monotonic()-start_time
        report.update(completed=True, fullbody_rl_frames=int(algo.frame), fullbody_rl_epochs=int(algo.epoch_num),
            checkpoint=str(checkpoint)+'.pth', wall_seconds=wall,
            transitions_per_second=(algo.frame-start_frame)/wall,
            device_free_gib=torch.cuda.mem_get_info()[0]/2**30,
            torch_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            status='checkpointed_stop' if observer.stop_requested else 'completed',
            manipulation_success_validated=False)
        print('SONIC_SAPG_COMPLETE '+json.dumps(report), flush=True)
        algo.writer.flush()
        algo.writer.close()
        if wandb.run:
            wandb.run.summary.update(report)
            wandb.finish(exit_code=0)
    except BaseException as error:
        report.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        if diagnostic is not None:
            try:
                report['diagnostic_capture'] = diagnostic.save(error)
            except Exception as capture_error:
                report['capture_error'] = f'{type(capture_error).__name__}: {capture_error}'
        import wandb
        if wandb.run:
            wandb.run.summary.update(dict(experiment_status='failed', error=report['error']))
            wandb.finish(exit_code=1)
        raise
    finally:
        (args.output/'training_result.json').write_text(json.dumps(report, indent=2))
        faulthandler.cancel_dump_traceback_later()
        # Kit shutdown can exit the process with status zero BEFORE returning
        # from close(). On failure, preserve the report/W&B result and let the
        # OS release CUDA resources without entering that shutdown path.
        if not report['completed']:
            import sys
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(1)
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
