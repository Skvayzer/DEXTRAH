#!/usr/bin/env python3
"""Continue the trained BPS+touch manipulation skill in a trainable SONIC body.

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
    p.add_argument('--bootstrap', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=768)
    p.add_argument('--max-epochs', type=int, default=120, help='Development smoke limit; --continuous removes it')
    p.add_argument('--continuous', action='store_true')
    p.add_argument('--resume', type=Path)
    p.add_argument('--memory-trace', action='store_true')
    p.add_argument('--memory-gc-probe-epoch', type=int, default=0,
                   help='One diagnostic garbage collection after this absolute epoch; zero disables')
    p.add_argument('--wandb', choices=['online', 'disabled'], default='online')
    p.add_argument('--numerical-failure-mode', choices=['reset', 'abort'], default='reset',
                   help='Finite extreme joint speeds reset that environment; non-finite states always abort')
    p.add_argument('--diagnostic-capture-seconds', type=float, default=0.,
                   help='Opt-in physics-state ring buffer for reproducing a numerical failure')
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    if args.num_envs < 6 or args.num_envs % 6 or not args.output.name.startswith('0_'):
        raise ValueError('Require six groups and a policy-index-prefixed output name')
    if args.output.exists():
        raise FileExistsError('Use a new output directory, including for a checkpointed continuation')
    if not args.bootstrap.is_file():
        raise FileNotFoundError(args.bootstrap)
    args.output.mkdir(parents=True)
    args.headless = True
    app = AppLauncher(args).app
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
        from dextrah_lab.wholebody.sonic import FrozenSonic
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
        agent = load_yaml(Path(TOUCH_RUN)/'params/agent.yaml')
        with args.bootstrap.open('rb') as f:
            bootstrap_sha = hashlib.file_digest(f, 'sha256').hexdigest()
        contract.update(bootstrap=str(args.bootstrap), bootstrap_sha256=bootstrap_sha,
            action_units='29 body joint-limit-normalized absolute targets + 6 original absolute finger commands',
            original_task_observation_clip=10., sonic_body_observation_clip=None,
            new_body_exploration_std_rad=.025, fresh_optimizers_for_architecture_migration=not bool(args.resume),
            continuous=args.continuous, periodic_evaluation=False,
            source_training_transitions=17364025344, source_commit=report['source_commit'])
        contract['body_termination'] = cfg.sonic_body.to_dict()
        contract['nonfinite_state_handling'] = 'abort before rewards and terminal observations'
        contract['finite_numerical_failure_reward'] = 'unchanged source reward; separately logged true termination'
        contract['touch_global_partners'] = 'enabled static collision shapes under /World/ground'
        contract['memory_trace'] = args.memory_trace
        contract['memory_gc_probe_epoch'] = args.memory_gc_probe_epoch
        contract['cuda_allocator_config'] = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
        (args.output/'task_contract.json').write_text(json.dumps(contract, indent=2))
        if args.wandb == 'online':
            wandb.init(project='adept', entity='skvayzer', group='g1-sonic-bps128-touch',
                name=args.output.name, id='unique_id_'+args.output.name, resume='allow',
                dir=str(args.output), sync_tensorboard=True, mode='online', config=contract,
                tags=['sapg', 'sonic', 'g1', 'revo2', 'bps128', 'touch70', 'floating-body',
                      'continuous' if args.continuous else 'training-smoke'])
            if wandb.run is None or wandb.run.settings.mode != 'online':
                raise RuntimeError('Requested online logging did not initialize')
            wandb.run.summary.update(dict(experiment_status='initializing_simulation', optimizer_updates_started=False))
            (args.output/'wandb.json').write_text(json.dumps(dict(id=wandb.run.id, url=wandb.run.url), indent=2))
            print('WANDB_INITIALIZING_SIMULATION '+wandb.run.url, flush=True)
        sonic = FrozenSonic(workspace/'GRAIL', workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', args.device)
        actor, critic = load_touch_sources(args.device)
        student, bootstrap_report = load_student_checkpoint(args.bootstrap, sonic, args.device)
        bootstrap_config = load_yaml(args.bootstrap.parent/'config.json')
        if (bootstrap_report['source_sapg_sha256'] != TOUCH_SHA256 or student.task_dim != 249
                or bootstrap_config.get('policy_hz') != 60):
            raise ValueError('Require the completed touch teacher distilled on the 60 Hz task clock')
        env = G1SonicTouchEnv(cfg, sonic=sonic)
        if args.diagnostic_capture_seconds > 0:
            from dextrah_lab.wholebody.failure_recording import FailureRecorder
            diagnostic = FailureRecorder(env, args.output/'failure_clip', args.diagnostic_capture_seconds,
                                         epoch=lambda: 0 if algo is None else int(algo.epoch_num))
        register_sonic_models(actor, critic, sonic, env._body_lower[0], env._body_upper[0], student)
        params = agent['params']
        params['model']['name'] = 'sonic_sapg_logstd'
        params['network']['name'] = 'sonic_sapg_actor'
        # Per-task clipping now occurs inside TaskBodyNormalizer, so body qd
        # and executed-action histories are not clipped to legacy +/-10.
        params['env']['clip_observations'] = float('inf')
        params['load_checkpoint'], params['load_path'] = False, ''
        ac = params['config']
        ac.update(name='g1_sonic_sapg_bps128_touch', num_actors=args.num_envs,
            expl_coef_block_size=args.num_envs//6, minibatch_size=4*args.num_envs,
            device=args.device, device_name=args.device, multi_gpu=False,
            max_frames=-1, max_epochs=-1 if args.continuous else args.max_epochs,
            train_dir=str(args.output.parent), full_experiment_name=args.output.name)
        ac.pop('score_to_win', None)
        ac['central_value_config']['minibatch_size'] = 4*args.num_envs
        ac['central_value_config']['model'] = dict(name='sonic_sapg_value')
        ac['central_value_config']['network']['name'] = 'sonic_sapg_critic'
        agent['hydra']['run']['dir'] = str(args.output)
        agent['wandb_activate'] = args.wandb == 'online'
        agent['wandb_group'] = 'g1-sonic-bps128-touch'
        dump_yaml(str(args.output/'params/env_resolved.yaml'), cfg)
        dump_yaml(str(args.output/'params/agent.yaml'), agent)
        wrapped = register_rlgames_env(env, rl_device=args.device, clip_obs=float('inf'), clip_actions=1.)
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
        install_complete_checkpoint(algo)
        install_atomic_checkpoint_saves(algo)
        if args.resume:
            previous = json.loads((args.resume.parent.parent/'task_contract.json').read_text())
            for key in ('source_sha256', 'bank_sha256', 'bootstrap_sha256', 'action_units'):
                if previous[key] != contract[key]:
                    raise ValueError(f'Incompatible continuation contract: {key}')
            report['resume'] = resume_at_episode_boundary(algo, args.resume)
        if args.memory_trace:
            from dextrah_lab.wholebody.memory_diagnostics import install_memory_trace
            install_memory_trace(algo, args.output, args.memory_gc_probe_epoch)
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
            actions=35, task_reward='unchanged', touch_hz=70, body_decoder_trainable=True)), flush=True)
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
