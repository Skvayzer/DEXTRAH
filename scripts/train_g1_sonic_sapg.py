#!/usr/bin/env python3
"""Continue BPS+touch SAPG with a trainable or frozen-pretrained SONIC body.

Original Play2Perfect task + original six-group SAPG Runner. No new manipulation
reward, no physics freezing, no fabrics/PCA, no periodic evaluation orchestrator.

Opt-in (mjlab): --teacher-checkpoint (another fixed-body teacher on the same task
contract instead of run 383) with --task-run (fresh start), --sgs (Success
Guided Sampling, dextrah_lab/sgs, whole-body port), --best-metric (best by the
SGS evaluation envs) and --minibatch-size (more PPO minibatches per epoch).
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


def mjlab_task_config(args):
    """The parent run's resolved task, verified against the touch-383 contract."""
    from dextrah_lab.mj.env import load_run_cfg
    from dextrah_lab.g1_adept.touch_continuation import load_yaml
    from dextrah_lab.wholebody.task_contract import TOUCH_RUN, verify_task_contract
    checkpoint = args.resume or args.init_weights
    parent = checkpoint.parent.parent if checkpoint else args.task_run
    cfg = load_run_cfg(parent)
    cfg.scene.num_envs = args.num_envs
    cfg.bps_artifact_dir = str(args.output/'bps')
    contract = verify_task_contract(load_yaml(Path(TOUCH_RUN)/'params/env_resolved.yaml'), cfg)
    contract['task_config_source'] = str(parent/'params/env_resolved.yaml')
    if args.teacher_checkpoint is not None:
        # The teacher must have solved this same task (fails closed otherwise).
        teacher_run = args.teacher_checkpoint.parent.parent
        verify_task_contract(load_yaml(teacher_run/'params/env_resolved.yaml'), cfg)
        contract['teacher_task_config_source'] = str(teacher_run/'params/env_resolved.yaml')
    return cfg, contract


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a Slurm GPU allocation')
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--backend', choices=['isaac', 'mjlab'], default='isaac')
    backend = pre.parse_known_args()[0].backend
    if backend == 'isaac':
        from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['isaac', 'mjlab'], default='isaac',
                   help='mjlab: MuJoCo Warp port of the run-1105 task (dextrah_lab/mj); isaac: unchanged PhysX path')
    p.add_argument('--mj-nconmax', type=int, help='mjlab: contacts allocated per world (pooled)')
    p.add_argument('--mj-njmax', type=int, help='mjlab: constraint rows per world')
    p.add_argument('--mj-cone-hessian-fix', action='store_true',
                   help='mjlab: require the vendored mujoco_warp with the elliptic cone Hessian fix (third_party/mujoco_warp_cone_fix)')
    p.add_argument('--mj-nan-capture', action='store_true',
                   help='mjlab diagnostic: record the first non-finite world to OUTPUT/nan_capture (no dynamics change)')
    p.add_argument('--mj-fast-step', action='store_true',
                   help='mjlab: G1SonicTouchMjEnv.enable_fast_step (kinematics-only refresh between steps, active-only '
                        'contact reader, SONIC encoder once per step + g1_dyn only); same simulation and observations, '
                        'checked by scripts/check_fast_env_step.py')
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
    p.add_argument('--grasp-reward-scale', type=float, default=None,
                   help='mjlab: + scale * InterMimicGen contact-prioritized grasp score with thumb-finger opposition '
                        'per step (dextrah_lab/wholebody/grasp_reward.py); 0 = compute and log only')
    p.add_argument('--object-energy-scale', type=float, default=0.,
                   help='mjlab: - scale * bounded object-acceleration energy per step (InterMimicGen regularizer)')
    p.add_argument('--dual-clip', type=float, default=0.,
                   help='Dual-clip PPO constant C > 1 (caps the actor loss of negative-advantage samples at -C A; '
                        'dextrah_lab/wholebody/dual_clip.py); 0: rl_games\' loss')
    p.add_argument('--lean-sapg-batches', action='store_true',
                   help='exact lower-peak-memory SAPG batch handling (dextrah_lab/wholebody/sapg_memory.py)')
    p.add_argument('--boundary-gc-freeze', action='store_true',
                   help="gc.freeze() after the first boundary collection (ms collections), plus a post-rollout collection and attribution of large cyclic garbage (OUTPUT/boundary_gc_garbage.jsonl)")
    p.add_argument('--boundary-gc-interval', type=int, default=0,
                   help='Full cyclic GC between completed updates; zero disables; never flushes CUDA cache')
    p.add_argument('--memory-gc-probe-epoch', type=int, default=0,
                   help='One diagnostic garbage collection after this absolute epoch; zero disables')
    p.add_argument('--wandb', choices=['online', 'disabled'], default='online')
    p.add_argument('--numerical-failure-mode', choices=['reset', 'abort'], default='reset',
                   help='Finite extreme joint speeds reset that environment; non-finite states always abort')
    p.add_argument('--diagnostic-capture-seconds', type=float, default=0.,
                   help='Opt-in physics-state ring buffer for reproducing a numerical failure')
    p.add_argument('--best-delay-epochs', type=int, default=200,
                   help='With --reset-best: epochs after the resume before best-return tracking starts')
    p.add_argument('--reset-best', action='store_true',
                   help='After --resume, forget the parent run best return so this run saves its own best checkpoint')
    p.add_argument('--body-table-contact-monitor-only', action='store_true',
                   help='Keep body-table contact sensors/metrics but do not end episodes on contact')
    p.add_argument('--spawn-on-table', action='store_true',
                   help='Re-place reset objects (orientation kept): no table intersection, footprint on the table')
    p.add_argument('--per-env-object-physics', action='store_true',
                   help='Upstream scene path: each environment simulates its own assigned object')
    p.add_argument('--body-table-contact-termination', action='store_true',
                   help='Real table collisions; end the episode when any body link except the right forearm/hand/fingers presses on the table')
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
    p.add_argument('--teacher-checkpoint', type=Path,
                   help='Frozen SONIC: fixed-body BPS+touch SAPG checkpoint (same task contract) whose recurrent features, '
                        'finger head, value head, normalizers and critic initialize the policy (default: completed run 383). '
                        'On --resume it is read from the parent contract')
    p.add_argument('--task-run', type=Path,
                   help='mjlab fresh start: whole-body run whose params/env_resolved.yaml defines the task')
    p.add_argument('--minibatch-size', type=int,
                   help='PPO minibatch for actor and critic (default 4*num_envs: 4 minibatches per epoch)')
    p.add_argument('--sgs', action='store_true',
                   help='mjlab: Success Guided Sampling of episode configurations (arXiv:2610.12465, dextrah_lab/sgs)')
    p.add_argument('--sgs-num-configs', type=int, default=32768)
    p.add_argument('--sgs-history', type=int, default=100)
    p.add_argument('--sgs-temperature', type=float, default=2.)
    p.add_argument('--sgs-target', type=float, default=.5)
    p.add_argument('--sgs-kappa', type=float, default=1.)
    p.add_argument('--sgs-eps', type=float, default=1e-4)
    p.add_argument('--sgs-families', default='reaching,stable_grasp,near_goal')
    p.add_argument('--sgs-eval-every', type=int, default=16,
                   help='Every Nth env resets from fresh rho samples (Reaching) and is logged as sgs/eval_*')
    p.add_argument('--sgs-harvest-max-steps', type=int, default=6000)
    p.add_argument('--sgs-reharvest-every', type=int, default=0,
                   help='Self-evolving held starts (InterMimicGen flywheel): env steps between re-harvest rounds (0: off)')
    p.add_argument('--sgs-reharvest-steps', type=int, default=300, help='Env steps of collection per round')
    p.add_argument('--sgs-reharvest-fraction', type=float, default=.2, help='Max share of each held family replaced per round')
    p.add_argument('--sgs-edit-fraction', type=float, default=.5, help='Share of replacements made by goal edits of verified held states')
    p.add_argument('--sgs-held-max-body-action', type=float, default=0.,
                   help='Whole body: harvest held states only while SONIC body actions (and history) stay within '
                        'this bound; saved held states beyond it are replaced on load (0: off)')
    p.add_argument('--sgs-hard-object-share', type=float, default=0.,
                   help='Share of training resets drawn object-first by hardness (InterMimicGen Eq. 9; 0: off)')
    p.add_argument('--sgs-validate-spawns', action='store_true',
                   help='Reaching configurations and evaluation starts must pass the passive settle check')
    p.add_argument('--sgs-eval-pool', type=int, default=16384,
                   help='With --sgs-validate-spawns: validated fresh rho starts the evaluation envs draw from')
    p.add_argument('--hand-surface-obs', action='store_true',
                   help='Frozen SONIC (mjlab): add 17 hand-point -> nearest object surface vectors (51 inputs, '
                        'wholebody/hand_surface.py) to actor and critic; with --init-weights from a run without them, '
                        'their network columns start at zero')
    p.add_argument('--hand-surface-arm-links', action='store_true',
                   help='With --hand-surface-obs: also the right elbow and wrist link -> object surface vectors '
                        '(12 inputs, InterMimicGen link-to-surface), appended')
    p.add_argument('--hand-surface-goal', action='store_true',
                   help='With --hand-surface-obs: also the 17 hand points -> object surface at the GOAL pose '
                        '(51 inputs, InterMimicGen reference geometry), appended')
    p.add_argument('--sgs-state', type=Path,
                   help='SGS state to continue (X.sgs.pt); default: the --resume checkpoint\'s')
    p.add_argument('--best-metric', choices=['reward', 'sgs_eval_goals', 'sgs_eval_success'], default='reward',
                   help='Best checkpoint by training reward (rl_games) or, with --sgs, by the rolling statistic of the '
                        'fresh-rho evaluation envs')
    if backend == 'isaac':
        AppLauncher.add_app_launcher_args(p)
    else:
        p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if backend == 'mjlab':
        if not (args.frozen_pretrained_sonic and args.per_env_object_physics and args.body_table_contact_monitor_only):
            raise ValueError('The mjlab port implements the run-1105 task: frozen SONIC, per-env objects, monitor-only table contact')
        if args.carry or args.diagnostic_capture_seconds or args.no_table_body_support or args.body_table_contact_termination:
            raise ValueError('Carry, failure capture and table-filter options exist only on the Isaac backend')
        if (args.resume or args.init_weights) is None and args.task_run is None:
            raise ValueError('mjlab training continues an existing run or names the task with --task-run')
        if args.task_run is not None and (args.resume or args.init_weights):
            raise ValueError('--task-run is for fresh starts (a continuation takes its task from the parent run)')
    if args.sgs and (backend != 'mjlab' or args.carry):
        raise ValueError('--sgs is implemented for the mjlab reposing task')
    if args.best_metric != 'reward' and not args.sgs:
        raise ValueError(f'--best-metric {args.best_metric} needs --sgs (its evaluation envs)')
    if args.teacher_checkpoint is not None and (not args.frozen_pretrained_sonic or not args.teacher_checkpoint.is_file()):
        raise ValueError('--teacher-checkpoint initializes a frozen-SONIC policy from an existing checkpoint')
    if (args.hand_surface_arm_links or args.hand_surface_goal) and not args.hand_surface_obs:
        raise ValueError('--hand-surface-arm-links / --hand-surface-goal extend --hand-surface-obs')
    if args.hand_surface_obs and (backend != 'mjlab' or not args.frozen_pretrained_sonic or args.carry):
        raise ValueError('--hand-surface-obs is implemented for the frozen-SONIC mjlab reposing task')
    if args.sgs_state is not None and (not args.sgs or not args.sgs_state.is_file()):
        raise ValueError('--sgs-state needs --sgs and an existing X.sgs.pt')
    parent_checkpoint = args.resume or args.init_weights
    if parent_checkpoint is not None and (parent_checkpoint.parent.parent/'task_contract.json').exists():
        # A continuation or fine-tune keeps its teacher (the network architecture was built from it).
        parent_contract = json.loads((parent_checkpoint.parent.parent/'task_contract.json').read_text())
        parent_teacher = parent_contract.get('source_checkpoint')
        from dextrah_lab.wholebody.task_contract import TOUCH_CHECKPOINT as _run383
        if args.teacher_checkpoint is None and parent_teacher not in (None, _run383):
            args.teacher_checkpoint = Path(parent_teacher)
        elif args.teacher_checkpoint is not None and str(args.teacher_checkpoint) != parent_teacher:
            raise ValueError(f'Continuation teacher {args.teacher_checkpoint} != parent {parent_teacher}')
        from dextrah_lab.wholebody.hand_surface import HAND_SURFACE_DIM, hand_surface_dim
        parent_hand = bool(parent_contract.get('body_termination', {}).get('hand_surface_obs', False))
        parent_dim = int(parent_contract.get('hand_surface', {}).get('dim', HAND_SURFACE_DIM if parent_hand else 0))
        new_dim = hand_surface_dim(args.hand_surface_arm_links, args.hand_surface_goal) if args.hand_surface_obs else 0
        if args.resume is not None and parent_dim != new_dim:
            raise ValueError(f'Hand-surface block {new_dim} but the resumed run used {parent_dim} (use --init-weights)')
        if new_dim < parent_dim:
            raise ValueError(f'Hand-surface block {new_dim} smaller than the parent run\'s {parent_dim}')
        args.expand_hand_inputs = new_dim > parent_dim
        args.parent_hand_dim = parent_dim
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
                  'continuous' if args.continuous else 'training-smoke', 'backend-'+backend] + (['carry-expert'] if args.carry else []))
        if wandb.run is None or wandb.run.settings.mode != 'online':
            raise RuntimeError('Requested online logging did not initialize')
        wandb.run.summary.update(dict(experiment_status='initializing_simulation', optimizer_updates_started=False))
        (args.output/'wandb.json').write_text(json.dumps(dict(id=wandb.run.id, url=wandb.run.url), indent=2))
        print('WANDB_INITIALIZING_SIMULATION '+wandb.run.url, flush=True)
    try:
        app = AppLauncher(args).app if backend == 'isaac' else None
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
        if backend == 'mjlab':
            from dextrah_lab.mj.compat import install
            install()
            from dextrah_lab.mj.rlgames import register_mj_rlgames_env
            import yaml
            def dump_yaml(path, data):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(yaml.dump(data.to_dict() if hasattr(data, 'to_dict') else data))
        else:
            from isaaclab.utils.io import dump_yaml
            from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
        from isaacsimenvs.utils.rlgames_utils import EnvStatsAlgoObserver, MultiObserver
        from rl_games.torch_runner import Runner
        from dextrah_lab.g1_adept.touch_continuation import (load_yaml, install_complete_checkpoint,
            install_atomic_checkpoint_saves, resume_at_episode_boundary)
        from dextrah_lab.wholebody.sonic import FrozenSonic, WEIGHTS_SHA256
        from dextrah_lab.wholebody.frozen_sapg import (FROZEN_ARCHITECTURE,
            install_latent_action_bounds, install_latent_optimizer)
        from dextrah_lab.wholebody.student_checkpoint import load_student_checkpoint
        from dextrah_lab.wholebody.sapg_network import load_touch_sources, register_sonic_models
        from dextrah_lab.wholebody.task_contract import TOUCH_RUN, TOUCH_SHA256, P2P_REVISION
        from dextrah_lab.wholebody.training_observer import SonicTransferObserver
        if backend == 'isaac':
            from dextrah_lab.wholebody.source_config import source_task_config
            from dextrah_lab.tasks.g1_revo2_adept.g1_sonic_touch_env import G1SonicTouchEnv
        torch.set_num_threads(4)
        workspace = Path('/data1/users/konstantin.smirnov')
        revision = subprocess.check_output(['git', '-C', str(workspace/'play2perfect'), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != P2P_REVISION:
            raise ValueError('Pinned Play2Perfect revision changed')
        if backend == 'mjlab':
            cfg, contract = mjlab_task_config(args)
        else:
            cfg, contract = source_task_config(args.output, args.num_envs, args.device)
        cfg.sonic_body.numerical_failure_mode = args.numerical_failure_mode
        cfg.sonic_body.table_supports_body = not args.no_table_body_support
        cfg.sonic_body.body_table_contact_termination = args.body_table_contact_termination
        cfg.sonic_body.per_env_object_physics = args.per_env_object_physics
        cfg.sonic_body.spawn_on_table = args.spawn_on_table
        cfg.sonic_body.hand_surface_obs = args.hand_surface_obs
        cfg.sonic_body.hand_surface_arm_links = args.hand_surface_arm_links
        cfg.sonic_body.hand_surface_goal = args.hand_surface_goal
        if args.body_table_contact_monitor_only:
            cfg.sonic_body.body_table_contact_termination = True
            cfg.sonic_body.body_table_contact_terminates = False
        if args.body_table_contact_termination and args.no_table_body_support:
            raise ValueError('Contact termination requires real table collisions')
        frozen = args.frozen_pretrained_sonic
        cfg.sonic_body.controller_mode = 'frozen_pretrained_latent' if frozen else 'trainable_decoder'
        agent = load_yaml(Path(TOUCH_RUN)/'params/agent.yaml')
        teacher_sha, teacher_frames = None, 17364025344
        if args.teacher_checkpoint is not None:
            with args.teacher_checkpoint.open('rb') as f:
                teacher_sha = hashlib.file_digest(f, 'sha256').hexdigest()
            teacher_payload = torch.load(args.teacher_checkpoint, map_location='cpu', weights_only=False)
            teacher_frames = int((teacher_payload[0] if 0 in teacher_payload else teacher_payload)['frame'])
            del teacher_payload
            teacher_agent = load_yaml(args.teacher_checkpoint.parent.parent/'params/agent.yaml')
            if teacher_agent['params']['network'] != agent['params']['network']:
                raise ValueError('Teacher network differs from the run-383 SAPG architecture')
            contract.update(source_run=str(args.teacher_checkpoint.parent.parent), source_checkpoint=str(args.teacher_checkpoint),
                            source_sha256=teacher_sha, teacher_replaces_run383=True)
        ac_cfg = agent['params']['config']
        minibatch_size = args.minibatch_size or 4*args.num_envs
        if (ac_cfg['horizon_length']*args.num_envs) % minibatch_size:
            raise ValueError(f'--minibatch-size {minibatch_size} must divide the rollout '
                             f'{ac_cfg["horizon_length"]}*{args.num_envs}')
        optimization = dict(minibatch_size=minibatch_size,
            minibatches_per_epoch=ac_cfg['horizon_length']*args.num_envs//minibatch_size,
            actor_gradient_steps_per_epoch=ac_cfg['mini_epochs']*ac_cfg['horizon_length']*args.num_envs//minibatch_size,
            critic_gradient_steps_per_epoch=ac_cfg['central_value_config']['mini_epochs']*ac_cfg['horizon_length']*args.num_envs//minibatch_size)
        contract['optimization'] = optimization
        print('OPTIMIZATION '+json.dumps(optimization), flush=True)
        sgs_settings = None
        if args.sgs:
            from dataclasses import asdict
            from dextrah_lab.sgs.mj_task import SgsSettings
            sgs_settings = SgsSettings(num_configs=args.sgs_num_configs, history=args.sgs_history,
                temperature=args.sgs_temperature, target=args.sgs_target, kappa=args.sgs_kappa, eps=args.sgs_eps,
                families=tuple(args.sgs_families.split(',')), eval_every=args.sgs_eval_every,
                harvest_max_steps=args.sgs_harvest_max_steps, validate_spawns=args.sgs_validate_spawns,
                eval_pool_size=args.sgs_eval_pool, reharvest_every=args.sgs_reharvest_every,
                reharvest_steps=args.sgs_reharvest_steps, reharvest_fraction=args.sgs_reharvest_fraction,
                edit_fraction=args.sgs_edit_fraction, hard_object_share=args.sgs_hard_object_share,
                held_max_body_action=args.sgs_held_max_body_action)
            contract['sgs'] = dict(settings=asdict(sgs_settings), paper='arXiv:2610.12465', best_metric=args.best_metric)
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
            source_training_transitions=teacher_frames, source_commit=report['source_commit'])
        contract.update(controller_mode=cfg.sonic_body.controller_mode,
            pretrained_sonic_sha256=WEIGHTS_SHA256, pretrained_sonic_frozen=frozen,
            action_dim=70 if frozen else 35)
        if frozen:
            contract.update(architecture=FROZEN_ARCHITECTURE, latent_residual_scale=.1,
                initial_latent_std_leader=1., adapter_learning_rate=args.adapter_learning_rate,
                body_joint_exploration_noise=False, latent_clipping=False,
                action_delay='source delay on latent/finger meta-actions; SONIC uses current body feedback',
                initialization=('completed touch-383' if teacher_sha is None else 'teacher')
                    +' SAPG features/fingers/critic; zero-initialized new latent output',
                old_arm_policy_exactly_preserved=False, pretrained_sonic_audit='all tensors bitwise after every update',
                online_teacher_loss=False, new_balance_reward=False)
            contract['intentional_changes'] = [x for x in contract['intentional_changes']
                if x != '29_body_plus_6_right_finger_actions'] + [
                '64_latent_plus_6_right_finger_meta_actions', 'frozen_pretrained_SONIC_in_environment',
                'exploration_and_delay_before_body_decoding', 'latent_adapter_optimizer_group']
        else:
            contract['new_body_exploration_std_rad'] = .025
        contract['body_termination'] = cfg.sonic_body.to_dict()
        if args.body_table_contact_monitor_only:
            contract['intentional_changes'] = contract['intentional_changes'] + [
                'body_table_contact_logged_only_no_termination']
        if args.spawn_on_table:
            contract['intentional_changes'] = contract['intentional_changes'] + [
                'object_spawn_above_table_footprint_on_table']
        if args.per_env_object_physics:
            contract['intentional_changes'] = contract['intentional_changes'] + [
                'per_env_object_physics_upstream_scene_path_no_physx_replication']
        if args.sgs:
            contract['intentional_changes'] = contract['intentional_changes'] + ['sgs_success_guided_episode_configurations'] + (
                ['sgs_spawn_validation'] if args.sgs_validate_spawns else []) + (
                ['sgs_self_evolving_held_starts'] if args.sgs_reharvest_every else []) + (
                ['sgs_hardness_weighted_object_sampling'] if args.sgs_hard_object_share else [])
        if args.body_table_contact_termination:
            contract['intentional_changes'] = contract['intentional_changes'] + [
                'terminate_on_body_table_contact_except_right_forearm_hand_fingers']
        if args.no_table_body_support:
            contract['intentional_changes'] = contract['intentional_changes'] + [
                'table_collides_only_with_right_forearm_hand_fingers']
        contract['nonfinite_state_handling'] = 'abort before rewards and terminal observations'
        contract['finite_numerical_failure_reward'] = 'unchanged source reward; separately logged true termination'
        contract['touch_global_partners'] = 'enabled static collision shapes under /World/ground'
        contract['memory_trace'] = args.memory_trace
        contract['memory_gc_probe_epoch'] = args.memory_gc_probe_epoch
        contract['boundary_gc_interval'] = args.boundary_gc_interval
        contract['boundary_gc_freeze'] = bool(args.boundary_gc_freeze)
        contract['lean_sapg_batches'] = bool(args.lean_sapg_batches)
        if args.dual_clip:
            contract['dual_clip'] = dict(c=args.dual_clip, reference='Ye et al. AAAI 2020')
            contract['intentional_changes'] = contract['intentional_changes'] + ['ppo_dual_clip']
        if args.grasp_reward_scale is not None or args.object_energy_scale:
            from dextrah_lab.wholebody import grasp_reward as gr
            contract['grasp_reward'] = dict(
                scale=args.grasp_reward_scale or 0., object_energy_scale=args.object_energy_scale,
                source='InterMimicGen arXiv:2610.06850, App. C.2 (Eqs. 10-11) and C.1 (object energy)',
                weights=gr.WEIGHTS, sigma_m=gr.SIGMA, contact_gap_m=gr.D_CONTACT, force_threshold_n=gr.FORCE_THRESHOLD,
                definition=gr.__doc__.strip())
            contract['intentional_changes'] = contract['intentional_changes'] + (
                ['grasp_opposition_reward'] if args.grasp_reward_scale else []) + (
                ['object_energy_penalty'] if args.object_energy_scale else [])
        contract['cuda_allocator_config'] = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
        contract['periodic_stack_dumps_during_training'] = False
        contract['wandb_initialization'] = 'before_Isaac_Kit_asyncio_patch' if backend == 'isaac' else 'standard'
        if backend == 'isaac':
            import carb
            settings = carb.settings.get_settings()
            contract['runtime_worker_settings'] = {key: settings.get(key) for key in (
                '/plugins/carb.tasking.plugin/threadCount', '/plugins/omni.tbb.globalcontrol/maxThreadCount',
                '/app/asyncRendering', '/app/asyncRenderingLowLatency')}
        else:
            import mujoco, mujoco_warp, mjlab
            from importlib.metadata import version
            vendored = 'mujoco_warp_cone_fix' in Path(mujoco_warp.__file__).parts
            if args.mj_cone_hessian_fix != vendored:
                raise RuntimeError(f'--mj-cone-hessian-fix={args.mj_cone_hessian_fix} but mujoco_warp is {mujoco_warp.__file__}')
            from dextrah_lab.mj.factory import DEFAULT_MUJOCO, DEFAULT_NCONMAX, DEFAULT_NJMAX
            from dextrah_lab.mj.robot import ContactParams
            from dataclasses import asdict
            mj_settings = dict(backend='mjlab', mjlab=version('mjlab'), mujoco=mujoco.__version__,
                mujoco_warp=version('mujoco-warp'), torch=torch.__version__,
                nconmax=args.mj_nconmax or DEFAULT_NCONMAX, njmax=args.mj_njmax or DEFAULT_NJMAX,
                mujoco_options=DEFAULT_MUJOCO, contact=asdict(ContactParams()),
                joint_velocity_limits='body: PD setpoint rate-limited at the PhysX max joint velocity; Revo2: velocity servo kv=effort/vmax, v_cmd=clamp(kp*err/kv, +-vmax) (robot.hand_servo)',
                asset_parity='scripts/mj_asset_parity.py: 0 mismatches vs Isaac dump (job 1112)',
                nan_capture=bool(args.mj_nan_capture),
                fast_step=dict(enabled=bool(args.mj_fast_step), check='scripts/check_fast_env_step.py',
                               components=['kinematics refresh (fwd_kinematics + com_vel) instead of sim.forward() after '
                                           'physics substeps and after resets', 'contact reader over the active contacts',
                                           'frozen SONIC: reference encoder once per policy step, g1_dyn decoder only']),
                mujoco_warp_path=str(Path(mujoco_warp.__file__).parent),
                cone_hessian_fix=dict(enabled=True, sites=3, change='T=max(T,1e-12); ttt=T^3 (was max(T^3,1e-15))',
                                      solver_sha256=hashlib.sha256(Path(mujoco_warp.__file__).with_name('_src').joinpath('solver.py').read_bytes()).hexdigest())
                if args.mj_cone_hessian_fix else dict(enabled=False))
            contract['simulator'] = mj_settings
            contract['intentional_changes'] = contract['intentional_changes'] + ['simulator_mjlab_mujoco_warp'] + (
                ['mujoco_warp_elliptic_cone_hessian_fix'] if args.mj_cone_hessian_fix else [])
        (args.output/'task_contract.json').write_text(json.dumps(contract, indent=2))
        if args.wandb == 'online':
            wandb.config.update(contract, allow_val_change=True)
        sonic = FrozenSonic(workspace/'GRAIL', workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', args.device)
        actor, critic = load_touch_sources(args.device, args.teacher_checkpoint, teacher_sha)
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
        elif backend == 'mjlab':
            from dextrah_lab.mj.env import G1SonicTouchMjEnv
            env = G1SonicTouchMjEnv(cfg, sonic=sonic, num_envs=args.num_envs, device=args.device,
                                    nconmax=mj_settings['nconmax'], njmax=mj_settings['njmax'],
                                    mujoco_cfg=mj_settings['mujoco_options'])
            if args.grasp_reward_scale is not None or args.object_energy_scale:
                env.enable_grasp_reward(args.grasp_reward_scale or 0., args.object_energy_scale)
            if args.mj_fast_step:
                env.enable_fast_step()
            if args.mj_nan_capture:
                env.enable_nan_capture(args.output/'nan_capture')
            if args.sgs:
                # A resumed SGS run continues its configuration set and outcome windows.
                sgs_file = args.sgs_state or (args.resume.with_suffix('.sgs.pt') if args.resume else None)
                sgs_state = (torch.load(sgs_file, map_location='cpu', weights_only=False)
                             if sgs_file is not None and sgs_file.exists() else None)
                env.enable_sgs(sgs_settings, sgs_state)
                report['sgs'] = dict(contract['sgs'], resumed_from=str(sgs_file) if sgs_state else None,
                                     spawn_check=dict(env._sgs.spawn_stats))
                print('SGS_ENABLED '+json.dumps(report['sgs']), flush=True)
        else:
            env = G1SonicTouchEnv(cfg, sonic=sonic)
        if args.diagnostic_capture_seconds > 0:
            from dextrah_lab.wholebody.failure_recording import FailureRecorder
            diagnostic = FailureRecorder(env, args.output/'failure_clip', args.diagnostic_capture_seconds,
                                         epoch=lambda: 0 if algo is None else int(algo.epoch_num))
        hand_dim = getattr(env, 'hand_surface_dim', 0)
        if args.hand_surface_obs:
            from dextrah_lab.wholebody import hand_surface
            contract['intentional_changes'] = contract['intentional_changes'] + [f'hand_to_surface_observation_{hand_dim}d']
            contract['hand_surface'] = dict(points=['palm_centre', '5_fingertip_pads', *hand_surface.HAND_SURFACE_LINKS]
                                            + (list(hand_surface.ARM_SURFACE_LINKS) if args.hand_surface_arm_links else []),
                goal_surface=('the 17 hand points -> object surface at the goal pose, appended (actor = critic: goal is exact)'
                              if args.hand_surface_goal else None),
                arm_links=bool(args.hand_surface_arm_links), goal=bool(args.hand_surface_goal),
                dim=hand_dim, encoding='world-axis vector to nearest object-primitive surface, decimetres, norm clip 3',
                actor_object_pose='same delayed/noised pose as the other actor object observations',
                critic_object_pose='true', network='zero-initialized input columns in the task LSTM and critic layer 1')
            (args.output/'task_contract.json').write_text(json.dumps(contract, indent=2))
            if args.wandb == 'online':
                wandb.config.update(dict(hand_surface=contract['hand_surface'], intentional_changes=contract['intentional_changes']),
                                    allow_val_change=True)
        register_sonic_models(actor, critic, sonic, env._body_lower[0], env._body_upper[0], student, frozen=frozen,
                              hand_dim=hand_dim)
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
            expl_coef_block_size=args.num_envs//6, minibatch_size=minibatch_size,
            device=args.device, device_name=args.device, multi_gpu=False,
            max_frames=-1, max_epochs=-1 if args.continuous else args.max_epochs,
            train_dir=str(args.output.parent), full_experiment_name=args.output.name)
        ac.pop('score_to_win', None)
        if frozen:
            ac['clip_actions'] = False
            ac['name'] = 'g1_frozen_sonic_sapg_bps128_touch'
            ac['adapter_learning_rate'] = args.adapter_learning_rate
        ac['central_value_config']['minibatch_size'] = minibatch_size
        ac['central_value_config']['model'] = dict(name='sonic_sapg_value')
        ac['central_value_config']['network']['name'] = 'sonic_sapg_critic'
        agent['hydra']['run']['dir'] = str(args.output)
        agent['wandb_activate'] = args.wandb == 'online'
        agent['wandb_group'] = 'g1-sonic-bps128-touch'
        dump_yaml(str(args.output/'params/env_resolved.yaml'), cfg)
        dump_yaml(str(args.output/'params/agent.yaml'), agent)
        register = register_mj_rlgames_env if backend == 'mjlab' else register_rlgames_env
        wrapped = register(env, rl_device=args.device, clip_obs=float('inf'),
                           clip_actions=float('inf') if frozen else 1.)
        observer = SonicTransferObserver(env, args.output)
        observers = [EnvStatsAlgoObserver(), observer]
        eval_best = None
        if args.best_metric != 'reward':
            from dextrah_lab.sgs.best import SgsEvalBest
            eval_best = SgsEvalBest(env, args.best_metric, args.best_delay_epochs)
            observers.append(eval_best)
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
        if args.sgs:
            # Every network checkpoint X.pth gets the SGS state beside it (X.sgs.pt).
            network_save = algo.save
            def save_with_sgs(filename, override_state=None):
                network_save(filename, override_state)
                target = Path(str(filename)+'.sgs.pt')
                torch.save(env._sgs.state_dict(), str(target)+'.pending')
                os.replace(str(target)+'.pending', target)
            algo.save = save_with_sgs
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
            if args.reset_best and eval_best is None:
                # The parent's best return belongs to a different MDP; keeping
                # it would block every best-checkpoint save in this run. The
                # first post-resume returns still include episodes from before
                # the change (run 1063 locked its best at the 2nd epoch), and
                # SAPG's continuous path saves from absolute epoch 10. So block
                # saves until enough episodes have run under the new rules.
                report['resume']['parent_best_return_discarded'] = float(algo.last_mean_rewards)
                from dextrah_lab.wholebody.carry_training import install_delayed_best
                report['resume']['best_tracking'] = install_delayed_best(algo, args.best_delay_epochs)
        if eval_best is not None:
            # rl_games' reward-based best never saves; best follows the SGS evaluation envs.
            report['best_tracking'] = eval_best.disable_reward_best()
        if args.init_weights is not None:
            from dextrah_lab.wholebody.carry_training import load_initial_weights, install_critic_warmup
            if getattr(args, 'expand_hand_inputs', False):
                from dextrah_lab.wholebody.hand_surface import check_zero_hand_columns, load_expanded_initial_weights
                report['init_weights'] = load_expanded_initial_weights(algo, args.init_weights, hand_dim)
                report['init_weights']['inert_hand_inputs'] = check_zero_hand_columns(
                    algo.model, algo.central_value_net.model, env.cfg.observation_space, env.cfg.state_space,
                    hand_dim-args.parent_hand_dim, args.device)
                print('HAND_SURFACE_INIT '+json.dumps(report['init_weights']['inert_hand_inputs']), flush=True)
            else:
                report['init_weights'] = load_initial_weights(algo, args.init_weights, reset_values=args.carry)
            if args.critic_warmup_epochs:
                install_critic_warmup(algo, args.critic_warmup_epochs)
        if args.dual_clip:
            from dextrah_lab.wholebody.dual_clip import install_dual_clip
            report['dual_clip'] = install_dual_clip(algo, args.dual_clip)
        if args.lean_sapg_batches:
            from dextrah_lab.wholebody.sapg_memory import install_lean_sapg_batches
            report['lean_sapg_batches'] = install_lean_sapg_batches(algo)
        if args.memory_trace:
            from dextrah_lab.wholebody.memory_diagnostics import install_memory_trace
            install_memory_trace(algo, args.output, args.memory_gc_probe_epoch)
        if args.boundary_gc_interval:
            from dextrah_lab.wholebody.memory_cleanup import install_boundary_gc
            install_boundary_gc(algo, args.output, args.boundary_gc_interval, freeze=args.boundary_gc_freeze)
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
        if not report['completed'] or backend == 'mjlab':
            import sys
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(0 if report['completed'] else 1)
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
