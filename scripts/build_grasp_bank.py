#!/usr/bin/env python3
"""Capture secure-grasp states from the reposing expert for carry training.

Runs a frozen-SONIC SAPG checkpoint (deterministic leader, no optimizer) in the
unchanged reposing task with the table present. A grasp is captured when the
gate (thumb + another fingertip in contact, object >= 5 cm above its resting
reset height, low hand-object slip) has held for 0.5 s, and again every 0.75 s
while it keeps holding, up to K snapshots per environment. Snapshots are keyed
by environment index, i.e. by the physical object assigned to it.
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
        raise RuntimeError('Use an allocated Slurm step')
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=9216)
    p.add_argument('--seconds', type=float, default=60.)
    p.add_argument('--per-env', type=int, default=8)
    p.add_argument('--hold-steps', type=int, default=30)
    p.add_argument('--repeat-steps', type=int, default=45)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    if args.output.exists() or not args.checkpoint.is_file() or args.num_envs % 6:
        raise ValueError('Need an existing checkpoint, a fresh output and a multiple of six environments')
    args.output.mkdir(parents=True)
    args.headless = True
    app = AppLauncher(args).app
    stack_log = (args.output/'stacks.log').open('w')
    faulthandler.enable(file=stack_log)
    faulthandler.dump_traceback_later(120, repeat=True, file=stack_log)
    report = dict(completed=False, optimizer_updates=0, source_commit=os.environ.get('FULLBODY_SOURCE_COMMIT'))
    try:
        import torch
        from rl_games.algos_torch import model_builder
        from dextrah_lab.g1_adept.touch_continuation import load_yaml
        from dextrah_lab.wholebody.sonic import FrozenSonic, WEIGHTS_SHA256
        from dextrah_lab.wholebody.frozen_sapg import FrozenControllerAudit
        from dextrah_lab.wholebody.recording_contract import action_layout, execution_means
        from dextrah_lab.wholebody.source_config import source_task_config
        from dextrah_lab.wholebody.sapg_network import load_touch_sources, register_sonic_models
        from dextrah_lab.wholebody.carry import grasp_gate
        from dextrah_lab.wholebody.carry_env import SNAPSHOT_DIM, SNAPSHOT_LAYOUT, palm_center_w
        from dextrah_lab.tasks.g1_revo2_adept.g1_sonic_touch_env import G1SonicTouchEnv
        torch.set_num_threads(4)
        workspace = Path('/data1/users/konstantin.smirnov')
        run = args.checkpoint.parent.parent
        previous = json.loads((run/'task_contract.json').read_text())
        frozen, action_dim = action_layout(previous)
        if not frozen or previous.get('pretrained_sonic_sha256') != WEIGHTS_SHA256:
            raise ValueError('Grasp bank requires the frozen pretrained SONIC expert')
        cfg, contract = source_task_config(args.output, args.num_envs, args.device)
        cfg.sonic_body.from_dict(previous['body_termination'])
        sonic = FrozenSonic(workspace/'GRAIL', workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', args.device)
        audit = FrozenControllerAudit(sonic)
        env = G1SonicTouchEnv(cfg, sonic=sonic)
        source, critic = load_touch_sources(args.device)
        register_sonic_models(source, critic, sonic, env._body_lower[0], env._body_upper[0], frozen=True)
        params = load_yaml(run/'params/agent.yaml')['params']
        model = model_builder.ModelBuilder().load(params).build(dict(
            actions_num=action_dim, input_shape=(1243+32,), num_seqs=args.num_envs, value_size=1,
            normalize_value=True, normalize_input=True, type='extra_param',
            coef_ids=source.a2c_network.param_ids, coef_id_idx=1243)).to(args.device)
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        payload = payload[0] if 0 in payload else payload
        model.load_state_dict(payload['model'], strict=True)
        model.eval().requires_grad_(False)
        with args.checkpoint.open('rb') as f:
            checkpoint_sha = hashlib.file_digest(f, 'sha256').hexdigest()
        n, k_max, device = env.num_envs, args.per_env, env.device
        bank = torch.zeros(n, k_max, SNAPSHOT_DIM, device=device)
        count = torch.zeros(n, dtype=torch.long, device=device)
        hold = torch.zeros(n, dtype=torch.long, device=device)
        obs, _ = env.reset()
        states = tuple(x.to(device) for x in model.get_default_rnn_state())
        faulthandler.cancel_dump_traceback_later()
        start = time.monotonic()
        rule = (f'gate (thumb + >=1 finger normal > 0.2 N, lift > 5 cm above resting reset height, '
                f'slip < 0.1 m/s) held {args.hold_steps} steps, then every {args.repeat_steps} steps, '
                f'max {k_max} per env; deterministic leader')
        print('GRASP_BANK_START '+json.dumps(dict(envs=n, checkpoint_sha256=checkpoint_sha, rule=rule)), flush=True)
        steps = round(args.seconds/env.step_dt)
        with torch.inference_mode():
            for step in range(steps):
                wrapped = torch.cat((obs['policy'], obs['policy'].new_zeros(n, 1)), -1)
                out = model(dict(obs=wrapped, is_train=False, prev_actions=None, rnn_states=states))
                states = out['rnn_states']
                obs, reward, terminated, truncated, info = env.step(execution_means(out['mus'], frozen=True))
                done = terminated | truncated
                for s in states:
                    s[:, done, :] = 0
                origin = env.scene.env_origins
                palm_c, palm = palm_center_w(env)
                o = env.object.data
                offset = o.root_pos_w-palm_c
                rel = o.root_lin_vel_w-(palm[:, 7:10]+torch.cross(palm[:, 10:13], offset, dim=-1))
                lift = o.root_pos_w[:, 2]-origin[:, 2]-env._object_init_z
                gate = grasp_gate(lift, env.touch_raw[..., 0], rel.norm(dim=-1)) & ~done
                hold = torch.where(gate, hold+1, torch.zeros_like(hold))
                due = (hold == args.hold_steps) | ((hold > args.hold_steps) & ((hold-args.hold_steps) % args.repeat_steps == 0))
                take = torch.nonzero(due & (count < k_max)).flatten()
                if take.numel():
                    r = env.robot.data
                    root = r.root_state_w[take].clone(); root[:, :3] -= origin[take]
                    obj = o.root_state_w[take].clone(); obj[:, :3] -= origin[take]
                    snap = torch.cat((root, r.joint_pos[take], r.joint_vel[take], env._cur_targets[take],
                                      obj, offset[take].norm(dim=-1, keepdim=True)), -1)
                    bank[take, count[take]] = snap
                    count[take] += 1
                if (step+1) % 600 == 0:
                    print('GRASP_BANK_PROGRESS '+json.dumps(dict(sim_s=(step+1)*env.step_dt,
                        wall_s=time.monotonic()-start, coverage=float((count > 0).float().mean()),
                        full=float((count == k_max).float().mean()), snapshots=int(count.sum()))), flush=True)
                if bool((count == k_max).all()):
                    break
        objects = env._object_asset_index_per_env.cpu()
        covered = torch.zeros(int(objects.max())+1, dtype=torch.bool)
        covered[objects[count.cpu() > 0]] = True
        result = dict(snapshots=bank.cpu(), count=count.cpu(), num_envs=n, object_asset_index=objects,
                      snapshot_dim=SNAPSHOT_DIM, snapshot_layout=SNAPSHOT_LAYOUT, checkpoint=str(args.checkpoint),
                      checkpoint_sha256=checkpoint_sha, seconds=(step+1)*env.step_dt, capture_rule=rule,
                      seed=cfg.seed, pretrained_sonic_sha256=WEIGHTS_SHA256)
        torch.save(result, args.output/'grasp_bank.pt')
        with (args.output/'grasp_bank.pt').open('rb') as f:
            bank_sha = hashlib.file_digest(f, 'sha256').hexdigest()
        report.update(completed=True, checkpoint_sha256=checkpoint_sha, bank_sha256=bank_sha,
            env_coverage=float((count > 0).float().mean()), full_envs=float((count == k_max).float().mean()),
            object_coverage=float(covered[torch.unique(objects)].float().mean()),
            snapshots=int(count.sum()), simulated_seconds=(step+1)*env.step_dt,
            wall_seconds=time.monotonic()-start, rule=rule, **audit.check())
        print('GRASP_BANK_COMPLETE '+json.dumps(report), flush=True)
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        (args.output/'grasp_bank_result.json').write_text(json.dumps(report, indent=2))
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)  # skip Kit teardown, which hangs


if __name__ == '__main__':
    main()
