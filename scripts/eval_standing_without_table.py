#!/usr/bin/env python3
"""Does the reposing policy stand on its own feet? Inference only, no optimizer.

Unchanged reposing task with the table unable to support the body (as in the
no-lean fine-tune). Per environment, once the secure-grasp gate has held for
0.5 s, that environment's table is parked 2 m below the floor until its next
reset. Reports robot falls within 3 s of removal, plus how often lower-body
link origins sit inside the table volume while the table is present (a
proxy: with filtered collisions a leg could stand inside the table).
"""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import traceback

LOWER_BODY = ('pelvis', 'left_hip_pitch_link', 'right_hip_pitch_link', 'left_hip_roll_link', 'right_hip_roll_link',
              'left_hip_yaw_link', 'right_hip_yaw_link', 'left_knee_link', 'right_knee_link',
              'left_ankle_pitch_link', 'right_ankle_pitch_link', 'left_ankle_roll_link', 'right_ankle_roll_link',
              'torso_link')


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an allocated Slurm step')
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=1200)
    p.add_argument('--seconds', type=float, default=30.)
    p.add_argument('--window', type=float, default=3.)
    p.add_argument('--torch-memory-limit-gib', type=float, default=5.)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    if len(args.checkpoints) != len(args.labels) or args.output.exists():
        raise ValueError('One label per checkpoint and a fresh output directory')
    args.output.mkdir(parents=True)
    args.headless = True
    app = AppLauncher(args).app
    faulthandler.enable(file=(args.output/'stacks.log').open('w'))
    report = dict(completed=False, optimizer_updates=0, source_commit=os.environ.get('FULLBODY_SOURCE_COMMIT'))
    try:
        import torch
        from rl_games.algos_torch import model_builder
        from dextrah_lab.g1_adept.touch_continuation import load_yaml
        from dextrah_lab.wholebody.sonic import FrozenSonic
        from dextrah_lab.wholebody.frozen_sapg import FrozenControllerAudit
        from dextrah_lab.wholebody.recording_contract import execution_means
        from dextrah_lab.wholebody.source_config import source_task_config
        from dextrah_lab.wholebody.sapg_network import load_touch_sources, register_sonic_models
        from dextrah_lab.wholebody.carry import grasp_gate
        from dextrah_lab.wholebody.carry_env import palm_center_w, TABLE_PARKING_Z
        from dextrah_lab.wholebody.brush_transfer import TABLE_SIZE
        from dextrah_lab.tasks.g1_revo2_adept.g1_sonic_touch_env import G1SonicTouchEnv
        torch.set_num_threads(4)
        total = torch.cuda.get_device_properties(args.device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., args.torch_memory_limit_gib*2**30/total), args.device)
        workspace = Path('/data1/users/konstantin.smirnov')
        run = args.checkpoints[0].parent.parent
        previous = json.loads((run/'task_contract.json').read_text())
        cfg, _ = source_task_config(args.output, args.num_envs, args.device)
        cfg.seed = 42
        cfg.sonic_body.from_dict(previous['body_termination'])
        cfg.sonic_body.numerical_failure_mode = 'reset'
        cfg.sonic_body.table_supports_body = False
        sonic = FrozenSonic(workspace/'GRAIL', workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', args.device)
        audit = FrozenControllerAudit(sonic)
        env = G1SonicTouchEnv(cfg, sonic=sonic)
        source, critic = load_touch_sources(args.device)
        register_sonic_models(source, critic, sonic, env._body_lower[0], env._body_upper[0], frozen=True)
        params = load_yaml(run/'params/agent.yaml')['params']
        n, device, dt = env.num_envs, env.device, env.step_dt
        names = env.robot.body_names
        lower = torch.tensor([names.index(b) for b in LOWER_BODY], device=device)
        half = torch.tensor(TABLE_SIZE, device=device)/2
        window = round(args.window/dt)
        results = {}
        torch.inference_mode().__enter__()
        for path, label in zip(args.checkpoints, args.labels):
            model = model_builder.ModelBuilder().load(params).build(dict(
                actions_num=70, input_shape=(1243+32,), num_seqs=n, value_size=1,
                normalize_value=True, normalize_input=True, type='extra_param',
                coef_ids=source.a2c_network.param_ids, coef_id_idx=1243)).to(device)
            payload = torch.load(path, map_location='cpu', weights_only=False)
            payload = payload[0] if 0 in payload else payload
            model.load_state_dict(payload['model'], strict=True)
            model.eval()
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            obs, _ = env.reset()
            states = tuple(x.to(device) for x in model.get_default_rnn_state())
            hold = torch.zeros(n, dtype=torch.long, device=device)
            removed_at = torch.full((n,), -1, dtype=torch.long, device=device)   # step of removal this episode
            trials = survived = fell_in_window = dropped_in_window = 0
            fall_times = []
            inside_steps = present_steps = 0
            max_depth = torch.zeros(len(LOWER_BODY), device=device)
            inside_per_link = torch.zeros(len(LOWER_BODY), device=device)
            falls_total = episodes = hits = 0
            for step in range(round(args.seconds/dt)):
                wrapped = torch.cat((obs['policy'], obs['policy'].new_zeros(n, 1)), -1)
                out = model(dict(obs=wrapped, is_train=False, prev_actions=None, rnn_states=states))
                states = out['rnn_states']
                obs, reward, terminated, truncated, info = env.step(execution_means(out['mus'], frozen=True))
                done = terminated | truncated
                final = info['episode_final']
                fell = done & final['robot_fall'].bool()
                falls_total += int(fell.sum()); episodes += int(done.sum())
                hits += int(final['successes'][done].sum()) if 'successes' in final else 0
                active = removed_at >= 0
                elapsed = step-removed_at
                # Outcome of removal trials: fall/drop within window, or survived the window.
                in_window = active & (elapsed <= window)
                fell_in_window += int((in_window & fell).sum())
                dropped_in_window += int((in_window & done & ~fell).sum())
                fall_times += ((elapsed[in_window & fell]).float()*dt).tolist()
                completed = active & (elapsed == window) & ~done
                survived += int(completed.sum())
                for s in states:
                    s[:, done, :] = 0
                removed_at[done] = -1
                hold[done] = 0
                # Leg-in-table proxy while the table is present.
                present = (removed_at < 0) & ~done
                rel = env.robot.data.body_pos_w[:, lower]-env.table.data.root_pos_w[:, None]
                depth = (half-rel.abs()).amin(-1)
                inside = (depth > 0) & present[:, None]
                inside_per_link += inside.float().sum(0)
                inside_steps += int(inside.any(-1).sum()); present_steps += int(present.sum())
                max_depth = torch.maximum(max_depth, torch.where(present[:, None], depth, torch.full_like(depth, -1)).amax(0))
                # Start new removal trials once the grasp gate has held 0.5 s.
                palm_c, palm = palm_center_w(env)
                o = env.object.data
                offset = o.root_pos_w-palm_c
                rel_v = o.root_lin_vel_w-(palm[:, 7:10]+torch.cross(palm[:, 10:13], offset, dim=-1))
                lift = o.root_pos_w[:, 2]-env.scene.env_origins[:, 2]-env._object_init_z
                gate = grasp_gate(lift, env.touch_raw[..., 0], rel_v.norm(dim=-1)) & ~done & (removed_at < 0)
                hold = torch.where(gate, hold+1, torch.zeros_like(hold))
                start = torch.nonzero(hold == 30).flatten()
                if start.numel():
                    pose = env.table.data.root_state_w[start, :7].clone()
                    pose[:, 2] = env.scene.env_origins[start, 2]+TABLE_PARKING_Z
                    env.table.write_root_pose_to_sim(pose, env_ids=start)
                    removed_at[start] = step
                    trials += start.numel()
                if (step+1) % 600 == 0:
                    print('STANDING_PROGRESS '+json.dumps(dict(label=label, sim_s=(step+1)*dt, trials=trials,
                        survived=survived, fell=fell_in_window, dropped=dropped_in_window)), flush=True)
            resolved = survived+fell_in_window+dropped_in_window
            fall_times.sort()
            results[label] = dict(checkpoint=str(path), sha256=sha, epoch=int(payload['epoch']),
                removal_trials=trials, resolved_trials=resolved,
                stood_3s=survived, fell_within_3s=fell_in_window, other_termination_within_3s=dropped_in_window,
                stand_rate=survived/resolved if resolved else None,
                fall_rate=fell_in_window/resolved if resolved else None,
                median_fall_time_s=fall_times[len(fall_times)//2] if fall_times else None,
                table_present_steps=present_steps,
                lower_body_inside_table_fraction=inside_steps/max(present_steps, 1),
                inside_fraction_per_link={b: float(v)/max(present_steps, 1) for b, v in zip(LOWER_BODY, inside_per_link)},
                max_inside_depth_m={b: float(v) for b, v in zip(LOWER_BODY, max_depth)},
                episodes=episodes, robot_falls_per_episode=falls_total/max(episodes, 1), goal_hits=hits)
            print('STANDING_RESULT '+json.dumps({k: v for k, v in results[label].items()
                                                if k not in ('inside_fraction_per_link', 'max_inside_depth_m')}), flush=True)
            del model
        report.update(completed=True, results=results, num_envs=n, seconds=args.seconds, window_s=args.window,
            table_supports_body=False, seed=42, policy='deterministic leader', **audit.check())
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        (args.output/'standing_eval.json').write_text(json.dumps(report, indent=2))
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)


if __name__ == '__main__':
    main()
