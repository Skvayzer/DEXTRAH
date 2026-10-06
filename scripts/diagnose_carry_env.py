#!/usr/bin/env python3
"""Inference-only diagnostic of carry-episode starts (no optimizer).

Runs the reposing expert in G1CarryEnv from restored grasps for a few seconds
under four conditions: table parked vs kept, and the expert's latent vs zero
latent (fingers always from the expert). Reports falls, drops, pelvis height,
root speed and latent/body-input magnitudes over time.
"""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import traceback


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an allocated Slurm step')
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--grasp-bank', type=Path, required=True)
    p.add_argument('--carry-clips', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=9216)
    p.add_argument('--seconds', type=float, default=3.)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
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
        from dextrah_lab.wholebody.recording_contract import execution_means
        from dextrah_lab.wholebody.source_config import source_task_config
        from dextrah_lab.wholebody.sapg_network import load_touch_sources, register_sonic_models
        from dextrah_lab.wholebody.carry_env import G1CarryEnv
        torch.set_num_threads(4)
        workspace = Path('/data1/users/konstantin.smirnov')
        run = args.checkpoint.parent.parent
        previous = json.loads((run/'task_contract.json').read_text())
        cfg, _ = source_task_config(args.output, args.num_envs, args.device)
        cfg.sonic_body.from_dict(previous['body_termination'])
        cfg.sonic_body.numerical_failure_mode = 'reset'
        cfg.episode_length_s = 20.
        cfg.termination.episode_length = 1200
        cfg.termination.max_consecutive_successes = 0
        sonic = FrozenSonic(workspace/'GRAIL', workspace/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', args.device)
        env = G1CarryEnv(cfg, sonic=sonic, clips_path=args.carry_clips, grasp_bank_path=args.grasp_bank)
        source, critic = load_touch_sources(args.device)
        register_sonic_models(source, critic, sonic, env._body_lower[0], env._body_upper[0], frozen=True)
        params = load_yaml(run/'params/agent.yaml')['params']
        model = model_builder.ModelBuilder().load(params).build(dict(
            actions_num=70, input_shape=(1243+32,), num_seqs=args.num_envs, value_size=1,
            normalize_value=True, normalize_input=True, type='extra_param',
            coef_ids=source.a2c_network.param_ids, coef_id_idx=1243)).to(args.device)
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict((payload[0] if 0 in payload else payload)['model'], strict=True)
        model.eval().requires_grad_(False)
        n = env.num_envs
        steps = round(args.seconds/env.step_dt)
        results = {}
        for park in (True, False):
            for latent in ('expert', 'zero'):
                name = f"{'parked' if park else 'table'}_{latent}_latent"
                env.park_table = park
                obs, _ = env.reset()
                states = tuple(x.to(env.device) for x in model.get_default_rnn_state())
                fell = torch.zeros(n, dtype=torch.bool, device=env.device)
                dropped = torch.zeros_like(fell)
                first_fall = torch.full((n,), float('nan'), device=env.device)
                timeline = []
                with torch.inference_mode():
                    for step in range(steps):
                        wrapped = torch.cat((obs['policy'], obs['policy'].new_zeros(n, 1)), -1)
                        out = model(dict(obs=wrapped, is_train=False, prev_actions=None, rnn_states=states))
                        states = out['rnn_states']
                        mu = execution_means(out['mus'], frozen=True)
                        action = mu.clone()
                        if latent == 'zero':
                            action[:, :64] = 0
                        body_extra = obs['policy'][:, 249:]
                        obs, _, terminated, truncated, _ = env.step(action)
                        newly = env._body_fallen & ~fell
                        first_fall[newly] = (step+1)*env.step_dt
                        fell |= env._body_fallen
                        dropped |= env._carry_dropped
                        done = terminated | truncated
                        for s in states:
                            s[:, done, :] = 0
                        if step % 6 == 0:
                            r = env.robot.data
                            timeline.append(dict(t=round((step+1)*env.step_dt, 3),
                                ever_fell=float(fell.float().mean()), ever_dropped=float(dropped.float().mean()),
                                pelvis_height=float((r.root_pos_w[:, 2]-env.scene.env_origins[:, 2]).mean()),
                                root_speed=float(r.root_lin_vel_w[:, :2].norm(dim=-1).mean()),
                                expert_latent_abs_mean=float(mu[:, :64].abs().mean()),
                                expert_latent_abs_max=float(mu[:, :64].abs().max()),
                                body_input_abs_max=float(body_extra.abs().max()),
                                body_history_qd_abs_max=float(body_extra[:, 3*10+29*10:3*10+29*20].abs().max())))
                ff = first_fall[~first_fall.isnan()]
                results[name] = dict(ever_fell=float(fell.float().mean()), ever_dropped=float(dropped.float().mean()),
                    first_fall_s_median=float(ff.median()) if ff.numel() else None,
                    first_fall_s_q10=float(ff.quantile(.1)) if ff.numel() else None, timeline=timeline)
                print('CARRY_DIAG '+name+' '+json.dumps({k: v for k, v in results[name].items() if k != 'timeline'}), flush=True)
                for row in timeline[:6]+timeline[-2:]:
                    print('CARRY_DIAG_T '+name+' '+json.dumps(row), flush=True)
        report.update(completed=True, conditions=results)
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        (args.output/'carry_diagnostic.json').write_text(json.dumps(report, indent=2))
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)


if __name__ == '__main__':
    main()
