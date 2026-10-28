#!/usr/bin/env python3
"""Paired counterfactual transitions for the hand-object world model (2026-10-28).

The stage-1.5 model is action-blind (gradient check v5: its finger-command effects are 4-6% of MuJoCo Warp's,
uncorrelated) because in the expert's data the finger targets are almost a function of the state
(action_sensitivity.py: R^2 0.94-1.00 for 5 of 6 joints). Here groups of K worlds are restored from one holding
state with domain randomisation off, tables parked and no mass / COM / jitter / pushes, so they are bit-identical
at the restore. Copy 0 follows the carry expert; copies 1..K-1 add independent finger-command perturbations from
2 control steps after the restore. The hand-object system stays deterministic to ~0.001-0.01 mm for ~0.1 s
(replica_divergence.py), so over --horizon-s the differences between copies are the effect of the commands.

Per round: one env.reset() with new holding states (the world model's held-out objects excluded), t0 + H + 1
control steps, rows t0-2 .. t0+H as in gradient_check.py (row 1 = the last unperturbed state). Perturbation of
copy c >= 1 (command units, on top of the expert's, clamped to [-1, 1]), one kind per copy in turn:
  step   constant offset U(-a, a) per finger for the whole window;
  pulse  constant offset U(-a, a) per finger for U{1..P} steps from t0;
  noise  i.i.d. N(0, s) per finger and step.
Output: round_<i>.pt = the logged fields (rows, n, d) + done + bps / asset per world; meta.json.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside a Slurm GPU allocation')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True, help='carry expert')
    p.add_argument('--world-model', type=Path, required=True, help='its held-out objects are excluded')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--groups', type=int, default=2048)
    p.add_argument('--copies', type=int, default=8, help='K: copy 0 unperturbed')
    p.add_argument('--rounds', type=int, default=30)
    p.add_argument('--horizon-s', type=float, default=.134, help='8 control steps')
    p.add_argument('--amplitude', type=float, default=.15, help='a: step / pulse offsets U(-a, a)')
    p.add_argument('--pulse-steps', type=int, default=4, help='P')
    p.add_argument('--noise-std', type=float, default=.08, help='s')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = dict(completed=False, args={k: str(v) for k, v in vars(args).items()})
    try:
        from dextrah_lab.mj.compat import install
        install()
        report.update(run(args))
        report['completed'] = True
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (args.output/'meta.json').write_text(json.dumps(report, indent=2))
        print('COUNTERFACTUAL_DONE '+json.dumps({k: report.get(k) for k in ('completed', 'error', 'stats')}), flush=True)
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)


def run(args):
    import torch
    from dextrah_lab.neurosim.env_io import build_carry_env, hand_joint_ids, snapshot
    from dextrah_lab.wholebody.carry_compose import finger_command
    t_start = time.time()
    G, K = args.groups, args.copies
    n = G*K                                                       # world = group * K + copy
    override = dict(mass_scale=(1., 1.), com_offset_m=0., disturbance=0., push_probability=0., carry_jitter=0.,
                    table_keep_probability=0.)
    env, policy, contract, settings = build_carry_env(args.checkpoint, n, settings_override=override, seed=args.seed,
                                                      no_randomization=True)
    dev, dt = env.device, env.step_dt
    ck = torch.load(args.world_model, map_location='cpu', weights_only=False)
    held_out = torch.tensor(ck['test_assets'], device=dev)
    allowed = torch.nonzero(~torch.isin(env._sgs.assets, held_out)).flatten()       # indices into env._sgs.configs
    env._sgs.eval_mask[:] = False
    assigned = {'idx': None}
    env._sgs.sampler.sample = lambda k, generator=None: assigned['idx'][:k]
    hand_ids = hand_joint_ids(env)
    close, open_ = settings.finger_close, settings.finger_open
    t0 = 2
    H = round(args.horizon_s/dt)
    copy = torch.arange(n, device=dev) % K
    kind = torch.where(copy == 0, -1, (copy-1) % 3)               # -1 baseline, 0 step, 1 pulse, 2 noise
    g = torch.Generator(device=dev).manual_seed(args.seed+11)
    gc = torch.Generator().manual_seed(args.seed)
    ended = torch.zeros(n, dtype=torch.bool, device=dev)
    record = env._carry_record

    def recording(ids):
        ended[ids] = True
        return record(ids)
    env._carry_record = recording
    stats = dict(worlds=n, groups=G, copies=K, rounds=args.rounds, horizon_steps=H, t0_steps=t0, allowed_configs=int(allowed.numel()),
                 held_out_assets=int(held_out.numel()), groups_with_end=0, perturbation_abs_mean=0.)
    for rnd in range(args.rounds):
        chosen = allowed[torch.randperm(allowed.numel(), generator=gc)[:G].to(dev)]
        assigned['idx'] = chosen.repeat_interleave(K)
        with torch.inference_mode():                              # env buffers are inference tensors after round 0
            obs, _ = env.reset()
        ended[:] = False                                          # the reset records the previous round's worlds
        states = tuple(s.to(dev) for s in policy.get_default_rnn_state())
        leader = torch.zeros(n, 1, device=dev)
        step_off = (torch.rand(n, 6, device=dev, generator=g)*2-1)*args.amplitude
        pulse_len = torch.randint(1, args.pulse_steps+1, (n,), device=dev, generator=g)
        log, applied = {}, []
        with torch.inference_mode():
            for step in range(t0+H+1):
                out = policy(dict(obs=torch.cat((obs['policy'], leader), -1), is_train=False, prev_actions=None, rnn_states=states))
                states = out['rnn_states']
                raw = out['mus'].clone()
                if step >= t0:
                    offset = torch.zeros(n, 6, device=dev)
                    offset = torch.where((kind == 0)[:, None], step_off, offset)
                    offset = torch.where(((kind == 1) & (step-t0 < pulse_len))[:, None], step_off, offset)
                    noise = torch.randn(n, 6, device=dev, generator=g)*args.noise_std
                    offset = torch.where((kind == 2)[:, None], noise, offset)
                    q_ref = env._q_ref
                    c = finger_command(q_ref, raw[:, 64:70], settings.beta_fingers, close, open_)
                    c2 = (c+offset).clamp(-1., 1.)
                    applied.append((c2-c).abs()[copy > 0].mean())
                    d = c2-q_ref
                    t_ = torch.where(d >= 0, d/close, d/open_)
                    raw[:, 64:70] = torch.atanh(t_.clamp(-.999, .999))
                obs, _, terminated, truncated, _ = env.step(raw)
                if step >= t0-2:
                    row = snapshot(env, hand_ids)
                    row['done'] = (terminated | truncated).float()
                    row['end'] = torch.zeros(n, device=dev)
                    for k, v in row.items():
                        log.setdefault(k, []).append(v.float().reshape(n, -1).clone())
        data = {k: torch.stack(v).cpu() for k, v in log.items()}
        group_end = ended.reshape(G, K).any(1)
        data['group_end'] = group_end.cpu()
        data['bps'] = env._bps_features.float().cpu()
        data['asset'] = env._object_asset_index_per_env.cpu()
        torch.save(data, args.output/f'round_{rnd:03d}.pt')
        stats['groups_with_end'] += int(group_end.sum())
        stats['perturbation_abs_mean'] += float(torch.stack(applied).mean())/args.rounds
        print(f'COUNTERFACTUAL_ROUND {rnd} rows {data["done"].shape[0]} groups_with_end {int(group_end.sum())} '
              f'perturbation {float(torch.stack(applied).mean()):.4f} wall {time.time()-t_start:.0f}s', flush=True)
    stats['wall_s'] = round(time.time()-t_start, 1)
    stats['dt'] = dt
    return dict(stats=stats)


if __name__ == '__main__':
    main()
