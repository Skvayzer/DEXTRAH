#!/usr/bin/env python3
"""Distribution of frozen SONIC's decoded body actions (normalized, per joint) and robot
falls under a trained frozen-SONIC policy (sampled actions, default resets).

Prints SONIC_ACTIONS {...}: per-joint |action| quantiles, share of steps above 6 / 12, and
the robot fall count, for comparing checkpoints (e.g. before / after a reward change).
"""
import argparse
import json
import os
from pathlib import Path


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a Slurm GPU allocation')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--label', default='')
    args = p.parse_args()
    import torch
    from dextrah_lab.mj.compat import install
    install()
    from dextrah_lab.mj.factory import make_mj_env
    from eval_wholebody_checkpoint import build_policy
    device = 'cuda:0'
    run = args.checkpoint.resolve().parent.parent
    if args.checkpoint.parent.name == 'best':
        run = args.checkpoint.resolve().parent.parent
    contract = json.loads((run/'task_contract.json').read_text())
    env = make_mj_env(args.num_envs, device, nconmax=64, njmax=300, run=run,
                      spawn_on_table=contract['body_termination']['spawn_on_table'])
    env.enable_fast_step()
    model, _ = build_policy(env, args.checkpoint, contract, device)
    obs, _ = env.reset()
    states = tuple(s.to(device) for s in model.get_default_rnn_state())
    coef = torch.zeros(env.num_envs, 1, device=device)
    absmax, over6, over12, falls, lifted_over12, lifted_n = None, 0, 0, 0, 0, 0
    with torch.inference_mode():
        for t in range(args.steps):
            out = model(dict(obs=torch.cat((obs['policy'], coef), -1), is_train=False, prev_actions=None, rnn_states=states))
            states = out['rnn_states']
            obs, _, term, trunc, _ = env.step(out['actions'])
            a = env._last_decoded_sonic_action.abs()                       # (n, 29) normalized body action
            absmax = a.amax(0) if absmax is None else torch.maximum(absmax, a.amax(0))
            over6 = over6+(a > 6).float().sum(0)
            over12 = over12+(a > 12).float().sum(0)
            lifted = env._lifted_object
            lifted_over12 += int(((a > 12).any(-1) & lifted).sum())
            lifted_n += int(lifted.sum())
            falls += int(env._body_fallen.sum())
            done = term | trunc
            for s in states:
                s[:, done, :] = 0
    total = args.steps*env.num_envs
    report = dict(label=args.label, checkpoint=str(args.checkpoint), envs=env.num_envs, steps=args.steps,
                  absmax_per_joint=[round(float(x), 2) for x in absmax], share_over6=[round(float(x)/total, 4) for x in over6],
                  share_over12=[round(float(x)/total, 4) for x in over12], steps_any_over12_while_lifted=round(lifted_over12/max(lifted_n, 1), 4),
                  robot_fall_steps=falls)
    print('SONIC_ACTIONS '+json.dumps(report), flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
