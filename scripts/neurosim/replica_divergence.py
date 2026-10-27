#!/usr/bin/env python3
"""Do identical carry worlds stay identical? Diagnostic for the stage-2 gradient check (2026-10-28).

gradient_check.py measured a simulator noise floor (unperturbed copies of one holding state) as large as the
effect of a finger-command perturbation: median |difference| 0.76 in log(1 + pad + palm force), 3.9 deg net
rotation, 1.4 mm displacement over 0.5 s, and the same ratio after averaging 8 jittered copies. Either the copies
are not identical (task-side randomisation left on: its overrides do not set table_keep_probability, so each copy
keeps its table with probability 0.2, and C v2 parks the table at the end of the warm-up, 0.1 s before the
perturbation) or MuJoCo Warp's contact dynamics amplify float reordering (atomics).

This restores --states holding states into --replicas worlds each and reports, after the restore and after every
control step, the spread across each state's copies (max over copies of |x - x_copy0|): robot + object qpos,
object position (mm) and orientation (deg), hand joints, log(1 + finger + palm force), the policy observation and
action, and whether the copies differ in keeping the table. Blocks in which any copy ended are left out.

--variant (one per run):
  as_check   the gradient check's overrides (no mass / COM / jitter / pushes; table kept with probability 0.2)
  no_table   + table_keep_probability 0 (every copy parks its table)
  shared     no_table + every copy executes copy 0's carry-expert action (removes the expert's feedback;
             frozen SONIC still closes its loop on the body state)
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
    p.add_argument('--output', type=Path, required=True, help='report .json')
    p.add_argument('--variant', choices=('as_check', 'no_table', 'shared'), required=True)
    p.add_argument('--states', type=int, default=32)
    p.add_argument('--replicas', type=int, default=8)
    p.add_argument('--seconds', type=float, default=1.2, help='from the restore (the gradient check ends at 1.1 s)')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print('REPLICA_DIVERGENCE '+json.dumps({k: report.get(k) for k in ('completed', 'args', 'summary', 'error')}), flush=True)
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)


def run(args):
    import torch
    from dextrah_lab.neurosim.env_io import build_carry_env, hand_joint_ids
    t_start = time.time()
    S, R = args.states, args.replicas
    n = S*R                                                         # world = state * R + copy
    override = dict(mass_scale=(1., 1.), com_offset_m=0., disturbance=0., push_probability=0., carry_jitter=0.)
    if args.variant != 'as_check':
        override['table_keep_probability'] = 0.
    env, policy, contract, settings = build_carry_env(args.checkpoint, n, settings_override=override, seed=args.seed)
    dev, dt = env.device, env.step_dt
    g = torch.Generator().manual_seed(args.seed)
    chosen = torch.randperm(env._sgs.configs.numel(), generator=g)[:S].to(dev)
    assigned = chosen.repeat_interleave(R)
    env._sgs.eval_mask[:] = False
    env._sgs.sampler.sample = lambda k, generator=None: assigned[:k]
    hand_ids = hand_joint_ids(env)
    ended = torch.zeros(n, dtype=torch.bool, device=dev)
    record = env._carry_record

    def recording(ids):
        ended[ids] = True
        return record(ids)
    env._carry_record = recording

    def spread(x):
        """(n, ...) -> (S,): max over a state's copies and components of |x - x_copy0|."""
        x = x.reshape(S, R, -1).double()
        return (x-x[:, :1]).abs().amax(dim=(1, 2))

    def angle_spread(q):
        q = q.reshape(S, R, 4).double()
        dot = (q*q[:, :1]).sum(-1).abs().clamp(max=1.)
        return torch.rad2deg(2*torch.acos(dot)).amax(1)

    def stats(v, live):
        v = v[live]
        if not v.numel():
            return None
        return dict(median=float(v.median()), p90=float(torch.quantile(v.float(), .9)), max=float(v.max()),
                    zero_share=round(float((v == 0).float().mean()), 4))

    rows = []

    def measure(step, obs, action, action_policy):
        live = ~ended.reshape(S, R).any(1)
        c = getattr(env, '_last_contacts', None)
        row = dict(step=step, t=round(step*dt, 4), blocks_live=int(live.sum()),
                   qpos=stats(spread(env.sim.data.qpos), live),
                   object_mm=stats(1e3*spread(env.object.free_qpos[:, :3]), live),
                   object_deg=stats(angle_spread(env.object.data.root_quat_w), live),
                   hand_q=stats(spread(env.robot.data.joint_pos[:, hand_ids]), live),
                   obs=stats(spread(obs['policy']), live),
                   keep_table_mismatch=round(float((spread(env._keep_table.float()) > 0).float()[live].mean()), 4) if live.any() else None)
        if c is not None:
            force = c['finger_object_force_l1'].reshape(n, -1).sum(-1)+c['palm_object_force_l1'].reshape(n, -1).sum(-1)
            row['log_force'] = stats(spread(torch.log1p(force)), live)
        if action is not None:
            row['action'] = stats(spread(action), live)
            row['action_policy'] = stats(spread(action_policy), live)
        rows.append(row)

    obs, _ = env.reset()
    states = tuple(s.to(dev) for s in policy.get_default_rnn_state())
    leader = torch.zeros(n, 1, device=dev)
    steps = round(args.seconds/dt)
    with torch.inference_mode():
        measure(0, obs, None, None)
        for step in range(1, steps+1):
            out = policy(dict(obs=torch.cat((obs['policy'], leader), -1), is_train=False, prev_actions=None, rnn_states=states))
            states = out['rnn_states']
            raw = out['mus'].clone()
            act = raw
            if args.variant == 'shared':
                act = raw.reshape(S, R, -1)[:, :1].expand(S, R, raw.shape[-1]).reshape(n, -1).contiguous()
            obs, _, _, _, _ = env.step(act)
            measure(step, obs, act, raw)

    def first(key, above):
        for r in rows:
            v = r.get(key)
            if v is not None and v['max'] > above:
                return r['t']
        return None
    end = rows[-1]
    t_check = [r for r in rows if abs(r['t']-1.1) < dt/2]
    summary = dict(variant=args.variant, override={k: list(v) if isinstance(v, tuple) else v for k, v in override.items()},
                   worlds=n, states=S, replicas=R, blocks_live_end=end['blocks_live'],
                   restore=dict(qpos=rows[0]['qpos'], obs=rows[0]['obs'], keep_table_mismatch=rows[0]['keep_table_mismatch']),
                   first_t_qpos_nonzero=first('qpos', 0.), first_t_object_over_0p1mm=first('object_mm', .1),
                   first_t_object_over_1mm=first('object_mm', 1.), first_t_obs_nonzero=first('obs', 0.),
                   first_t_action_nonzero=first('action_policy', 0.),
                   at_1p1s={k: t_check[0].get(k) for k in ('qpos', 'object_mm', 'object_deg', 'hand_q', 'log_force', 'obs', 'action_policy')} if t_check else None,
                   end={k: end.get(k) for k in ('qpos', 'object_mm', 'object_deg', 'hand_q', 'log_force', 'obs', 'action_policy')})
    return dict(summary=summary, rows=rows, wall_s=round(time.time()-t_start, 1))


if __name__ == '__main__':
    main()
