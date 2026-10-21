#!/usr/bin/env python3
"""Check the opt-in fast env step (G1SonicTouchMjEnv.enable_fast_step) against the
default path on the same states, while a trained policy acts:

1. kinematics refresh vs sim.forward(): the fields the task reads must agree within
   the simulator's own run-to-run noise (subtree COM is accumulated with float atomics,
   so two sim.forward() calls on the same state already differ in the last bits; a
   control measures that); the fields only sim.forward() updates are listed (the task
   must not read them between steps);
2. active-only contact reader vs the whole contact pool (every output);
3. SONIC fast path vs the full controller: reference tokens and decoded body action
   (bitwise).

The simulated trajectory follows the default path (sim.forward(), whole pool, full
controller); each fast result is computed on the same state and discarded.
Prints FAST_STEP_CHECK {...}; exits non-zero on a mismatch.
"""
import argparse
import json
import os
from pathlib import Path
import sys

# Fields read between steps (mjlab entity data accessors and this task).
READ_FIELDS = ('xpos', 'xquat', 'xmat', 'xipos', 'ximat', 'geom_xpos', 'geom_xmat', 'site_xpos',
               'site_xmat', 'subtree_com', 'cdof', 'cvel', 'cdof_dot', 'cinert', 'qpos', 'qvel')
# Accessors that read full-forward-only fields; the task must not use them (checked by grep in tests).
FULL_FORWARD_ONLY_ACCESSORS = ('joint_acc', 'actuator_force', 'applied_torque')


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a Slurm GPU allocation')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=22080)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--all-field-steps', type=int, default=8, help='refreshes on which every Data field is compared')
    p.add_argument('--contact-rtol', type=float, default=1e-5)
    p.add_argument('--refresh-rtol', type=float, default=1e-5)
    p.add_argument('--control-steps', type=int, default=20, help='refreshes with a sim.forward() vs sim.forward() control')
    args = p.parse_args()
    import torch
    import warp as wp
    from dextrah_lab.mj.compat import install
    install()
    import mujoco_warp as mjwarp
    from dextrah_lab.mj.factory import make_mj_env
    from dextrah_lab.mj.nan_capture import _walk
    from eval_wholebody_checkpoint import build_policy
    device = 'cuda:0'
    run = args.checkpoint.resolve().parent.parent
    contract = json.loads((run/'task_contract.json').read_text())
    env = make_mj_env(args.num_envs, device, nconmax=64, njmax=300, run=run,
                      spawn_on_table=contract['body_termination']['spawn_on_table'])
    model, _ = build_policy(env, args.checkpoint, contract, device)
    sonic = env._sonic_reference
    sonic.check_fast_path()
    env._mjwarp = mjwarp                      # _refresh_kinematics without enabling the fast step
    d = env.sim.wp_data
    report = dict(envs=env.num_envs, steps=args.steps, refresh=dict(calls=0, read_fields_unequal={}, full_forward_only=set(),
                  max_rel_diff={}, max_abs_diff={}, control_max_rel_diff={}, control_unequal={}),
                  contacts=dict(calls=0, max_abs_diff={}, max_rel_diff={}), sonic=dict(calls=0, action_unequal_envs=0,
                  action_max_abs_diff=0., tokens_unequal_envs=0))

    def fields(names=None):
        out = {}
        for name, value in _walk(d):
            if not isinstance(value, wp.array) or value.size == 0 or (names is not None and name not in names):
                continue
            try:
                out[name] = wp.to_torch(value).clone()
            except Exception:
                pass
        return out

    def rel(a, b):
        a, b = a.double(), b.double()
        diff = (a-b).abs().nan_to_num(0.)
        return float(diff.max()), float(diff.max()/b.abs().nan_to_num(0.).max().clamp_min(1e-12))

    def checked_refresh():
        r = report['refresh']
        every = r['calls'] < args.all_field_steps
        if r['calls'] < args.control_steps:
            env.sim.forward()
            first = fields(READ_FIELDS)
            env.sim.forward()
            second = fields(READ_FIELDS)
            for name in READ_FIELDS:
                if name in first and first[name].is_floating_point():
                    _, rd = rel(first[name], second[name])
                    r['control_max_rel_diff'][name] = max(r['control_max_rel_diff'].get(name, 0.), rd)
                    if not torch.equal(first[name], second[name]):
                        r['control_unequal'][name] = r['control_unequal'].get(name, 0)+1
        env._refresh_kinematics()
        fast = fields(None if every else READ_FIELDS)
        env.sim.forward()
        full = fields(None if every else READ_FIELDS)
        for name in READ_FIELDS:
            if name in fast and fast[name].is_floating_point():
                ad, rd = rel(fast[name], full[name])
                r['max_abs_diff'][name] = max(r['max_abs_diff'].get(name, 0.), ad)
                r['max_rel_diff'][name] = max(r['max_rel_diff'].get(name, 0.), rd)
        for name, value in fast.items():
            other = full[name]
            equal = bool(((value == other) | (value.isnan() & other.isnan())).all()) if value.is_floating_point() \
                else torch.equal(value, other)
            if not equal:
                if name in READ_FIELDS:
                    r['read_fields_unequal'][name] = r['read_fields_unequal'].get(name, 0)+1
                else:
                    r['full_forward_only'].add(name)
        r['calls'] += 1
    env._refresh = checked_refresh

    obs, _ = env.reset()
    reader = env.contacts
    original_read = reader.read

    def checked_read():
        c = report['contacts']
        reader.active_only = True
        fast = {k: v.clone() for k, v in original_read().items() if torch.is_tensor(v)}
        reader.active_only = False
        full = original_read()
        for k, v in fast.items():
            diff = float((v.float()-full[k].float()).abs().max()) if v.numel() else 0.
            scale = float(full[k].float().abs().max()) if v.numel() else 0.
            c['max_abs_diff'][k] = max(c['max_abs_diff'].get(k, 0.), diff)
            c['max_rel_diff'][k] = max(c['max_rel_diff'].get(k, 0.), diff/max(scale, 1e-12))
        c['calls'] += 1
        return full
    reader.read = checked_read

    original_forward = sonic.forward

    def checked_forward(proprio, reference_q, reference_qd, reference_ori6, residual=None):
        full = original_forward(proprio, reference_q, reference_qd, reference_ori6, residual)
        if residual is not None:
            s = report['sonic']
            tokens, latent = sonic.reference_tokens_and_latent(reference_q, reference_qd, reference_ori6)
            fast = sonic.decode_latent(proprio, latent, residual)
            s['action_unequal_envs'] += int((fast != full).any(-1).sum())
            s['action_max_abs_diff'] = max(s['action_max_abs_diff'], float((fast-full).abs().max()))
            s['tokens_unequal_envs'] += int((tokens != env._body_extra[:, 930:]).any(-1).sum())
            s['calls'] += 1
        return full
    sonic.forward = checked_forward

    states = tuple(s.to(device) for s in model.get_default_rnn_state())
    coef = torch.zeros(env.num_envs, 1, device=device)
    resets = 0
    with torch.inference_mode():
        for _ in range(args.steps):
            out = model(dict(obs=torch.cat((obs['policy'], coef), -1), is_train=False, prev_actions=None, rnn_states=states))
            states = out['rnn_states']
            obs, _, term, trunc, _ = env.step(out['actions'])
            done = term | trunc
            resets += int(done.sum())
            for s in states:
                s[:, done, :] = 0
    r, c, s = report['refresh'], report['contacts'], report['sonic']
    r['full_forward_only'] = sorted(r['full_forward_only'])
    report['episode_resets'] = resets
    contact_ok = all(v <= args.contact_rtol for v in c['max_rel_diff'].values())
    report['ok'] = dict(refresh=all(v <= args.refresh_rtol for v in r['max_rel_diff'].values())
                        and set(r['read_fields_unequal']) <= set(r['control_unequal'])
                        and not set(r['full_forward_only']) & set(READ_FIELDS),
                        contacts=contact_ok, sonic=s['action_unequal_envs'] == 0 and s['tokens_unequal_envs'] == 0)
    print('FAST_STEP_CHECK '+json.dumps(report), flush=True)
    sys.stdout.flush()
    os._exit(0 if all(report['ok'].values()) else 1)


if __name__ == '__main__':
    main()
