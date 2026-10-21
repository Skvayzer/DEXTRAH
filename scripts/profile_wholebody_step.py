#!/usr/bin/env python3
"""Profile the whole-body (frozen SONIC) mjlab env step with a trained policy.

Times each part of env.step (synchronized per part, for the breakdown) and the
natural, unsynchronized step rate; reports GPU memory after the env is built.
Variants via flags: --njmax/--nconmax, --tf32, --no-nan-capture.
"""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a Slurm GPU allocation')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=22080)
    p.add_argument('--nconmax', type=int, default=64)
    p.add_argument('--njmax', type=int, default=300)
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--tf32', action='store_true')
    p.add_argument('--nan-capture', action='store_true')
    p.add_argument('--label', default='')
    p.add_argument('--kernel-table', action='store_true', help='print the top CUDA kernels (torch.profiler, 3 steps)')
    p.add_argument('--fast-step', action='store_true', help='env.enable_fast_step()')
    p.add_argument('--sgs-state', type=Path, help="enable SGS with the run's settings and this state (as in training)")
    p.add_argument('--drop-forward-graph', action='store_true', help='report memory freed by dropping the sim.forward CUDA graph')
    args = p.parse_args()
    import torch
    from dextrah_lab.mj.compat import install
    install()
    from dextrah_lab.mj.factory import make_mj_env
    from eval_wholebody_checkpoint import build_policy
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = 'cuda:0'
    run = args.checkpoint.resolve().parent.parent
    contract = json.loads((run/'task_contract.json').read_text())
    free0 = torch.cuda.mem_get_info()[0]
    env = make_mj_env(args.num_envs, device, nconmax=args.nconmax, njmax=args.njmax, run=run,
                      spawn_on_table=contract['body_termination']['spawn_on_table'])
    torch.cuda.synchronize()
    env_gib = (free0-torch.cuda.mem_get_info()[0])/2**30
    if args.fast_step:
        env.enable_fast_step()
    if args.nan_capture:
        env.enable_nan_capture(Path(os.environ.get('TMPDIR', '/tmp'))/f'nan_{os.getpid()}')
    if args.sgs_state:
        from dextrah_lab.sgs.mj_task import SgsSettings
        env.enable_sgs(SgsSettings(**contract['sgs']['settings']),
                       torch.load(args.sgs_state, map_location='cpu', weights_only=False))
    model, _ = build_policy(env, args.checkpoint, contract, device)
    timers = {}

    def timed(obj, name, key):
        fn = getattr(obj, name)

        def wrapper(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            timers[key] = timers.get(key, 0.)+time.perf_counter()-t
            return out
        setattr(obj, name, wrapper)
        return fn

    obs, _ = env.reset()
    states = tuple(s.to(device) for s in model.get_default_rnn_state())
    coef = torch.zeros(env.num_envs, 1, device=device)

    def run_steps(n):
        nonlocal obs, states
        for _ in range(n):
            out = model(dict(obs=torch.cat((obs['policy'], coef), -1), is_train=False, prev_actions=None, rnn_states=states))
            states = out['rnn_states']
            obs, _, term, trunc, _ = env.step(out['actions'])
            done = term | trunc
            for s in states:
                s[:, done, :] = 0
    with torch.inference_mode():
        run_steps(10)                                   # warm-up (graph capture, allocator)
        torch.cuda.synchronize()
        t = time.perf_counter()
        run_steps(args.steps)
        torch.cuda.synchronize()
        natural = (time.perf_counter()-t)/args.steps
        originals = [(env, n, timed(env, n, n)) for n in ('_pre_physics_step', '_apply_action', '_after_physics_step',
                                                          '_get_dones', '_get_rewards', '_reset_idx', '_get_observations')]
        originals.append((env.sim, 'step', timed(env.sim, 'step', 'physics')))
        originals.append((env.sim, 'forward', timed(env.sim, 'forward', 'sim_forward')))
        originals.append((env, '_refresh_kinematics', timed(env, '_refresh_kinematics', 'refresh_kinematics')))
        if env._sgs is not None:
            for name in ('after_step', 'before_reset', 'after_reset'):
                originals.append((env._sgs, name, timed(env._sgs, name, 'sgs_'+name)))
        originals.append((env, '_write_data_to_sim', timed(env, '_write_data_to_sim', 'write_data_to_sim')))
        originals.append((env, '_wholebody_observation', timed(env, '_wholebody_observation', 'wholebody_observation')))
        originals.append((env.contacts, 'read', timed(env.contacts, 'read', 'contacts_read')))
        if env._nan_capture is not None:
            originals.append((env._nan_capture, 'after_step', timed(env._nan_capture, 'after_step', 'nan_check')))
        originals.append((env._sonic_reference, 'reference_tokens', timed(env._sonic_reference, 'reference_tokens', 'sonic_tokens')))
        originals.append((env._sonic_reference, 'forward', timed(env._sonic_reference, 'forward', 'sonic_decode')))
        originals.append((env._sonic_reference, 'reference_tokens_and_latent',
                          timed(env._sonic_reference, 'reference_tokens_and_latent', 'sonic_tokens_latent')))
        originals.append((env._sonic_reference, 'decode_latent', timed(env._sonic_reference, 'decode_latent', 'sonic_decode_latent')))
        fwd = model.forward

        def model_wrapper(*a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fwd(*a, **k)
            torch.cuda.synchronize()
            timers['policy_forward'] = timers.get('policy_forward', 0.)+time.perf_counter()-t0
            return out
        model.forward = model_wrapper
        torch.cuda.synchronize()
        t = time.perf_counter()
        run_steps(args.steps)
        torch.cuda.synchronize()
        synced = (time.perf_counter()-t)/args.steps
        if args.kernel_table:
            for obj, name, fn in originals:
                setattr(obj, name, fn)
            model.forward = fwd
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                run_steps(3)
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=40, max_name_column_width=70), flush=True)
    memory = {}
    if args.drop_forward_graph:
        import gc
        import warp as wp
        torch.cuda.synchronize()
        memory['free_before_gib'] = torch.cuda.mem_get_info()[0]/2**30
        memory['mempool_release_threshold'] = wp.get_mempool_release_threshold('cuda:0')
        memory['mempool_used_gib'] = wp.get_mempool_used_mem_current('cuda:0')/2**30
        env.sim.forward_graph = None
        gc.collect()
        wp.synchronize()
        memory['free_after_drop_gib'] = torch.cuda.mem_get_info()[0]/2**30
        memory['mempool_used_after_drop_gib'] = wp.get_mempool_used_mem_current('cuda:0')/2**30
        wp.set_mempool_release_threshold('cuda:0', 0)
        wp.synchronize()
        memory['free_after_release_gib'] = torch.cuda.mem_get_info()[0]/2**30
    breakdown = {k: round(1e3*v/args.steps, 2) for k, v in sorted(timers.items(), key=lambda x: -x[1])}
    report = dict(label=args.label, envs=env.num_envs, njmax=args.njmax, nconmax=args.nconmax, tf32=args.tf32,
                  nan_capture=args.nan_capture, fast_step=args.fast_step, sgs=bool(args.sgs_state),
                  env_build_gib=round(env_gib, 2), memory={k: round(v, 2) if isinstance(v, float) else v for k, v in memory.items()},
                  natural_ms_per_step=round(1e3*natural, 1), env_steps_per_s=round(env.num_envs/natural),
                  synced_ms_per_step=round(1e3*synced, 1), breakdown_ms_per_step=breakdown,
                  peak_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30, 2),
                  device_free_gib=round(torch.cuda.mem_get_info()[0]/2**30, 2))
    print('PROFILE '+json.dumps(report), flush=True)
    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
