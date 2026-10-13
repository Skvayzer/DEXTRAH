#!/usr/bin/env python3
"""mjlab port: smoke test + contact-capacity measurement (inference only).

Builds the mjlab task with generous contact buffers, runs either a passive
robot (zero latent residual, open fingers) or a checkpoint's deterministic
leader, and records per world and step: active contacts, constraint rows
(nefc), MuJoCo Warp overflow bits, NaNs, GPU memory and throughput. The
recommended nconmax/njmax are the observed maxima times a safety factor.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback


def build_policy(env, checkpoint, device):
    import torch
    from rl_games.algos_torch import model_builder
    from dextrah_lab.g1_adept.touch_continuation import load_yaml
    from dextrah_lab.wholebody.sapg_network import load_touch_sources, register_sonic_models
    run = Path(checkpoint).parent.parent
    source, critic = load_touch_sources(device)
    register_sonic_models(source, critic, env._sonic_reference, env._body_lower[0], env._body_upper[0], frozen=True)
    params = load_yaml(run/'params/agent.yaml')['params']
    model = model_builder.ModelBuilder().load(params).build(dict(
        actions_num=70, input_shape=(1243+32,), num_seqs=env.num_envs, value_size=1,
        normalize_value=True, normalize_input=True, type='extra_param',
        coef_ids=source.a2c_network.param_ids, coef_id_idx=1243)).to(device)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    payload = payload[0] if 0 in payload else payload
    model.load_state_dict(payload['model'], strict=True)
    model.eval().requires_grad_(False)
    return model, int(payload['epoch'])


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an allocated Slurm step')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=1200)
    p.add_argument('--seconds', type=float, default=10.)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--nconmax', type=int, default=256)
    p.add_argument('--njmax', type=int, default=2000)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.output.exists() or args.num_envs % 6:
        raise ValueError('Fresh output and a multiple of six environments')
    args.output.mkdir(parents=True)
    report = dict(completed=False, num_envs=args.num_envs, nconmax=args.nconmax, njmax=args.njmax,
                  policy=str(args.checkpoint) if args.checkpoint else 'passive')
    try:
        import numpy as np
        import torch
        import warp as wp
        from dextrah_lab.mj.factory import make_mj_env
        torch.set_num_threads(4)
        t0 = time.monotonic()
        env = make_mj_env(args.num_envs, args.device, nconmax=args.nconmax, njmax=args.njmax)
        report['build_s'] = time.monotonic()-t0
        n = env.num_envs
        obs, _ = env.reset()
        report['first_reset_s'] = time.monotonic()-t0
        model, states = None, None
        if args.checkpoint:
            model, report['checkpoint_epoch'] = build_policy(env, args.checkpoint, args.device)
            states = tuple(x.to(args.device) for x in model.get_default_rnn_state())
        passive = torch.zeros(n, 70, device=args.device)
        passive[:, 64:] = -1.
        d = env.sim.wp_data
        naconmax = d.naconmax
        overflow = torch.zeros(n, dtype=torch.long, device=args.device)
        per_world_max = torch.zeros(n, dtype=torch.long, device=args.device)
        nefc_max = torch.zeros(n, dtype=torch.long, device=args.device)
        totals, per_world_hist = [], torch.zeros(args.nconmax+2, dtype=torch.long, device=args.device)
        hits = falls = episodes = drops = 0
        touch_any = 0.
        steps = round(args.seconds/env.step_dt)
        torch.cuda.synchronize(); start = time.monotonic()
        with torch.inference_mode():
            for step in range(steps):
                if model is not None:
                    wrapped = torch.cat((obs['policy'], obs['policy'].new_zeros(n, 1)), -1)
                    out = model(dict(obs=wrapped, is_train=False, prev_actions=None, rnn_states=states))
                    states = out['rnn_states']
                    action = out['mus']
                else:
                    action = passive
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated | truncated
                if states is not None:
                    for s in states:
                        s[:, done, :] = 0
                final = info['episode_final']
                episodes += int(done.sum()); hits += int(final['successes'][done].sum())
                falls += int((done & final['robot_fall'].bool()).sum())
                drops += int((done & final['done_fall'].bool()).sum())
                touch_any += float((env.touch_raw[..., 0] > 0).any(-1).float().mean())
                # contact statistics of the last physics step
                nacon = wp.to_torch(d.nacon)
                world = wp.to_torch(d.contact.worldid).long()
                active = torch.arange(naconmax, device=args.device) < nacon
                counts = torch.bincount(torch.where(active, world, n), minlength=n+1)[:n]
                per_world_max = torch.maximum(per_world_max, counts)
                per_world_hist += torch.bincount(counts.clamp_max(args.nconmax+1), minlength=args.nconmax+2)
                nefc_max = torch.maximum(nefc_max, wp.to_torch(d.nefc).long())
                overflow |= wp.to_torch(d.overflow).long()
                totals.append(nacon)
                if not torch.isfinite(obs['policy']).all():
                    raise RuntimeError(f'Non-finite observation at step {step}')
        torch.cuda.synchronize()
        wall = time.monotonic()-start
        totals = torch.cat(totals).cpu().numpy()
        free, total = torch.cuda.mem_get_info(args.device)
        cdf = per_world_hist.cpu().numpy().cumsum()/per_world_hist.sum().item()
        q = lambda x: int(np.searchsorted(cdf, x))
        report.update(completed=True, steps=steps, wall_s=wall, env_steps_per_s=steps*n/wall,
                      policy_steps_per_s=steps/wall, naconmax=naconmax,
                      max_total_contacts=int(totals.max()), mean_total_contacts=float(totals.mean()),
                      contacts_per_world=dict(max=int(per_world_max.max()), p50=q(.5), p99=q(.99), p999=q(.999),
                                              p99_of_world_max=float(per_world_max.float().quantile(.99))),
                      nefc_per_world=dict(max=int(nefc_max.max()), p99=float(nefc_max.float().quantile(.99))),
                      overflow_worlds=int((overflow > 0).sum()), overflow_bits=sorted(set(overflow[overflow > 0].tolist())),
                      gpu_used_gib=(total-free)/2**30, torch_peak_gib=torch.cuda.max_memory_allocated(args.device)/2**30,
                      episodes=episodes, goal_hits=hits, robot_falls=falls, object_drops=drops,
                      fingertip_touch_fraction=touch_any/steps,
                      recommended=dict(nconmax=int(np.ceil(1.5*max(q(.999), 1))), njmax=int(np.ceil(1.25*int(nefc_max.max())))))
        print('MJ_CAPACITY '+json.dumps(report), flush=True)
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        (args.output/'capacity.json').write_text(json.dumps(report, indent=2))
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)


if __name__ == '__main__':
    main()
