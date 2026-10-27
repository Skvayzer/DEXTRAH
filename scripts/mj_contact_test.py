#!/usr/bin/env python3
"""Contact-model test for the carry expert (inference only; no training, no task change).

Hypothesis (research review, 2026-10-27): every robot / object / world geom uses MuJoCo condim 3
(sliding friction only; the torsional coefficient 0.005 is inert), so a one-pad + palm grasp can spin
freely about the line through its contacts. Held objects turn ~2.8 rad/s relative to the palm
(median, job 1559) and carries are lost at ~2.8-3.3 %/s.

For each contact setting (condim, torsional coefficient) the carry env is built with those contact
parameters on every geom (dextrah_lab/mj/robot.ContactParams), the carry expert C runs with
deterministic mean actions at the given curriculum level, and the script reports for held objects
(within 3 cm of the hold, after the warm-up): object-vs-palm angular and linear speed, standing
(settle / final) vs walking; drops per held second (hazard); C's exact success rates.

Slip diagnostic (carry v2 reward review, 2026-10-27): percentiles of the raw, 0.1 s and 0.3 s smoothed
(exponential average) relative speeds for held objects in the secure phase (table present), walking
and the final stand; and net rotation vs path: every 0.5 s, for objects held throughout, the angle
between the object's palm-relative orientation at the window start and end vs the integral of
|w_rel| over the window. Vibration: a large integral with little net rotation. Held = supported by
the hand (carry v2) or within 3 cm of the hold (v1). Optional solimp (dmin, dmax) per setting.
"""
import argparse
import json
import math
import os
from pathlib import Path
import traceback


def run_setting(args, condim, torsional, solimp=None):
    import numpy as np
    import torch
    from isaaclab.utils.math import quat_conjugate, quat_mul
    from dextrah_lab.mj.carry_env import G1CarryMjEnv
    from dextrah_lab.mj.env import load_run_cfg
    from dextrah_lab.mj.factory import DEFAULT_MUJOCO, WORKSPACE
    from dextrah_lab.mj.robot import ContactParams
    from dextrah_lab.mj.scene import SceneParams
    from dextrah_lab.sgs.mj_task import SgsSettings
    from dextrah_lab.wholebody.carry_compose import CarrySettings, ClipSequencer
    from dextrah_lab.wholebody.sonic import FrozenSonic
    from mj_table_to_table import build_policy_dims
    device = 'cuda:0'
    run = args.checkpoint.resolve().parent.parent
    contract = json.loads((run/'task_contract.json').read_text())
    cm = contract['carry_mjlab']
    raw_settings = {k: tuple(v) if isinstance(v, list) else v for k, v in cm['settings'].items()}
    if args.no_disturbance:
        raw_settings.update(disturbance=0., push_probability=0.)
    settings = CarrySettings(**raw_settings)
    v2 = settings.version == 2
    task_run = Path(contract['task_config_source']).parent.parent
    task_contract = json.loads((task_run/'task_contract.json').read_text())
    cfg = load_run_cfg(task_run)
    cfg.seed = args.seed
    cfg.sonic_body.spawn_on_table = task_contract['body_termination']['spawn_on_table']
    holding = Path(cm['holding_state'])
    held_contract = json.loads((holding.parent.parent/'task_contract.json').read_text())
    sonic = FrozenSonic(WORKSPACE/'GRAIL', WORKSPACE/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', device)
    base = ContactParams()
    contact = dict(condim=condim, torsional=torsional)
    if solimp is not None:
        contact['solimp'] = (solimp[0], solimp[1])+tuple(base.solimp[2:])
    params = SceneParams(contact=ContactParams(**contact))
    njmax = args.njmax if condim == 3 else int(args.njmax*condim/3)
    env = G1CarryMjEnv(cfg, sonic=sonic, carry=settings, holding_state=holding,
                       held_settings=SgsSettings(**dict(held_contract['sgs']['settings'], eval_pool_size=0)),
                       clips_path=Path(cm['clips']), num_envs=args.num_envs, device=device, nconmax=args.nconmax,
                       njmax=njmax, mujoco_cfg=DEFAULT_MUJOCO, seed=args.seed, scene_params=params)
    env._curriculum.level = args.level
    env._seq.set_level(args.level)
    model, _ = build_policy_dims(env, args.checkpoint, contract, device, env.cfg.observation_space, env.hand_surface_dim)
    drops = {'stand': 0, 'walk': 0}
    record = env._carry_record

    def counting_record(ids):
        # Count drops by phase BEFORE the env resets these episodes (the reset clears the drop flag).
        ph = env._seq.phase(env._carry_t[ids].float()*env.step_dt, ids)
        d = env._dropped[ids]
        drops['walk'] += int((d & (ph == ClipSequencer.WALK)).sum())
        drops['stand'] += int((d & ((ph == ClipSequencer.SETTLE) | (ph == ClipSequencer.FINAL))).sum())
        return record(ids)
    env._carry_record = counting_record
    obs, _ = env.reset()
    states = tuple(s.to(device) for s in model.get_default_rnn_state())
    leader = torch.zeros(env.num_envs, 1, device=device)
    stand_ang, walk_ang, stand_lin, walk_lin = [], [], [], []
    held_seconds = {'stand': 0., 'walk': 0.}
    episodes = successes = falls = 0
    dt, n = env.step_dt, env.num_envs
    buckets = ('secure', 'walk', 'final')
    signals = ('ang_raw', 'ang_0.1s', 'ang_0.3s', 'lin_raw', 'lin_0.1s', 'lin_0.3s')
    samples = {b: {k: [] for k in signals} for b in buckets}
    ema1, ema3 = torch.zeros(n, 6, device=device), torch.zeros(n, 6, device=device)
    W = round(.5/dt)
    q_prev = torch.zeros(n, 4, device=device); q_prev[:, 0] = 1.
    integral = torch.zeros(n, device=device)
    held_all = torch.zeros(n, dtype=torch.bool, device=device)
    win_bucket = torch.zeros(n, dtype=torch.long, device=device)
    windows = {b: dict(net=[], path=[]) for b in buckets}

    def bucket_of(ph):
        return torch.where(ph == ClipSequencer.WALK, 1, torch.where(ph == ClipSequencer.FINAL, 2, 0))

    def rel_quat():
        _, palm = env._palm()
        return quat_mul(quat_conjugate(palm[:, 3:7]), env.object.data.root_quat_w)
    with torch.inference_mode():
        for step in range(round(args.seconds/dt)):
            out = model(dict(obs=torch.cat((obs['policy'], leader), -1), is_train=False, prev_actions=None, rnn_states=states))
            states = out['rnn_states']
            phase_before = env._seq.phase(env._episode_time())
            obs, _, terminated, truncated, _ = env.step(out['mus'])
            done = terminated | truncated
            for s_ in states:
                s_[:, done, :] = 0
            t = env._episode_time()
            rel = env._rel_vel_palm
            ema1 += min(1., dt/.1)*(rel-ema1)
            ema3 += min(1., dt/.3)*(rel-ema3)
            ema1[done] = 0.; ema3[done] = 0.
            hold = env._supported if v2 else env._offset_error < .03
            held = hold & (t > settings.warmup_s+.1) & ~done
            walking = phase_before == ClipSequencer.WALK
            standing = (phase_before == ClipSequencer.SETTLE) | (phase_before == ClipSequencer.FINAL)
            ang, lin = rel[:, 3:].norm(dim=-1), rel[:, :3].norm(dim=-1)
            for mask, a_list, l_list, key in ((held & standing, stand_ang, stand_lin, 'stand'), (held & walking, walk_ang, walk_lin, 'walk')):
                if bool(mask.any()):
                    a_list.append(ang[mask].cpu()); l_list.append(lin[mask].cpu())
                held_seconds[key] += float(mask.sum())*dt
            b_now = bucket_of(phase_before)
            if step % 2 == 0:
                values = dict(zip(signals, (ang, ema1[:, 3:].norm(dim=-1), ema3[:, 3:].norm(dim=-1),
                                            lin, ema1[:, :3].norm(dim=-1), ema3[:, :3].norm(dim=-1))))
                for i, b in enumerate(buckets):
                    m = held & (b_now == i)
                    if bool(m.any()):
                        for k in signals:
                            samples[b][k].append(values[k][m].cpu())
            # net rotation vs path over 0.5 s windows
            integral += ang*dt
            held_all &= held & (b_now == win_bucket)
            if step % W == W-1:
                q_now = rel_quat()
                dot = (q_now*q_prev).sum(-1).abs().clamp(max=1.)
                net = 2*torch.acos(dot)
                for i, b in enumerate(buckets):
                    m = held_all & (win_bucket == i)
                    if bool(m.any()):
                        windows[b]['net'].append(net[m].cpu()); windows[b]['path'].append(integral[m].cpu())
                q_prev, integral = q_now, torch.zeros_like(integral)
                held_all, win_bucket = held.clone(), b_now.clone()
            falls += int((done & env._body_fallen).sum())
            episodes += int(done.sum())
    q = lambda xs, p: float(torch.cat(xs).quantile(p)) if xs else float('nan')
    rates = {}
    for name, est in (('train', env._sgs.train_exact), ('eval', env._sgs.eval_exact)):
        r, n = est.rate(env._env_steps)
        rates[name] = dict(success=None if math.isnan(r) else round(r, 4), episodes=int(n))
    res = dict(condim=condim, torsional=torsional, level=args.level, seconds=args.seconds, njmax=njmax,
               held_angular_rads=dict(stand_median=q(stand_ang, .5), stand_p90=q(stand_ang, .9),
                                      walk_median=q(walk_ang, .5), walk_p90=q(walk_ang, .9)),
               held_linear_ms=dict(stand_median=q(stand_lin, .5), walk_median=q(walk_lin, .5)),
               drop_hazard_per_s=dict(stand=drops['stand']/max(held_seconds['stand'], 1e-9),
                                      walk=drops['walk']/max(held_seconds['walk'], 1e-9)),
               held_seconds=held_seconds, drops=drops, falls=falls, episodes_ended=episodes, exact=rates,
               outcomes={k: float(v.mean()) for k, v in env._sgs.train_stats.items()},
               solimp=list(params.contact.solimp), disturbance=settings.disturbance, slip_scales=dict(
                   angular=getattr(settings, 'slip_w_scale', None), linear=getattr(settings, 'slip_v_scale', None)))
    pct = (.1, .25, .5, .75, .9)
    res['slip_percentiles'] = {b: {k: ([round(float(torch.cat(v).quantile(p_)), 4) for p_ in pct] if v else None)
                                   for k, v in samples[b].items()} for b in buckets}
    res['slip_percentiles']['levels'] = list(pct)
    res['net_rotation_0.5s'] = {}
    for b in buckets:
        if windows[b]['net']:
            net, path = torch.cat(windows[b]['net']), torch.cat(windows[b]['path'])
            res['net_rotation_0.5s'][b] = dict(windows=int(len(net)), net_rate_median=round(float(net.median())/.5, 4),
                                               path_rate_median=round(float(path.median())/.5, 4),
                                               net_over_path_median=round(float((net/path.clamp_min(1e-6)).median()), 4),
                                               net_rate_p90=round(float(net.quantile(.9))/.5, 4))
    print('CONTACT_TEST '+json.dumps(res), flush=True)
    del env
    torch.cuda.empty_cache()
    return res


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside a Slurm GPU allocation')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True, help='carry expert checkpoint')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--settings', default='3:0.005/4:0.005/4:0.02/6:0.02',
                   help='condim:torsional[:solimp_dmin:solimp_dmax] list, "/"-separated (sbatch --export splits on commas)')
    p.add_argument('--no-disturbance', action='store_true', help='pushes off (carry v2 disturbance 0, v1 wrist pushes 0)')
    p.add_argument('--level', type=int, default=2)
    p.add_argument('--seconds', type=float, default=60.)
    p.add_argument('--num-envs', type=int, default=1200)
    p.add_argument('--nconmax', type=int, default=64)
    p.add_argument('--njmax', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results, report = [], dict(completed=False)
    try:
        from dextrah_lab.mj.compat import install
        install()
        for item in args.settings.replace(',', '/').split('/'):
            f = item.split(':')
            solimp = (float(f[2]), float(f[3])) if len(f) == 4 else None
            results.append(run_setting(args, int(f[0]), float(f[1]), solimp))
            args.output.write_text(json.dumps(results, indent=2))
        report.update(completed=True)
        print('CONTACT_TEST_COMPLETE', flush=True)
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        args.output.write_text(json.dumps(dict(results=results, **report), indent=2))
        import sys
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if report['completed'] else 1)


if __name__ == '__main__':
    main()
