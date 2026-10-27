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
"""
import argparse
import json
import math
import os
from pathlib import Path
import traceback


def run_setting(args, condim, torsional):
    import numpy as np
    import torch
    from dextrah_lab.mj.carry_env import CONTEXT_DIM, G1CarryMjEnv
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
    settings = CarrySettings(**{k: tuple(v) if isinstance(v, list) else v for k, v in cm['settings'].items()})
    task_run = Path(contract['task_config_source']).parent.parent
    task_contract = json.loads((task_run/'task_contract.json').read_text())
    cfg = load_run_cfg(task_run)
    cfg.seed = args.seed
    cfg.sonic_body.spawn_on_table = task_contract['body_termination']['spawn_on_table']
    holding = Path(cm['holding_state'])
    held_contract = json.loads((holding.parent.parent/'task_contract.json').read_text())
    sonic = FrozenSonic(WORKSPACE/'GRAIL', WORKSPACE/'G1-SONIC-models/checkpoint/SONIC/models/sonic_manipulation_base', device)
    params = SceneParams(contact=ContactParams(condim=condim, torsional=torsional))
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
    with torch.inference_mode():
        for step in range(round(args.seconds/env.step_dt)):
            out = model(dict(obs=torch.cat((obs['policy'], leader), -1), is_train=False, prev_actions=None, rnn_states=states))
            states = out['rnn_states']
            phase_before = env._seq.phase(env._episode_time())
            obs, _, terminated, truncated, _ = env.step(out['mus'])
            done = terminated | truncated
            for s_ in states:
                s_[:, done, :] = 0
            t = env._episode_time()
            held = (env._offset_error < .03) & (t > settings.warmup_s+.1) & ~done
            walking = phase_before == ClipSequencer.WALK
            standing = (phase_before == ClipSequencer.SETTLE) | (phase_before == ClipSequencer.FINAL)
            ang, lin = env._rel_vel_palm[:, 3:].norm(dim=-1), env._rel_vel_palm[:, :3].norm(dim=-1)
            for mask, a_list, l_list, key in ((held & standing, stand_ang, stand_lin, 'stand'), (held & walking, walk_ang, walk_lin, 'walk')):
                if bool(mask.any()):
                    a_list.append(ang[mask].cpu()); l_list.append(lin[mask].cpu())
                held_seconds[key] += float(mask.sum())*env.step_dt
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
               outcomes={k: float(v.mean()) for k, v in env._sgs.train_stats.items()})
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
    p.add_argument('--settings', default='3:0.005/4:0.005/4:0.02/6:0.02', help='condim:torsional list, "/"-separated (sbatch --export splits on commas)')
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
            c, tors = item.split(':')
            results.append(run_setting(args, int(c), float(tors)))
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
